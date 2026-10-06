from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


# Model preference: larger models first with appropriate parallel width.
MODEL_PREFERENCE = [
    ("Qwen3-14B", 7),
    ("Qwen3-8B", 9),
    ("Qwen3-4B", 11),
]


class CandidatePolicy(ControllerPolicy):
    """Parallel completion-vote controller that fixes premature early-stopping.

    Round-0 checked for a 3-of-5 majority at every checkpoint, including the
    very first one (checkpoint 0).  Traces show many incorrect episodes with
    only 6 steps where branches agreed on the wrong *intermediate* answer
    before reasoning completed.  This caused 42.7 % of episodes to be wrong.

    Improvements
    ------------
    1. **No intermediate early stopping.**  Votes are cast only by branches
       whose generation has finished (``can_continue is False``).  Intermediate
       checkpoints are never used for the stop decision.
    2. **Wider parallel spawn** (11 branches for Qwen3-4B).  The huge margins
       on latency (batch budget 3121 s vs. current p99 ~ 1009 s) and compute
       (budget 12 M vs. avg 267 K) easily accommodate the wider cohort.
    3. **Late safe early-stop.**  Once at least ``width // 2 + 1`` completed
       branches agree, remaining branches cannot overturn the majority.
    4. **Model selection.**  If a stronger model is available, it is used with
       a narrower cohort to stay within budget while improving per-branch
       accuracy.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def select_model_width(self, env) -> Tuple[str, int]:
        """Pick the largest available model with an appropriate cohort width."""
        for name, width in MODEL_PREFERENCE:
            if name in env.models:
                return name, width
        # Fallback: first available model with moderate width
        return sorted(env.models)[0], 9

    # ------------------------------------------------------------------ #
    #  Main solve loop                                                     #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model, width = self.select_model_width(env)
        majority_needed = width // 2 + 1   # strict majority of total width

        # ---- Phase 1: parallel spawn --------------------------------- #
        opened = None
        for attempt_width in (width, max(3, width // 2)):
            try:
                opened = env.spawn_many(model, attempt_width)
                width = attempt_width
                majority_needed = width // 2 + 1
                break
            except Exception:
                continue

        if opened is None:
            # Ultimate fallback: single branch run to completion
            branch = env.spawn(model)
            while branch["can_continue"]:
                branch = env.continue_branch(branch["branch_id"])
            return env.finish(valid_answer(branch.get("latest_answer")))

        branch_map: Dict[int, Dict[str, Any]] = {}
        for b in opened:
            branch_map[b["branch_id"]] = b

        # ---- Phase 2: parallel continue until completion ------------- #
        # Key change: NO early stopping on intermediate checkpoint answers.
        # Only completed branches (can_continue == False) cast votes.
        for _ in range(500):   # safety cap (branches terminate naturally)
            continuable = [bid for bid, b in branch_map.items()
                           if b["can_continue"]]
            if not continuable:
                break

            try:
                updated = env.continue_many(continuable)
            except Exception:
                break   # budget exhausted

            for b in updated:
                branch_map[b["branch_id"]] = b

            # Late early stop: only count *completed* branches
            completed_votes: List[str] = []
            for b in branch_map.values():
                if not b["can_continue"]:
                    a = valid_answer(b.get("latest_answer"))
                    if a is not None:
                        completed_votes.append(a)

            if len(completed_votes) >= majority_needed:
                counts = Counter(completed_votes)
                best_answer, best_count = counts.most_common(1)[0]
                # Majority is irrevocable: remaining branches cannot overturn
                if best_count >= majority_needed:
                    return env.finish(best_answer)

        # ---- Phase 3: final plurality vote --------------------------- #
        final_votes: List[str] = []
        for b in branch_map.values():
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                final_votes.append(a)

        answer = deterministic_plurality(final_votes) if final_votes else None
        return env.finish(answer)
