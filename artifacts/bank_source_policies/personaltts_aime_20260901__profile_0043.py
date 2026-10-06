from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import (
    ControllerPolicy,
    valid_answer,
    deterministic_plurality,
    top_two_counts,
    asc_posterior_confidence,
)


class CandidatePolicy(ControllerPolicy):
    """Parallel cohort majority-vote with latency-optimised execution.

    The baseline ASC spawns branches sequentially, giving O(width * depth)
    latency.  This policy spawns a fixed-width cohort in one parallel
    phase via spawn_many, then advances all branches together via
    continue_many.  Latency becomes O(depth * max-of-cohort), which is
    roughly O(depth) -- a ~10x reduction on the hardest problems.
    Accuracy is preserved through majority voting across 9 independent
    branches, with early stopping via ASC confidence on terminal answers.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _select_model(self, env) -> str:
        """Pick the smallest available model for best latency efficiency."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return env.models[0]

    def _check_terminal_consensus(
        self,
        terminal_answers: Dict[int, str],
        total_width: int,
    ) -> Optional[str]:
        """Return winner if terminal answers already decide the outcome."""
        votes = list(terminal_answers.values())
        if len(votes) < 2:
            return None

        counts = Counter(votes)
        winner, top_count = counts.most_common(1)[0]

        # Unstoppable majority: winner already holds more than half of all
        # branches, so no remaining branches can overturn the result.
        if top_count * 2 > total_width:
            return winner

        # High ASC posterior confidence among completed branches.
        _, top1, top2, tied = top_two_counts(votes)
        if not tied:
            if asc_posterior_confidence(top1, top2) >= 0.95:
                return winner

        return None

    # ------------------------------------------------------------------ #
    # main solve loop                                                      #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)
        width = 9

        # --- Phase 1: spawn all branches in one parallel phase ----------
        opened = env.spawn_many(model, width)
        actual_width = len(opened)

        branches: Dict[int, Dict[str, Any]] = {}
        terminal_answers: Dict[int, str] = {}
        last_valid: Dict[int, str] = {}

        for b in opened:
            bid = b["branch_id"]
            branches[bid] = b
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                last_valid[bid] = ans
            if not b["can_continue"]:
                if ans is not None:
                    terminal_answers[bid] = ans

        # --- Phase 2: parallel checkpoint advancement -------------------
        for _ in range(500):
            # Check for decisive consensus among terminal branches
            result = self._check_terminal_consensus(
                terminal_answers, actual_width
            )
            if result is not None:
                return env.finish(result)

            # Collect branches still generating
            continuable = [
                bid
                for bid, b in branches.items()
                if b["can_continue"] and not b["pruned"]
            ]
            if not continuable:
                break

            # One synchronous checkpoint round (latency = max of cohort)
            updated = env.continue_many(continuable)

            for b in updated:
                bid = b["branch_id"]
                branches[bid] = b
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    last_valid[bid] = ans
                # Record final answer when a branch completes its stage
                if not b["can_continue"]:
                    final_ans = (
                        ans if ans is not None else last_valid.get(bid)
                    )
                    if final_ans is not None:
                        terminal_answers[bid] = final_ans

        # --- Phase 3: final majority vote -------------------------------
        # One last consensus check after all branches are done
        result = self._check_terminal_consensus(
            terminal_answers, actual_width
        )
        if result is not None:
            return env.finish(result)

        # Prefer terminal answers; fall back to last-valid if needed
        votes = list(terminal_answers.values())
        if not votes:
            votes = list(last_valid.values())

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
