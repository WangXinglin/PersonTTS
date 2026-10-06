from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    asc_posterior_confidence,
    deterministic_plurality,
    top_two_counts,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Width-5 parallel ASC with two-phase confidence and terminal-answer preference.

    Improvement over incumbent (round 3, utility 0.882, accuracy 88.5%):
    The incumbent uses 4 Qwen3-32B branches with depth-35 cap, achieving
    77.2% per-problem accuracy.  The main failure mode is:
      - 2/2 ties in 4-branch voting lead to first-encountered tie-break errors
      - 3/4 agreement on a wrong answer at phase-2 confidence 0.80

    This candidate changes:
    1. Width 5 (was 4): eliminates most tie-breaking errors, improves
       plurality robustness.  Compute scales +25% per round but earlier
       consensus offsets most of the increase (budget margin ~621k).
    2. warmup 12 (was 15): 5-branch consensus is statistically stronger;
       opening the gate 3 rounds earlier saves latency on the ~60% of
       easy problems that reach unanimity quickly.
    3. phase2_start 22 (was 25): proportionally adjusted for earlier warmup.
    4. conf_phase2 0.82 (was 0.80): with 5 branches, 4/5 agreement gives
       conf=0.891 (passes); 3/5 gives conf=0.656 (blocked).  If only 4
       branches have valid answers, 3/4 conf=0.8125 still passes (>=0.82),
       preserving the incumbent's stop behavior for that common scenario.
    5. max_rounds 35 (same): the latency budget is generous (mean margin
       +751s) but we keep the same cap to avoid compute risk.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  Model selection                                                    #
    # ------------------------------------------------------------------ #

    def _select_model(self, env) -> Optional[str]:
        """Prefer the largest available model with pool >= target width."""
        state = env.observe()
        remaining = state.get("remaining_by_model", {})
        preference = ["Qwen3-32B", "Qwen3-14B", "Qwen3-8B", "Qwen3-4B"]
        # First pass: need enough pool for full width
        for model in preference:
            if model in env.models and int(remaining.get(model, 0)) >= 5:
                return model
        # Fallback: any model with at least 2
        for model in preference:
            if model in env.models and int(remaining.get(model, 0)) >= 2:
                return model
        for model in sorted(env.models):
            if int(remaining.get(model, 0)) >= 2:
                return model
        return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _terminal_majority(
        self, terminal_answers: Dict[int, str], min_terminal: int
    ) -> Optional[str]:
        """Return the majority answer among terminal branches, if any."""
        if len(terminal_answers) < min_terminal:
            return None
        counts = Counter(terminal_answers.values())
        if not counts:
            return None
        top_answer, top_count = counts.most_common(1)[0]
        if top_count * 2 > len(terminal_answers):
            return top_answer
        return None

    # ------------------------------------------------------------------ #
    #  Solve                                                              #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)
        if model is None:
            return env.finish(None)

        state = env.observe()
        remaining = state.get("remaining_by_model", {})
        pool = int(remaining.get(model, 0))
        width = min(5, pool)
        if width < 1:
            return env.finish(None)

        # --- Tunables ------------------------------------------------- #
        warmup = 12          # rounds before ASC gate opens
        conf_phase1 = 0.95   # rounds 12-21: strict (needs near-unanimity)
        conf_phase2 = 0.82   # rounds 22+: relaxed (4/5 or 3/4 OK)
        phase2_start = 22    # round to switch to relaxed confidence
        max_rounds = 35      # hard cap: bounds worst-case latency
        min_votes = 3        # minimum valid answers for ASC check
        min_terminal = 2     # terminal branches needed for majority check
        # -------------------------------------------------------------- #

        # --- Phase 1: Parallel spawn ---------------------------------- #
        try:
            if width >= 2:
                opened = env.spawn_many(model, width)
            else:
                opened = [env.spawn(model)]
        except Exception:
            return env.finish(None)

        # Track answers: all latest and terminal-only
        answers: Dict[int, str] = {}           # bid -> latest valid answer
        terminal_answers: Dict[int, str] = {}  # bid -> terminal answer

        for b in opened:
            bid = b["branch_id"]
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                answers[bid] = ans
            if b.get("stage_terminal"):
                if ans is not None:
                    terminal_answers[bid] = ans

        # --- Phase 2: Parallel continue with depth-limited ASC -------- #
        for rnd in range(max_rounds):
            obs = env.observe()
            active: List[int] = [
                b["branch_id"]
                for b in obs["branches"]
                if b["can_continue"] and not b["pruned"]
            ]

            # Collect terminal answers from observation
            for b in obs["branches"]:
                if b.get("stage_terminal") and not b["pruned"]:
                    bid = b["branch_id"]
                    ans = valid_answer(b.get("latest_answer"))
                    if ans is not None:
                        terminal_answers[bid] = ans

            if not active:
                break

            try:
                updated = env.continue_many(active)
            except Exception:
                break

            for b in updated:
                bid = b["branch_id"]
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    answers[bid] = ans
                if b.get("stage_terminal"):
                    if ans is not None:
                        terminal_answers[bid] = ans

            actual_round = rnd + 1

            # Terminal majority check (works at any round after warmup)
            if actual_round >= warmup:
                tmaj = self._terminal_majority(terminal_answers, min_terminal)
                if tmaj is not None:
                    return env.finish(tmaj)

            # ASC confidence check (two-phase threshold)
            if actual_round >= warmup and len(answers) >= min_votes:
                conf_thresh = (
                    conf_phase1 if actual_round < phase2_start else conf_phase2
                )
                votes = list(answers.values())
                winner, top1, top2, tied = top_two_counts(votes)
                if not tied:
                    if asc_posterior_confidence(top1, top2) >= conf_thresh:
                        return env.finish(winner)

        # --- Final vote: prefer terminal answers ---------------------- #
        final_obs = env.observe()
        for b in final_obs["branches"]:
            if b.get("stage_terminal") and not b["pruned"]:
                bid = b["branch_id"]
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    terminal_answers[bid] = ans

        if len(terminal_answers) >= 2:
            votes = list(terminal_answers.values())
        elif terminal_answers:
            votes = list(terminal_answers.values()) + [
                ans for bid, ans in answers.items()
                if bid not in terminal_answers
            ]
        else:
            votes = list(answers.values())

        return env.finish(deterministic_plurality(votes) if votes else None)
