from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Single deep Qwen3-8B branch with max_cp raised from 16 to 17.

    Round 2 used TOKENS_PER_STEP = 480 to derive max_cp = 16, achieving
    47.9 % accuracy and 103 725 mean weighted-compute against a budget of
    110 334 -- leaving 6 609 headroom.

    Going from round 1 (max_cp=14, WC=94 168) to round 2 (max_cp=16,
    WC=103 725) cost ~4 779 WC per extra checkpoint level and raised
    accuracy by ~1.5 % per level.  Extrapolating, max_cp=17 should add
    ~4 800 WC to the mean (→ ~108 500, safely under budget) and improve
    accuracy to ~49-50 %, significantly increasing batch satisfaction.

    Using TOKENS_PER_STEP = 440 yields max_cp = 17 for Qwen3-8B:
      wc_per_step = 440 * (8 / 0.6) = 5 867
      max_steps = floor(110 334 / 5 867) = 18
      max_cp = 17

    The extra checkpoint helps the ~50 % of episodes that previously hit
    the cap -- many with null answers because the model was still thinking.
    One additional reasoning step lets some of those finish and produce a
    correct answer.
    """

    NAME = "candidate"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_param_size(name: str) -> float:
        """Heuristic: extract the number preceding 'B' in a model name."""
        for part in name.replace("-", " ").replace("_", " ").split():
            upper = part.upper()
            if upper.endswith("B") and len(upper) > 1:
                try:
                    return float(upper[:-1])
                except ValueError:
                    continue
        return float("inf")

    def _select_model_and_cap(
        self, models: List[str], wc_budget: float
    ) -> tuple:
        """Choose the model that maximises expected accuracy under budget.

        Calibrated from round-2 traces: actual raw tokens per step is ~530
        at full depth.  Using TOKENS_PER_STEP = 440 accounts for the fact
        that the WC constraint is on the batch mean, not per-episode max;
        short episodes (natural termination well below cap) bring the mean
        down, providing headroom for capped episodes to run an extra step
        compared to round 2.
        """
        TOKENS_PER_STEP = 440
        MIN_BASE_SIZE = 0.6
        MIN_DEPTH = 8
        DEPTH_SATURATION = 25

        best_score = -1.0
        best_model: Optional[str] = None
        best_cap = MIN_DEPTH

        for m in models:
            ps = self._model_param_size(m)
            if ps == float("inf"):
                continue
            wc_weight = max(ps / MIN_BASE_SIZE, 1.0)
            wc_per_step = TOKENS_PER_STEP * wc_weight
            max_steps = int(wc_budget / wc_per_step)
            max_cp = max_steps - 1
            if max_cp < MIN_DEPTH:
                continue
            eff_depth = min(max_cp, DEPTH_SATURATION)
            score = ps * eff_depth / DEPTH_SATURATION
            if score > best_score:
                best_score = score
                best_model = m
                best_cap = max_cp

        if best_model is None:
            # Fallback: smallest parseable model, maximum affordable depth.
            parseable = [
                (self._model_param_size(m), m)
                for m in models
                if self._model_param_size(m) < float("inf")
            ]
            if parseable:
                ps, best_model = min(parseable)
                wc_weight = max(ps / MIN_BASE_SIZE, 1.0)
                wc_per_step = TOKENS_PER_STEP * wc_weight
                best_cap = max(int(wc_budget / wc_per_step) - 1, MIN_DEPTH)
            else:
                best_model = models[0]
                best_cap = 20

        return best_model, best_cap

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        reqs = env.requirements
        wc_budget = float(reqs["weighted_compute"]["value"])

        model, max_cp = self._select_model_and_cap(list(env.models), wc_budget)

        # Spawn a single branch and think as deep as the budget allows.
        branch = env.spawn(model)

        while (
            branch["can_continue"]
            and branch["checkpoint_index"] < max_cp
        ):
            branch = env.continue_branch(branch["branch_id"])

        # Extract the best available answer.
        answer = valid_answer(branch.get("latest_answer"))
        if answer is None:
            # Fall back to the last valid answer seen earlier.
            for past in reversed(branch.get("answer_history") or []):
                fallback = valid_answer(past)
                if fallback is not None:
                    answer = fallback
                    break

        return env.finish(answer)
