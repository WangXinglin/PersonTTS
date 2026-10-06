from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Budget-aware deeper reasoning with checkpoint self-consistency.

    Improvement over the round-0 incumbent (utility 0.989):

    The incumbent spawns one Qwen3-4B branch and stops at the FIRST
    valid answer (max 4 continues), giving 11.7% per-problem accuracy
    at avg 12,118 weighted-compute (budget is 25,831).

    Key insight from trace analysis: the first valid answer is an
    intermediate checkpoint result, not the model's final conclusion.
    Short problems (terminal at checkpoint 2-5) have a much more
    reliable terminal answer, and long problems benefit from a few
    extra checkpoint steps.

    This policy:
      1. Derives a safe max-continues cap from the WC budget (~5).
      2. Always runs the branch to stage terminal OR the cap -- never
         stops early at the first answer.
      3. Collects every valid answer across checkpoints and returns
         the deterministic plurality vote (self-consistency).

    For short problems (terminal <= 5 checkpoints), the branch reaches
    terminal, using the model's final refined answer instead of an
    early intermediate.  For long problems, the extra 1-2 checkpoints
    beyond the incumbent's cap provide incremental improvement.

    Estimated mean weighted-compute ~20,000 per problem (safely under
    the 25,831 budget).  Latency stays far below the 1,497 s batch cap.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def _select_model(self, env: Any) -> str:
        """Pick the cheapest available model (Qwen3-4B preferred)."""
        preferred = "Qwen3-4B"
        if preferred in env.models:
            return preferred
        return sorted(env.models)[0]

    def _compute_max_continues(self, env: Any) -> int:
        """Derive max continues from the weighted-compute budget.

        Uses a conservative per-step estimate (4000 WC) so that even
        with all-long-problem batches, the mean WC stays safely under
        the hard threshold.
        """
        req = env.requirements
        if req is None:
            return 5
        wc_budget = float(req["weighted_compute"]["value"])
        # Conservative per-step weighted-compute for Qwen3-4B.
        # Observed: ~3,400-3,530 WC per checkpoint step on average.
        # Use 4,000 as safety margin for tail variance.
        per_step_estimate = 4000.0
        max_steps = max(1, int(wc_budget / per_step_estimate))
        # Subtract 1 for the spawn step (which also generates tokens).
        max_cont = max(0, max_steps - 1)
        # Cap to avoid runaway with very generous budgets.
        return min(max_cont, 12)

    @staticmethod
    def _deterministic_plurality(answers: List[str]) -> Optional[str]:
        """Most common answer; first-appearance tie-break."""
        if not answers:
            return None
        counts = Counter(answers)
        top = max(counts.values())
        return next(a for a in answers if counts[a] == top)

    def solve(self, env: Any) -> Optional[str]:
        model = self._select_model(env)
        max_cont = self._compute_max_continues(env)

        # Spawn a single branch -- reveals checkpoint 0.
        branch = env.spawn(model)
        answers: List[str] = []

        ans = valid_answer(branch.get("latest_answer"))
        if ans is not None:
            answers.append(ans)

        # Continue to terminal or budget-derived cap.
        # Do NOT stop at first valid answer: the terminal checkpoint
        # answer is far more reliable than early intermediates, and
        # self-consistency across multiple answers adds robustness.
        cont = 0
        while branch["can_continue"] and cont < max_cont:
            branch = env.continue_branch(branch["branch_id"])
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                answers.append(ans)
            cont += 1

        # Self-consistency: deterministic majority vote across all
        # checkpoint answers.  Falls back to None when no checkpoint
        # produced a valid answer.
        final = self._deterministic_plurality(answers)

        return env.finish(final)
