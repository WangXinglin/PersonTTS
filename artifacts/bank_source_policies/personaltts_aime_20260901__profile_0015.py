from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Wide parallel spawn with first-valid-answer selection.

    Profile 0015 constraints:
      accuracy  >= 3.27%  (batch mean, lenient – ~2 correct out of 60)
      latency   <= 9.49s  (batch, tight)
      compute   <= 6.99M  (batch mean, very generous)

    Key insight from prior rounds:
    - Plurality voting over many weak-model branches DESTROYS accuracy
      (round 2: 50 branches + plurality → 0.003%) because wrong answers
      cluster and outvote the rare correct answer.
    - Single-branch selection preserves per-branch accuracy (round 0:
      single branch → 0.935%).
    - spawn_many is one parallel phase (~5 s), well under latency budget.

    Strategy:
    1. Pick the largest model whose size is ≤ 2 B (sweet spot for producing
       valid checkpoint-0 answers with reasonable accuracy). Fall back to the
       smallest available model.
    2. spawn_many a wide parallel cohort (up to 50 branches) in ONE phase.
    3. Select the FIRST valid answer from the cohort (NOT plurality).
       This gives per-episode accuracy ≈ P(correct | valid at cp0) because
       all branches are independent draws.  With 50 branches the probability
       of at least one valid answer is effectively 1, so the null-answer
       penalty vanishes.
    4. If no branch produces a valid answer at cp0: try one continue_many
       round (adds ~5 s) and pick the first valid answer from the updated
       cohort. This fallback rarely triggers for small models.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_size(name: str) -> float:
        """Parse billion-parameter count from model name."""
        for part in name.replace("_", "-").split("-"):
            p = part.strip()
            if len(p) > 1 and p[-1].upper() == "B":
                try:
                    return float(p[:-1])
                except ValueError:
                    continue
        return 999.0

    def _pick_model(self, models: List[str]) -> str:
        """Select the best model for checkpoint-0 first-valid strategy.

        Preference order:
          1. Largest model ≤ 2 B  (likely finishes thinking in 500 tokens
             AND has higher reasoning quality than 0.6 B).
          2. Smallest model available (0.6 B fallback – known to produce
             valid cp0 answers).
        """
        sized = sorted(models, key=self._model_size)
        best = None
        for m in sized:
            sz = self._model_size(m)
            if sz <= 2.0:
                best = m  # keep updating → picks the largest ≤ 2 B
        if best is not None:
            return best
        # No model ≤ 2 B → use the smallest available
        return sized[0]

    @staticmethod
    def _first_valid(branches: List[Dict[str, Any]]) -> Optional[str]:
        """Return the first valid answer from a list of branch dicts."""
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                return ans
        return None

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(list(env.models))

        # --- Phase 1: wide parallel spawn (one cohort phase, ~5 s) -------
        state = env.observe()
        remaining = int(state["remaining_by_model"].get(model, 0))
        width = min(50, max(1, remaining))

        if width >= 2:
            try:
                branches = list(env.spawn_many(model, width))
            except Exception:
                # Fallback to single spawn
                try:
                    branches = [env.spawn(model)]
                except Exception:
                    return env.finish(None)
        else:
            try:
                branches = [env.spawn(model)]
            except Exception:
                return env.finish(None)

        # Pick the first valid checkpoint-0 answer
        answer = self._first_valid(branches)
        if answer is not None:
            return env.finish(answer)

        # --- Phase 2 (fallback): one continue round (~5 s extra) ---------
        # Only triggered when ALL cp0 answers are null (rare for small
        # models, common for larger models like 4 B).  Total latency
        # becomes ~10 s.  Since this rarely triggers for small models,
        # batch latency stays well under 9.49 s.
        continuable = [b for b in branches if b["can_continue"]]
        if continuable:
            try:
                ids = [b["branch_id"] for b in continuable]
                if len(ids) >= 2:
                    updated = list(env.continue_many(ids))
                else:
                    updated = [env.continue_branch(ids[0])]
                answer = self._first_valid(updated)
                if answer is not None:
                    return env.finish(answer)
            except Exception:
                pass

        return env.finish(None)
