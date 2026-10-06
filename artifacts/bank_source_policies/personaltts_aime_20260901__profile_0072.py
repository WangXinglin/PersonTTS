from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Profile-aware policy that selects the largest affordable model with
    parallel majority voting to maximize batch accuracy satisfaction.

    Baseline evidence shows Qwen3-4B is available (weight 6.667) and far more
    accurate on AIME than Qwen3-0.6B.  Three parallel branches of 4B cost
    roughly 400k mean weighted_compute, well within the 603.8k budget, while
    delivering much higher accuracy than the 19.45 percent requirement.
    Width-3 voting is preferred over a single larger-model branch because
    it reduces per-batch variance, improving the satisfaction rate.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_size(name: str) -> float:
        """Extract numeric billion-parameter count from a model name."""
        for part in name.replace("-", "_").split("_"):
            if part.upper().endswith("B"):
                try:
                    return float(part[:-1])
                except ValueError:
                    pass
        return float("inf")

    def _can_afford(self, env: Any, model: str, width: int) -> bool:
        """Check whether the environment can afford the configuration."""
        try:
            return bool(
                env.can_afford_complete_candidates(
                    model,
                    count=width,
                    checkpoints_per_stage=40,
                    max_refine_rounds=0,
                )
            )
        except Exception:
            return True  # assume affordable if checker is unavailable

    def _fits_budget(self, size: float, base_size: float,
                     width: int, budget: float) -> bool:
        """Conservative manual cost estimate from baseline trace data."""
        weight = size / base_size if base_size > 0 else 1.0
        est_raw_per_branch = 20000  # conservative average per-branch raw tokens
        est_compute = width * est_raw_per_branch * weight
        return est_compute <= budget * 0.90

    def _select_config(self, env: Any) -> tuple:
        """Pick (model, width) prioritising width=3 voting across all models,
        then falling back to width=1 with the largest affordable model.
        """
        sized: List[tuple] = [
            (m, self._model_size(m)) for m in env.models
        ]
        sized = [(m, s) for m, s in sized if s != float("inf")]
        sized.sort(key=lambda x: x[1], reverse=True)  # largest first

        if not sized:
            return list(env.models)[0], 3

        base_size = sized[-1][1]

        # Read weighted-compute budget from requirements
        budget = 600000.0
        if self.requirements is not None:
            wc_info = self.requirements.get("weighted_compute")
            if isinstance(wc_info, dict):
                budget = float(wc_info.get("value", budget))

        # Pass 1: prefer width=3 (majority voting) — largest model first
        for model, size in sized:
            if (self._can_afford(env, model, 3)
                    and self._fits_budget(size, base_size, 3, budget)):
                return model, 3

        # Pass 2: width=1 (single branch, larger model) — largest first
        for model, size in sized:
            if (self._can_afford(env, model, 1)
                    and self._fits_budget(size, base_size, 1, budget)):
                return model, 1

        # Pass 3: relax manual check, trust only the environment checker
        for model, size in sized:
            for width in [3, 1]:
                if self._can_afford(env, model, width):
                    return model, width

        # Ultimate fallback: smallest model, width 3
        return sized[-1][0], 3

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, env: Any) -> Optional[str]:
        model, width = self._select_config(env)

        # --- Spawn cohort, with graceful width reduction on failure ---
        spawned = False
        for w in range(width, 0, -1):
            try:
                if w == 1:
                    env.spawn(model)
                else:
                    env.spawn_many(model, w)
                spawned = True
                break
            except Exception:
                continue

        if not spawned:
            return env.finish(None)

        # --- Advance all active branches to stage completion ---
        for _ in range(300):
            state = env.observe()
            active = [
                b["branch_id"]
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not active:
                break
            try:
                env.continue_many(active)
            except Exception:
                # Budget or pool exhausted mid-run; use partial answers
                break

        # --- Collect votes and finish ---
        state = env.observe()
        votes: List[str] = []
        for b in state["branches"]:
            if not b["pruned"]:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    votes.append(ans)

        final = deterministic_plurality(votes) if votes else None
        return env.finish(final)
