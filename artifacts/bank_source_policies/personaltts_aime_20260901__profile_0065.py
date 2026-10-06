from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy


class CandidatePolicy(ControllerPolicy):
    """Wide-and-deep parallel majority vote exploiting compute/latency headroom.

    The incumbent uses Qwen3-8B width=5 depth=5 and gets 23.5% accuracy,
    well below the 26.13% requirement.  Latency (33s) and compute (173k)
    are far under the 558s / 1.8M budgets.

    This policy uses ~20 parallel branches continued for ~10 rounds each,
    spending ~90% of the compute budget while staying well within latency.
    The wider vote plus deeper per-branch reasoning should push accuracy
    above the threshold.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _model_size_b(name: str) -> float:
        """Extract numeric size in billions from model name like 'Qwen3-8B'."""
        for part in name.replace("-", " ").replace("_", " ").split():
            if part.endswith("B") and len(part) > 1:
                try:
                    return float(part[:-1])
                except ValueError:
                    continue
        return 0.0

    def _pick_model(self, env) -> str:
        """Select the largest available model up to 8B for best accuracy."""
        models = list(env.models)
        if len(models) == 1:
            return models[0]
        sized = [(m, self._model_size_b(m)) for m in models]
        sized.sort(key=lambda x: x[1], reverse=True)
        # Prefer largest model in (0, 8] B
        for m, s in sized:
            if 0 < s <= 8.0:
                return m
        # Fallback: smallest parseable
        sized.sort(key=lambda x: x[1])
        for m, s in sized:
            if s > 0:
                return m
        return models[0]

    def _plan(self, env, model: str) -> tuple:
        """Determine (width, max_continue_rounds) from requirement budgets."""
        req = env.requirements
        lat_budget = 558.0
        compute_budget = 1800000.0
        if req is not None:
            lat_c = req.get("latency_s")
            if lat_c is not None:
                lat_budget = float(lat_c["value"])
            wc_c = req.get("weighted_compute")
            if wc_c is not None:
                compute_budget = float(wc_c["value"])

        model_size = self._model_size_b(model)
        if model_size <= 0:
            model_size = 8.0

        # Cost model calibrated from Qwen3-8B observed traces:
        #   ~520 raw tokens per checkpoint
        #   weight multiplier ~ model_size / 0.6
        #   ~7s per parallel continue round at 8B
        weight_factor = max(1.0, model_size / 0.6)
        est_compute_per_ckpt = 520.0 * weight_factor
        est_latency_per_round = max(4.0, model_size * 0.9)

        # Safety margins: 90% compute (mean), 65% latency (batch headroom)
        safe_compute = compute_budget * 0.90
        safe_latency = lat_budget * 0.65

        max_continues_by_lat = max(3, int(safe_latency / est_latency_per_round))
        max_total_ckpts = max(6, int(safe_compute / est_compute_per_ckpt))

        # Target at least 10 continues per branch for meaningful depth,
        # then maximize width within the remaining budget.
        min_continues = 10
        # width = floor(max_total_ckpts / (min_continues + 1))
        width = max(3, max_total_ckpts // (min_continues + 1))
        width = min(width, 30)

        # Recompute continues for the chosen width
        ckpts_per_branch = max_total_ckpts // max(1, width)
        continues = min(ckpts_per_branch - 1, max_continues_by_lat)
        continues = max(continues, 3)

        # Check pool availability
        state = env.observe()
        available = int(state["remaining_by_model"].get(model, 0))
        width = min(width, available)
        width = max(width, 1)

        return width, continues

    # ------------------------------------------------------------------ #
    #  main solve                                                          #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width, max_continues = self._plan(env, model)

        # Phase 1: spawn all branches in parallel (one latency round)
        opened = env.spawn_many(model, width)
        branches = {b["branch_id"]: b for b in opened}

        # Phase 2: continue all branches in parallel for max_continues rounds
        for _ in range(max_continues):
            continuable = [
                bid for bid, b in branches.items() if b["can_continue"]
            ]
            if not continuable:
                break
            updated = env.continue_many(continuable)
            for b in updated:
                branches[b["branch_id"]] = b

        # Phase 3: deterministic plurality vote across all branches
        return env.finish()
