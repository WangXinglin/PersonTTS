from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel multi-branch voting with budget-aware model and depth selection.

    Addresses the tradeoff between accuracy and latency/compute by:
    - Selecting a capable model (Qwen3-4B preferred, proven from baseline) that
      fits within the weighted-compute budget.
    - Spawning multiple branches in parallel via spawn_many to keep sequential
      latency low (each phase adds only the max cohort latency, not the sum).
    - Limiting checkpoint depth to respect the batch latency constraint.
    - Voting by deterministic plurality across all branches.

    For profile_0070 (accuracy >= 8.2%, latency batch <= 183s, compute <= 842K):
    typical config is Qwen3-4B, width=8, depth=5, yielding ~160K compute,
    ~120s latency, and accuracy well above 8.2%.
    """

    NAME = "candidate"

    MODEL_SIZES: Dict[str, float] = {
        "Qwen3-0.6B": 0.6,
        "Qwen3-1.7B": 1.7,
        "Qwen3-4B": 4.0,
        "Qwen3-8B": 8.0,
        "Qwen3-14B": 14.0,
        "Qwen3-30B": 30.0,
        "Qwen3-32B": 32.0,
        "Qwen3-235B": 235.0,
    }

    def model_weight(self, model: str) -> float:
        """Weighted-compute multiplier per raw token (model_size_B / 0.6)."""
        return self.MODEL_SIZES.get(model, 4.0) / 0.6

    def check_affordable(self, env, model: str, width: int, depth: int) -> bool:
        """Check if env can afford the planned work, True if checker unavailable."""
        try:
            return bool(
                env.can_afford_complete_candidates(
                    model,
                    count=width,
                    checkpoints_per_stage=depth + 1,
                    max_refine_rounds=0,
                )
            )
        except (AttributeError, TypeError):
            return True

    def select_config(self, env) -> tuple:
        """Choose model, parallel width, and depth limit from requirements."""
        req = env.requirements
        if req is None:
            return env.models[0], 1, 3

        compute_limit = float(req["weighted_compute"]["value"])
        latency_limit = float(req["latency_s"]["value"])

        # Conservative per-checkpoint estimates derived from baseline traces:
        # ~500-600 raw tokens per checkpoint, ~15-20s per parallel phase.
        est_tokens_per_cp = 600
        est_latency_per_phase = 20.0

        # Safety margins: 75% of latency budget (batch), 80% of compute (mean)
        safe_latency = latency_limit * 0.75
        safe_compute = compute_limit * 0.80

        # Model preference: Qwen3-4B is proven from baseline evidence,
        # then try other capable models in decreasing reliability order.
        model_order = [
            "Qwen3-4B", "Qwen3-8B", "Qwen3-1.7B",
            "Qwen3-14B", "Qwen3-0.6B",
        ]
        available = set(env.models)

        for model in model_order:
            if model not in available:
                continue

            wf = self.model_weight(model)

            # Depth: max continue rounds that fit within batch latency budget
            max_phases = max(2, int(safe_latency / est_latency_per_phase))
            depth = max_phases - 1  # one phase consumed by spawn

            # Width: max parallel branches that fit mean compute budget
            branch_cost = est_tokens_per_cp * (depth + 1) * wf
            if branch_cost <= 0:
                continue
            width = max(1, min(int(safe_compute / branch_cost), 8))

            # Validate against env pool/budget checker
            while width > 0:
                if self.check_affordable(env, model, width, depth):
                    break
                width -= 1
            if width < 1:
                continue

            if depth >= 1:
                return model, width, depth

        # Absolute fallback: first available model, single branch
        return env.models[0], 1, 3

    def solve(self, env) -> Optional[str]:
        model, width, max_depth = self.select_config(env)

        # Spawn branches in parallel for minimal latency
        initial: List[Dict[str, Any]] = []
        if width <= 1:
            initial = [env.spawn(model)]
        else:
            try:
                initial = list(env.spawn_many(model, width))
            except Exception:
                # Pool too small for requested width; fall back to one branch
                initial = [env.spawn(model)]

        # Track branch states by id
        branch_map: Dict[int, Dict[str, Any]] = {}
        for b in initial:
            branch_map[b["branch_id"]] = b

        # Advance all continuable branches in synchronous parallel rounds
        for round_idx in range(max_depth):
            continuable = [
                bid
                for bid, b in branch_map.items()
                if b["can_continue"] and not b["pruned"]
            ]
            if not continuable:
                break

            if len(continuable) == 1:
                upd = env.continue_branch(continuable[0])
                branch_map[upd["branch_id"]] = upd
            else:
                try:
                    for upd in env.continue_many(continuable):
                        branch_map[upd["branch_id"]] = upd
                except Exception:
                    break  # budget exhausted mid-round

        # Collect votes from all branches with valid answers
        votes: List[str] = []
        for b in branch_map.values():
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                votes.append(ans)

        # Determine final answer by deterministic plurality vote
        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
