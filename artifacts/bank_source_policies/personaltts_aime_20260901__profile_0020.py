from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Wide parallel Qwen3-1.7B branches with 2-phase execution for majority voting.

    Builds on round 3 insight: Qwen3-1.7B with spawn_many + continue_many
    achieves 5.48% accuracy at width=10, just below the 5.67% threshold.
    Latency barely changes with width (p99=11.49s for both width=2 and width=10)
    and compute usage is <0.3% of budget.  Widening to 20-30 branches pushes
    accuracy above threshold via logarithmic scaling (estimated +0.4-0.7pp).

    The continue phase is critical: at checkpoint 1, branches converge
    strongly toward the correct answer (9-10/10 agreement in observed traces).
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @staticmethod
    def _model_size(name: str) -> float:
        """Extract parameter count in billions from model name like 'Qwen3-1.7B'."""
        best = float("inf")
        i = 0
        while i < len(name):
            if name[i] in ("B", "b") and i > 0:
                j = i - 1
                while j >= 0 and (name[j].isdigit() or name[j] == "."):
                    j -= 1
                num_str = name[j + 1 : i]
                if num_str and not num_str.startswith(".") and not num_str.endswith("."):
                    try:
                        size = float(num_str)
                        if 0 < size < best:
                            best = size
                    except ValueError:
                        pass
            i += 1
        return best

    def _select_model_and_width(self, env: Any) -> tuple:
        """Select model and branch width that maximize accuracy within constraints.

        Strategy: pick the largest model whose 2-phase parallel execution
        (spawn_many + continue_many) fits within the batch latency budget,
        then maximize width within compute budget.
        """
        latency_limit = env.requirements["latency_s"]["value"]
        compute_limit = env.requirements["weighted_compute"]["value"]

        models = sorted(env.models, key=self._model_size)
        if not models:
            return (list(env.models)[0], 10)

        # Empirical calibration from rounds 2-3:
        # Qwen3-1.7B, 2 phases (spawn_many + continue_many), width 2-10
        #   avg_latency: 10.19-10.34s, p90: 10.66-10.83s, p99: 11.49s
        # Width effect on latency is negligible for parallel ops.
        # Scale by sqrt(model_size) for different models.
        BASE_TWO_PHASE_BATCH_1_7B = 11.3  # Conservative batch estimate

        best_model = None
        best_size = 0.0

        # Pick the largest model where 2-phase batch fits the latency budget
        for model in reversed(models):
            size = self._model_size(model)
            if size == float("inf") or size <= 0:
                continue
            estimated_batch = BASE_TWO_PHASE_BATCH_1_7B * (size / 1.7) ** 0.5
            if estimated_batch <= latency_limit and size > best_size:
                best_model = model
                best_size = size

        if best_model is None:
            # No model fits 2 phases; use smallest model
            best_model = models[0]
            best_size = self._model_size(best_model)
            if best_size == float("inf"):
                best_size = 1.0

        # Compute the weight multiplier for weighted compute
        weight = max(best_size / 0.6, 1.0)

        # Determine maximum width from compute budget
        # Per-branch: ~1100 raw tokens (observed ~1033 for 1.7B, add margin)
        cost_per_branch = 1100.0 * weight
        max_width_compute = max(1, int(compute_limit * 0.85 / cost_per_branch))

        # Cap width: more branches = more accuracy via voting, but diminishing
        # returns.  30 is aggressive enough to push past the 5.67% threshold
        # while remaining safe for pool limits.
        width = min(max_width_compute, 30)
        width = max(width, 2)

        return (best_model, width)

    def solve(self, env: Any) -> Optional[str]:
        model, target_width = self._select_model_and_width(env)

        # Phase 1: Spawn parallel cohort with progressive fallback
        branches = None
        fallback_widths = sorted(
            {target_width, 20, 15, 10, 5, 2, 1}, reverse=True
        )
        for try_width in fallback_widths:
            if try_width > target_width:
                continue
            try:
                if try_width == 1:
                    branches = [env.spawn(model)]
                else:
                    branches = env.spawn_many(model, try_width)
                break
            except Exception:
                if try_width <= 1:
                    return env.finish(None)
                continue

        if branches is None:
            return env.finish(None)

        # Collect checkpoint-0 answers as fallback
        ck0: Dict[int, Optional[str]] = {}
        for b in branches:
            bid = b["branch_id"]
            ck0[bid] = valid_answer(b.get("latest_answer"))

        # Phase 2: Continue all branches to checkpoint 1
        ck1: Dict[int, Optional[str]] = {}
        continuable = [b for b in branches if b["can_continue"]]
        if continuable:
            try:
                updated = env.continue_many(
                    [b["branch_id"] for b in continuable]
                )
                for b in updated:
                    bid = b["branch_id"]
                    ck1[bid] = valid_answer(b.get("latest_answer"))
            except Exception:
                pass

        # Build votes: prefer checkpoint-1 answer, fallback to checkpoint-0
        votes: List[str] = []
        for b in branches:
            bid = b["branch_id"]
            ans = ck1.get(bid)
            if ans is None:
                ans = ck0.get(bid)
            if ans is not None:
                votes.append(ans)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
