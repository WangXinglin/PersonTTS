from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel majority-vote policy using the best available model.

    The incumbent used Qwen3-0.6B (single branch, ~0.6% accuracy) to stay
    within compute limits, but the accuracy threshold (3.27%) was missed by
    most batches.  The compute and latency budgets have massive headroom
    (~639 vs 149K compute, ~6s vs 40.5s latency).

    This policy:
    1. Picks the largest model where we can afford >= 3 parallel branches.
    2. Spawns branches in parallel (spawn_many) so latency = max, not sum.
    3. Continues one round (continue_many) to reach checkpoint 1 answers.
    4. Falls back with an extra continue for branches still missing answers.
    5. Returns deterministic plurality vote across all branch answers.

    For Qwen3-4B with 5 branches and 2 checkpoint rounds:
      - Weighted compute ~ 5 * 2 * 750 * 6.667 ~ 50K  (budget 149K)
      - Latency ~ 2 phases * max_step ~ 20-30s          (budget 40.5s batch)
      - Per-branch accuracy >> 3.27%, so batch satisfaction >> 4.7%
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_size_b(name: str) -> float:
        """Extract parameter count in billions from a model name."""
        for part in name.replace("-", " ").replace("_", " ").split():
            upper = part.upper()
            if upper.endswith("B"):
                try:
                    return float(upper[:-1])
                except ValueError:
                    continue
        return float("inf")

    @staticmethod
    def _weight_for_size(size_b: float) -> float:
        """Weighted-compute multiplier (observed: Qwen3-4B = 4/0.6 = 6.667)."""
        if size_b <= 0 or size_b == float("inf"):
            return 100.0
        return size_b / 0.6

    def _select_model_and_width(
        self,
        models: List[str],
        compute_budget: float,
    ) -> tuple:
        """Choose the best model and branch count within the compute budget.

        Prefers the largest model that still allows >= 3 parallel branches at
        50% budget utilisation (safety margin for per-problem variance).
        Falls back to the largest model allowing >= 1 branch at 70%.
        """
        parsed = sorted(
            [(m, self._parse_size_b(m)) for m in models],
            key=lambda x: x[1],
        )

        # Conservative per-branch cost: 2 rounds * 1000 raw tokens * weight
        est_cost_per_round = 1000  # raw tokens (upper-bound estimate)
        rounds = 2  # spawn (cp0) + 1 continue (cp1)

        best_model = parsed[0][0]
        best_width = 1

        # Pass 1: prefer >= 3 branches at 50% budget
        for model_name, size_b in reversed(parsed):
            w = self._weight_for_size(size_b)
            per_branch = rounds * est_cost_per_round * w
            n = int(compute_budget * 0.50 / per_branch)
            if n >= 3:
                best_model = model_name
                best_width = min(n, 7)
                return best_model, best_width

        # Pass 2: accept >= 1 branch at 70% budget
        for model_name, size_b in reversed(parsed):
            w = self._weight_for_size(size_b)
            per_branch = rounds * est_cost_per_round * w
            n = int(compute_budget * 0.70 / per_branch)
            if n >= 1:
                best_model = model_name
                best_width = min(n, 7)
                return best_model, best_width

        return best_model, best_width

    # ------------------------------------------------------------------ #
    # Main solver                                                          #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        req = env.requirements
        compute_budget = req["weighted_compute"]["value"]

        model, width = self._select_model_and_width(env.models, compute_budget)

        # Clamp to the available model pool
        state = env.observe()
        pool = state["remaining_by_model"].get(model, 0)
        width = max(1, min(width, pool))

        # ------ Phase 1: Spawn parallel branches --------------------------
        branches: List[Dict[str, Any]]
        if width >= 2:
            try:
                branches = list(env.spawn_many(model, width))
            except Exception:
                # Pool / budget issue: fall back to a single spawn
                branches = [env.spawn(model)]
        else:
            branches = [env.spawn(model)]

        # Keep a map of branch_id -> latest branch dict
        branch_map: Dict[Any, Dict[str, Any]] = {
            b["branch_id"]: b for b in branches
        }

        # ------ Phase 2: Continue to checkpoint 1 (parallel) ---------------
        to_continue = [
            bid for bid, b in branch_map.items() if b["can_continue"]
        ]
        if to_continue:
            try:
                if len(to_continue) >= 2:
                    updated = env.continue_many(to_continue)
                else:
                    updated = [env.continue_branch(to_continue[0])]
                for b in updated:
                    branch_map[b["branch_id"]] = b
            except Exception:
                pass  # budget / pool exhausted; use checkpoint-0 answers

        # Collect current valid answers
        answers = self._collect_answers(branch_map)

        # ------ Phase 3 (conditional): extra continue for null branches -----
        # Only when fewer than half the branches have answers.
        if len(answers) < max(1, len(branches) // 2):
            null_ids = [
                bid
                for bid, b in branch_map.items()
                if valid_answer(b.get("latest_answer")) is None
                and b.get("can_continue")
            ]
            if null_ids:
                try:
                    if len(null_ids) >= 2:
                        updated = env.continue_many(null_ids)
                    else:
                        updated = [env.continue_branch(null_ids[0])]
                    for b in updated:
                        branch_map[b["branch_id"]] = b
                except Exception:
                    pass
                answers = self._collect_answers(branch_map)

        # ------ Finish: plurality vote -------------------------------------
        final = deterministic_plurality(answers) if answers else None
        return env.finish(final)

    @staticmethod
    def _collect_answers(
        branch_map: Dict[Any, Dict[str, Any]],
    ) -> List[str]:
        """Return a list of valid answers (one per branch, latest checkpoint)."""
        result: List[str] = []
        for b in branch_map.values():
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                result.append(ans)
        return result
