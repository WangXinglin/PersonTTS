from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel Qwen3-4B branches with majority voting.

    Profile 0075 constraints:
      - accuracy  mean >= 0.024  (very low threshold)
      - latency   batch  <= 29.34s
      - weighted_compute mean <= 12.09M

    The round-0 incumbent used Qwen3-0.6B (0.64% episode accuracy,
    utility 0.047).  Switching to Qwen3-4B with parallel branches
    dramatically boosts accuracy while staying within the generous
    latency and compute budgets.

    Approach:
      1. spawn_many(Qwen3-4B, 3) -- parallel, latency = max of cohort ~7s
      2. continue_many twice      -- 2 parallel phases, ~7s each
      3. Plurality vote across branch answers

    Expected per-problem budget:
      - Latency: ~21s average (3 phases x ~7s), batch well under 29.34s
      - Weighted compute: 3 branches x 3 checkpoints x ~500 tok x 6.67
        = ~30k, negligible vs 12M budget
      - Accuracy: Qwen3-4B single-branch accuracy ~20-40%, with 3-branch
        majority voting even higher, far above the 2.4% threshold
    """

    NAME = "candidate"

    @staticmethod
    def _model_size_key(name: str) -> float:
        """Extract numeric billion-parameter tag for sorting."""
        parts = name.replace("-", "_").split("_")
        for part in parts:
            upper = part.upper()
            if upper.endswith("B"):
                try:
                    return float(upper[:-1])
                except ValueError:
                    pass
        return float("inf")

    def _select_model(self, env: Any) -> str:
        """Pick Qwen3-4B if available, else the largest available model."""
        models = list(env.models)
        for m in models:
            if m == "Qwen3-4B":
                return m
        # Fall back to largest available model
        return max(models, key=self._model_size_key)

    def solve(self, env: Any) -> Optional[str]:
        model = self._select_model(env)

        # Phase 1: spawn 3 parallel branches
        width = 3
        try:
            branches = env.spawn_many(model, width)
        except Exception:
            # If spawn_many fails, fall back to single spawn
            branch = env.spawn(model)
            answer = valid_answer(branch.get("latest_answer"))
            if answer is not None:
                return env.finish(answer)
            if branch["can_continue"]:
                branch = env.continue_branch(branch["branch_id"])
                answer = valid_answer(branch.get("latest_answer"))
            return env.finish(answer)

        # Phase 2-3: continue twice in parallel
        for _ in range(2):
            continuable = [
                b["branch_id"] for b in branches
                if b["can_continue"]
            ]
            if not continuable:
                break
            try:
                updated = env.continue_many(continuable)
            except Exception:
                break
            # Merge updated info back
            updated_map = {b["branch_id"]: b for b in updated}
            branches = [
                updated_map.get(b["branch_id"], b) for b in branches
            ]

        # Collect answers from all branches
        votes: List[str] = []
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                votes.append(ans)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
