from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel majority-voting policy tuned for profile_0067.

    Profile 0067 requires accuracy >= 0.247 (lenient), latency batch <= 5172 s,
    and weighted-compute mean <= 578 703.

    The baseline (sequential ASC with Qwen3-4B, ~11.6 branches) achieves 78 %
    accuracy but massively exceeds latency and compute.  Round 0 used Qwen3-0.6B
    (cheapest model) but only reached 5 % accuracy.

    This policy fixes both shortcomings:
      * Selects the largest model that affords >= 3 parallel branches within the
        weighted-compute budget (Qwen3-4B at 3 branches costs ~400 K weighted
        compute on average, well under the 578 K cap).
      * Uses spawn_many / continue_many so per-episode latency equals the
        maximum branch duration, not the sum -- keeping batch latency far below
        5 172 s even on hard problems.
      * Stops as soon as a strict majority of branches agree (2 of 3), saving
        additional tokens on easy problems.
      * Skips refinement to avoid feedback-token overhead.
    """

    NAME = "candidate"

    # ------------------------------------------------------------------ helpers

    def _model_size(self, name: str) -> float:
        """Parse parameter-count tag (e.g. '4B' -> 4.0) from a model name."""
        for part in name.replace("_", "-").split("-"):
            upper = part.upper()
            if upper.endswith("B"):
                try:
                    return float(upper[:-1])
                except ValueError:
                    continue
        return float("inf")

    def _choose_model_and_width(
        self, models: List[str], compute_budget: float
    ) -> tuple:
        """Pick the largest model that fits >= 3 branches in the budget.

        Weighted-compute multiplier = model_size_B / 0.6 (verified from trace
        data: Qwen3-0.6B has multiplier 1, Qwen3-4B has multiplier 6.667).
        Estimated average raw tokens per branch is about 20 000 (baseline mean
        230 143 raw tokens / 11.6 avg branches).
        """
        est_raw_per_branch = 20000
        base_b = 0.6  # smallest known Qwen3 model size

        known: List[tuple] = []
        for m in models:
            s = self._model_size(m)
            if s < float("inf"):
                known.append((m, s))
        if not known:
            return models[0], 3

        # Sort descending by size -- try the most capable model first
        known.sort(key=lambda x: -x[1])

        for model, size_b in known:
            multiplier = size_b / base_b
            cost_3 = 3 * est_raw_per_branch * multiplier
            if cost_3 <= compute_budget * 0.85:
                # Try wider cohort if affordable
                width = 3
                for w in (5, 4):
                    if w * est_raw_per_branch * multiplier <= compute_budget * 0.75:
                        width = w
                        break
                return model, width

        # Fallback: pick best model that affords at least 1 branch
        for model, size_b in known:
            multiplier = size_b / base_b
            affordable = max(
                1,
                int(compute_budget * 0.80 / (est_raw_per_branch * multiplier)),
            )
            return model, min(affordable, 7)

        # Absolute fallback
        return known[-1][0], 1

    # ------------------------------------------------------------------ voting

    def _gather_answers(self, bmap: Dict[Any, Dict[str, Any]]) -> List[str]:
        """Collect valid answers from all branches."""
        out: List[str] = []
        for b in bmap.values():
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                out.append(a)
        return out

    def _majority_answer(
        self, bmap: Dict[Any, Dict[str, Any]], need: int
    ) -> Optional[str]:
        """Return the answer if any value has >= *need* votes, else None."""
        answers = self._gather_answers(bmap)
        if len(answers) < need:
            return None
        counts = Counter(answers)
        best, best_n = counts.most_common(1)[0]
        return best if best_n >= need else None

    # ------------------------------------------------------------------ solve

    def solve(self, env) -> Optional[str]:
        reqs = env.requirements
        compute_budget = float(reqs["weighted_compute"]["value"])

        models = list(env.models)
        model, width = self._choose_model_and_width(models, compute_budget)
        majority_needed = width // 2 + 1

        # --- spawn parallel cohort ---
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            return self._run_single(env, model)

        bmap: Dict[Any, Dict[str, Any]] = {b["branch_id"]: b for b in opened}

        # --- advance and vote ---
        for _ in range(250):  # generous safety cap
            ans = self._majority_answer(bmap, majority_needed)
            if ans is not None:
                return env.finish(ans)

            active = [bid for bid, b in bmap.items() if b["can_continue"]]
            if not active:
                break

            try:
                updated = env.continue_many(active)
            except Exception:
                break

            for b in updated:
                bmap[b["branch_id"]] = b

        # --- final fallback ---
        ans = self._majority_answer(bmap, majority_needed)
        if ans is not None:
            return env.finish(ans)

        answers = self._gather_answers(bmap)
        return env.finish(deterministic_plurality(answers) if answers else None)

    # ------------------------------------------------------------------ single-branch fallback

    def _run_single(self, env, model: str) -> Optional[str]:
        """Fallback: run one branch to completion when spawn_many fails."""
        branch = env.spawn(model)
        while branch["can_continue"]:
            try:
                branch = env.continue_branch(branch["branch_id"])
            except Exception:
                break
        return env.finish(valid_answer(branch.get("latest_answer")))
