from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Conditional-depth policy: 3 base checkpoints + adaptive 4th for hard problems.

    The incumbent (round 3) runs Qwen3-4B with 3 checkpoints plus a Qwen3-0.6B
    fallback, achieving 7.08% per-problem accuracy and 98.9% batch satisfaction.
    The binding constraint is accuracy: 11 out of 1000 batches fail to reach the
    2.77% mean accuracy threshold (fewer than 2 correct out of 60 problems).

    Round 1 demonstrated that 4 checkpoints of Qwen3-4B yields 10.54% accuracy
    (100% batch accuracy satisfaction) but violated the weighted-compute budget
    (mean 14018 vs limit 12964) because every problem paid the 4-checkpoint cost.

    This policy exploits the mean-aggregation constraint: run 3 checkpoints first,
    then check for a valid answer.  Problems that find an answer by checkpoint 2
    (~42% of episodes) stop at cost ~10515 weighted compute.  Problems without an
    answer continue to a 4th checkpoint at cost ~14020.  The batch mean is
    approximately 0.42*10515 + 0.58*14020 = 12545, well under the 12964 budget,
    while achieving near-round-1 accuracy (~10.5%).  Per-problem hardness is
    largely deterministic (fixed per problem across seeds), keeping the per-batch
    variance low enough that compute violations are negligible.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_size_b(name: str) -> float:
        """Return the largest 'NB'-suffixed number in *name*, or 0.6 default."""
        best = 0.0
        i = 0
        while i < len(name):
            if name[i].isdigit() or name[i] == ".":
                j = i
                while j < len(name) and (name[j].isdigit() or name[j] == "."):
                    j += 1
                if j < len(name) and name[j] in ("B", "b"):
                    try:
                        val = float(name[i:j])
                        if val > best:
                            best = val
                    except ValueError:
                        pass
                    i = j + 1
                else:
                    i = j + 1
            else:
                i += 1
        return best if best > 0.0 else 0.6

    @staticmethod
    def _best_answer(branch: Dict[str, Any]) -> Optional[str]:
        """Extract best available answer, checking answer_history as fallback."""
        answer = valid_answer(branch.get("latest_answer"))
        if answer is not None:
            return answer
        history = branch.get("answer_history")
        if history:
            for past in reversed(history):
                a = valid_answer(past)
                if a is not None:
                    return a
        return None

    # ── main solve ────────────────────────────────────────────────

    def solve(self, env) -> Optional[str]:
        reqs = env.requirements
        if reqs is None:
            branch = env.spawn(env.models[0])
            return env.finish(valid_answer(branch.get("latest_answer")))

        compute_budget = float(reqs["weighted_compute"]["value"])
        raw_per_ckpt = 540.0  # Conservative per-checkpoint raw token estimate

        # ── Select primary model: largest that can afford >= 3 checkpoints ──
        models_info: List[tuple] = []
        for model in env.models:
            size_b = self._extract_size_b(model)
            weight = max(1.0, size_b / 0.6)
            wc_per_ckpt = raw_per_ckpt * weight
            affordable = int(compute_budget / wc_per_ckpt)
            models_info.append((model, size_b, weight, wc_per_ckpt, affordable))

        # Sort by size descending to prefer larger models
        models_info.sort(key=lambda x: x[1], reverse=True)

        primary = None
        for m in models_info:
            if m[4] >= 3:  # Can afford at least 3 checkpoints
                primary = m
                break

        if primary is None:
            # Fallback to cheapest model with maximum affordable depth
            primary = min(models_info, key=lambda x: x[3])

        primary_name = primary[0]
        base_depth = min(3, max(1, primary[4]))

        # ── Phase 1: Run primary model for base_depth checkpoints ──
        branch = env.spawn(primary_name)
        for _ in range(base_depth - 1):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(branch["branch_id"])

        # Check for a valid answer after the base phase
        answer = self._best_answer(branch)
        if answer is not None:
            return env.finish(answer)

        # ── Phase 2: Extend by one additional checkpoint ──
        # Only ~58% of problems reach here (those without an answer at base depth).
        # The extra checkpoint nearly doubles per-problem accuracy (5.7% -> 10.5%)
        # while keeping the batch-mean compute under budget because ~42% of
        # problems stopped cheaply in Phase 1.
        if branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])
            answer = self._best_answer(branch)
            if answer is not None:
                return env.finish(answer)

        # ── No valid answer found ──
        return env.finish(None)
