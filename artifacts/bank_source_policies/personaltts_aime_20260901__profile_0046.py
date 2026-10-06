from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn first-non-null with continue_many recovery.

    Profile 0046 constraints:
      - accuracy  >= 0.17 %  (mean; need at least 1 correct per batch of 60)
      - latency_s <= 6.77 s  (batch; at most 3 episodes per batch can exceed)
      - weighted_compute <= 16.5 M  (mean; ample headroom)

    Incumbent (round 3, utility 0.113): spawns 3 Qwen3-0.6B branches via
    spawn_many, takes first-non-null from checkpoint 0.  When all 3 are null,
    continues ONLY branch 0 to checkpoint 1.  This gives 1904 continue
    episodes (3.17%), each adding ~5.3 s latency.  Of those, 30 produce
    correct cp1 answers, but some cp1 answers are also null, losing potential
    accuracy.

    Improvement: when all branches are null at cp0, use continue_many on
    ALL 3 branches instead of continue_branch on branch 0 alone.  Because
    continue_many is a parallel cohort, its latency equals the maximum of
    the individual continues (~5.3 s), identical to continuing a single
    branch.  But we now obtain 3 independent cp1 answers instead of 1.
    Taking first-non-null from 3 cp1 answers recovers episodes where the
    single-branch cp1 would have been null (~10-15 % of continues).  This
    yields ~4-5 extra correct answers at zero additional latency, pushing
    per-problem accuracy from 0.223 % to ~0.231 % and batch satisfaction
    from 11.3 % to ~12.0 %.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def solve(self, env) -> Optional[str]:
        model = self._pick_fastest_model(env)
        n_branches = self._choose_width(env, model)

        if n_branches <= 1:
            return self._solve_single(env, model)

        # ---- Multi-branch parallel spawn ---- #
        branches = env.spawn_many(model, n_branches)

        # Take first-non-null checkpoint-0 answer
        for b in branches:
            answer = valid_answer(b.get("latest_answer"))
            if answer is not None:
                return env.finish(answer)

        # All branches null at cp0 -> continue_many on all continuable
        # branches.  Parallel continue has the same latency as a single
        # continue but yields N independent cp1 answers.
        continuable_ids = [
            b["branch_id"] for b in branches if b["can_continue"]
        ]
        if continuable_ids:
            updated = env.continue_many(continuable_ids)
            for b in updated:
                answer = valid_answer(b.get("latest_answer"))
                if answer is not None:
                    return env.finish(answer)

        return env.finish(None)

    # ------------------------------------------------------------------ #
    # Single-branch fallback (when pool < desired width)                  #
    # ------------------------------------------------------------------ #

    def _solve_single(self, env, model: str) -> Optional[str]:
        branch = env.spawn(model)
        answer = valid_answer(branch.get("latest_answer"))
        if answer is not None:
            return env.finish(answer)
        # cp0 null -> one continue to cp1
        if branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])
            answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _choose_width(self, env, model: str) -> int:
        """Pick branch count respecting pool budget and compute cap."""
        target = 3  # Best accuracy/latency balance for this profile

        state = env.observe()
        pool = int(state["remaining_by_model"].get(model, 0))
        if pool < target:
            return max(1, pool)

        # Verify weighted-compute headroom (very conservative)
        req = env.requirements
        if req is not None:
            compute_limit = None
            for c in req.get("constraints", []):
                if c["metric"] == "weighted_compute":
                    compute_limit = c["value"]
                    break
            if compute_limit is not None:
                # Per-branch cost upper bound for smallest model (cp0 + cp1)
                cost_per_branch = 750
                max_n = int(compute_limit / cost_per_branch)
                if max_n < target:
                    return max(1, max_n)

        return target

    @staticmethod
    def _pick_fastest_model(env) -> str:
        """Select the smallest (fastest-inference) model from env.models."""
        models = list(env.models)
        if len(models) == 1:
            return models[0]

        def _size_key(name: str) -> float:
            parts = name.replace("-", " ").replace("_", " ").split()
            for part in parts:
                upper = part.upper()
                if upper.endswith("B") and len(upper) > 1:
                    try:
                        return float(upper[:-1])
                    except ValueError:
                        pass
            return float("inf")

        return min(models, key=_size_key)
