from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Wide parallel sampling to maximise batch accuracy under a very low threshold.

    Profile 0081 requires only ~1.48 % mean accuracy but gives enormous
    weighted-compute headroom (~1.76 M vs ~7 K per single branch) and a
    comfortable latency budget (26.5 s batch vs ~12 s for two phases).

    At checkpoint 1 each Qwen3-4B branch independently produces a correct
    non-null answer with ~1.5 % probability; incorrect branches almost always
    return null.  Spawning N branches in parallel and plurality-voting gives
    P(correct) = 1 - (1 - 0.015)^N, which reaches ~97 % at N ~ 240.
    Because spawn_many and continue_many execute as single cohort phases,
    latency stays at two phases (~12-13 s) regardless of width.
    """

    NAME = "candidate"

    def pick_model(self, models: list) -> str:
        """Select the model with known low latency and good branch diversity."""
        for preferred in [
            "Qwen3-4B",
            "Qwen3-1.5B",
            "Qwen3-0.5B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
        ]:
            if preferred in models:
                return preferred
        return models[0]

    def model_weight(self, model: str) -> float:
        """Estimated weighted-compute weight per raw token, by model name."""
        weights = {
            "Qwen3-0.5B": 0.833,
            "Qwen3-1.5B": 2.5,
            "Qwen3-4B": 6.667,
            "Qwen3-8B": 13.333,
            "Qwen3-14B": 23.333,
            "Qwen3-32B": 53.333,
        }
        return weights.get(model, 6.667)

    def plurality(self, answers: list) -> Optional[str]:
        """Deterministic plurality: first-appearance tie-break."""
        if not answers:
            return None
        counts = Counter(answers)
        top = max(counts.values())
        return next(a for a in answers if counts[a] == top)

    def max_branches(self, env: Any, model: str) -> int:
        """Determine the largest safe branch count."""
        req = env.requirements
        compute_limit = req["weighted_compute"]["value"]
        w = self.model_weight(model)
        raw_per_branch = 1070  # conservative raw token estimate
        wc_per_branch = raw_per_branch * w

        # Budget-based cap (5 % safety margin)
        n_budget = max(1, int(compute_limit * 0.95 / wc_per_branch))

        # Pool-based cap
        state = env.observe()
        pool = int(state["remaining_by_model"].get(model, 0))
        n = min(n_budget, pool)
        n = max(n, 1)

        # Refine with the authoritative affordability checker
        try:
            while n > 1 and not env.can_afford_complete_candidates(
                model,
                count=n,
                checkpoints_per_stage=2,
                max_refine_rounds=0,
            ):
                n = max(1, n * 3 // 4)
        except Exception:
            pass  # checker unavailable; use budget/pool estimate

        return n

    def try_spawn(self, env: Any, model: str, n: int) -> list:
        """Spawn n branches, falling back to fewer on InvalidAction."""
        while n > 1:
            try:
                return list(env.spawn_many(model, n))
            except Exception:
                n = max(1, n * 3 // 4)
        # Final single-branch fallback
        try:
            return [env.spawn(model)]
        except Exception:
            return []

    def solve(self, env: Any) -> Optional[str]:
        model = self.pick_model(env.models)
        n = self.max_branches(env, model)

        # Phase 1 - parallel spawn (1 latency phase)
        opened = self.try_spawn(env, model, n)
        if not opened:
            return env.finish(None)

        # Phase 2 - advance every branch by one checkpoint (1 latency phase)
        continuable = [
            b["branch_id"] for b in opened if b["can_continue"]
        ]
        if continuable:
            try:
                if len(continuable) == 1:
                    env.continue_branch(continuable[0])
                else:
                    env.continue_many(continuable)
            except Exception:
                pass  # fall through to collect whatever answers exist

        # Collect every non-null answer from the current branch state
        state = env.observe()
        answers = []
        for b in state["branches"]:
            if not b["pruned"]:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    answers.append(ans)

        # Finish with deterministic plurality vote
        return env.finish(self.plurality(answers))
