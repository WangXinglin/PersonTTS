from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Single-branch early-stop policy for tight compute/latency budgets.

    Profile 0054 has an extremely low accuracy requirement (mean >= 1/60)
    but very tight weighted-compute (mean <= ~23k) and latency (batch <= ~2740s)
    budgets.  The baseline (ASC with ~12 branches each run to completion)
    exceeds the compute budget by ~65x.

    This policy spawns exactly one Qwen3-4B branch per problem and continues
    only until the first valid answer appears, capped at a small number of
    checkpoints.  This keeps mean weighted-compute around 10k-18k (well under
    the 23,425 limit) while achieving non-trivial accuracy that easily clears
    the 1.67% bar.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _pick_model(self, env) -> str:
        """Select the smallest available model to minimize weighted compute."""
        # Prefer the known-cheapest model; fall back to first in the pool.
        for preferred in ("Qwen3-4B",):
            if preferred in env.models:
                return preferred
        return env.models[0]

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # Spawn a single branch -- checkpoint 0 (typically no answer yet).
        branch = env.spawn(model)

        # Continue until we obtain a valid answer or exhaust our checkpoint
        # budget.  Empirically, first valid answers appear at checkpoint 1-5.
        # Capping at 5 continues keeps mean weighted-compute safely under
        # the per-episode budget (~23k) even in the worst case.
        answer = valid_answer(branch.get("latest_answer"))
        max_continues = 5
        continues_done = 0
        while answer is None and branch["can_continue"] and continues_done < max_continues:
            branch = env.continue_branch(branch["branch_id"])
            answer = valid_answer(branch.get("latest_answer"))
            continues_done += 1

        return env.finish(answer)
