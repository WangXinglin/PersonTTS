from __future__ import annotations

from typing import Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Minimal single-branch early-exit policy for tight compute/latency budgets.

    Profile 0079 requires only >= 2.5 % mean accuracy but imposes a strict
    weighted-compute mean cap (~117 K) and a tight batch latency cap (~5 179 s).
    The baseline overserves accuracy (78 %) at ~1.5 M compute -- 13x over budget.

    This policy spawns exactly one Qwen3-4B branch (cheapest model) and advances
    it only until a valid answer appears, then finishes immediately.  A safety
    cap of 15 checkpoint continues protects against extremely long branches on
    hard problems, keeping worst-case per-episode compute well under budget.
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        # Pick the cheapest available model (Qwen3-4B has the lowest weight
        # multiplier, confirmed at ~6.67x).
        model = "Qwen3-4B"
        if model not in env.models:
            model = env.models[0]

        max_continues = 15

        # --- single branch, early exit on first valid answer -----------------
        branch = env.spawn(model)
        branch_id = branch["branch_id"]
        answer = valid_answer(branch.get("latest_answer"))

        continues = 0
        while answer is None and branch["can_continue"] and continues < max_continues:
            branch = env.continue_branch(branch_id)
            answer = valid_answer(branch.get("latest_answer"))
            continues += 1

        # Finish with whatever answer we have (may be None for very rare cases
        # where no valid answer is produced within the cap; the accuracy
        # requirement is so low this barely affects the batch pass rate).
        return env.finish(answer)
