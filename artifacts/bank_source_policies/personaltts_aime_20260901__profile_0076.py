from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Ultra-lean single-branch policy for tight compute/latency budgets.

    Strategy:
    - Spawn exactly 1 Qwen3-4B branch (cheapest model).
    - Continue only until a valid answer appears, with a hard cap on
      checkpoint count to stay well within the weighted-compute budget.
    - Finish immediately with whatever answer is available.

    For profile_0076 the accuracy threshold is just 1.68 %, so even a
    single branch with limited thinking easily satisfies accuracy while
    keeping mean weighted-compute and batch latency far below limits.
    """

    NAME = "candidate"

    # Maximum total checkpoints (spawn counts as 1).
    # With Qwen3-4B at ~3-4 k weighted-compute per checkpoint, 8 checkpoints
    # keeps the worst-case episode around 28 k wc.  Most episodes get an
    # answer by checkpoint 1-2 and stop early at ~7-10 k wc.  The budget
    # of 75.5 k mean wc is easily satisfied.
    MAX_CHECKPOINTS = 8

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _pick_model(self, env) -> str:
        """Choose the lightest available model to minimise weighted compute."""
        preferred = "Qwen3-4B"
        if preferred in env.models:
            return preferred
        return env.models[0]

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # Spawn one branch — reveals checkpoint 0
        branch = env.spawn(model)
        answer = valid_answer(branch.get("latest_answer"))
        checkpoints_seen = 1

        # Continue until we have a valid answer or exhaust our cap
        while (
            answer is None
            and branch["can_continue"]
            and checkpoints_seen < self.MAX_CHECKPOINTS
        ):
            branch = env.continue_branch(branch["branch_id"])
            answer = valid_answer(branch.get("latest_answer"))
            checkpoints_seen += 1

        return env.finish(answer)
