from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Minimal single-branch policy optimised for profile_0052's generous
    accuracy floor (>=0.149) and tight compute/latency ceilings.

    Strategy
    --------
    1. Spawn one Qwen3-4B branch (cheapest available model).
    2. If the spawn already produced a valid answer, finish immediately.
    3. Otherwise continue checkpoints until a valid answer appears,
       the branch reaches its terminal checkpoint, or a safety cap
       is hit.
    4. Return the first valid answer observed, or None.

    This avoids the multi-branch sequential cost of ASC/ESC while
    comfortably clearing the 14.9 % accuracy bar with single-attempt
    answers from Qwen3-4B.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pick_cheapest_model(env) -> str:
        """Return Qwen3-4B if available, else the lexicographically first
        model name (a rough proxy for smallest)."""
        models = env.models
        if "Qwen3-4B" in models:
            return "Qwen3-4B"
        return sorted(models)[0]

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self.pick_cheapest_model(env)

        # 1. Spawn a single branch
        branch = env.spawn(model)
        answer = valid_answer(branch.get("latest_answer"))
        if answer is not None:
            return env.finish(answer)

        # 2. Continue until a valid answer appears or the branch ends.
        #    Safety cap avoids unbounded cost on pathological inputs.
        max_continues = 50
        for _ in range(max_continues):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(branch["branch_id"])
            answer = valid_answer(branch.get("latest_answer"))
            if answer is not None:
                return env.finish(answer)

        # 3. No valid answer found anywhere; finish with None
        return env.finish(None)
