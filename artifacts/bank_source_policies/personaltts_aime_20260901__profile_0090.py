from __future__ import annotations

from typing import Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Ultra-lean single-branch policy for tight compute and latency budgets.

    The profile requires only ~3.3% mean accuracy but enforces strict limits
    on weighted compute (mean <= 343 728) and latency (batch <= 5 180 s).
    The baseline blows both budgets (~4.5x on compute, latency 0% satisfied)
    by running ASC with ~12 sequential Qwen3-4B branches per episode.

    This policy spawns exactly one Qwen3-4B branch, continues only until
    a valid answer is extracted, then finishes immediately.  No refinement,
    no extra width, no unnecessary checkpoints.  Expected per-episode cost
    is roughly 1/10 of the baseline while single-branch accuracy (~40-60%)
    far exceeds the 3.3% threshold.
    """

    NAME = "candidate"

    def _pick_model(self, env) -> str:
        """Select the cheapest available model to minimise weighted compute."""
        # Qwen3-4B is the smallest model; fall back to first available
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return env.models[0]

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # Spawn a single branch (reveals checkpoint 0)
        branch = env.spawn(model)
        answer = valid_answer(branch.get("latest_answer"))

        # Continue until we obtain the first valid answer or the stage ends
        while answer is None and branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])
            answer = valid_answer(branch.get("latest_answer"))

        return env.finish(answer)
