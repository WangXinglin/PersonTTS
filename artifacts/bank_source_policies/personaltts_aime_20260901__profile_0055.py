from __future__ import annotations

from typing import Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Minimal single-branch policy targeting profile_0055's generous accuracy
    threshold (~16%) while staying well within the tight latency (batch <= 1131s)
    and weighted-compute (mean <= 187k) budgets.

    Strategy: spawn one Qwen3-4B branch and run all checkpoints to terminal.
    A single branch provides ~60-70% accuracy (far above 16%), costs ~50-130k
    weighted compute on average (under 187k), and has worst-case latency
    around 900-1000s per problem (under the 1131s batch threshold).
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # Spawn a single branch — checkpoint 0 is revealed immediately
        branch = env.spawn(model)

        # Run every checkpoint until the branch reaches its terminal state
        while branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])

        # Use the terminal answer (may be None if the branch never produced one)
        answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_model(self, env) -> str:
        """Pick the cheapest available model to minimise weighted compute."""
        models = list(env.models)
        # Qwen3-4B has the lowest per-token weight observed in baseline traces
        if "Qwen3-4B" in models:
            return "Qwen3-4B"
        # Fallback: prefer names that suggest smaller parameter counts
        for candidate in (
            "Qwen3-0.5B",
            "Qwen3-1.5B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
        ):
            if candidate in models:
                return candidate
        return models[0]
