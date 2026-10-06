from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Latency-minimizing policy for profile_0097.

    The binding constraint is batch latency <= 1646s while accuracy >= 1.53%
    is trivially easy.  Strategy: single Qwen3-4B branch, run to stage
    terminal or a small checkpoint cap, no refinement.  This keeps
    per-problem latency to a few hundred seconds at most while a single
    branch still achieves 30-60% accuracy (far above the 1.53% floor).
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ---- helpers --------------------------------------------------------

    def _pick_model(self, env) -> str:
        """Select the smallest available model for minimum per-step latency."""
        models = list(env.models)
        # Prefer known small models in ascending size order
        for name in ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]:
            if name in models:
                return name
        return models[0]

    def _max_continues(self, env) -> int:
        """Derive a checkpoint cap from the latency budget.

        Target per-problem latency well below the batch limit so that even
        the worst three problems in a 60-problem batch stay under the cap.
        """
        reqs = env.requirements
        if reqs is not None:
            latency_limit = float(reqs["latency_s"]["value"])
            # Aim for per-problem latency < latency_limit / 4
            # with a conservative 25 s per checkpoint for Qwen3-4B
            cap = max(3, int(latency_limit / 4.0 / 25.0))
            return min(cap, 30)  # hard upper bound
        return 15  # safe default

    # ---- main solve -----------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        max_cont = self._max_continues(env)  # ~16 for profile_0097

        # Single branch keeps total latency = sum of one branch's steps.
        branch = env.spawn(model)
        bid = branch["branch_id"]

        # Early exit if spawn already produced an answer (rare but possible).
        if valid_answer(branch.get("latest_answer")) is not None and branch.get("stage_terminal"):
            return env.finish(valid_answer(branch["latest_answer"]))

        # Advance until the branch's stage terminates or we hit the cap.
        for _ in range(max_cont):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(bid)

        answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)
