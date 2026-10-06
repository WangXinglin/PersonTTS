from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Single-branch minimal-cost policy for low accuracy thresholds.

    With profile_0093 requiring only 1.13% accuracy, the dominant constraints
    are latency (batch <= 2724s) and weighted compute (mean <= 376K).  The
    baseline burns ~1.5M weighted compute and ~5.4Ks latency by running ~12
    sequential ASC branches.  Spawning a single cheapest-model branch and
    running it to natural completion yields ~132K compute and ~463s latency
    on average while still achieving ~50-70% accuracy -- far above the 1.13%
    threshold.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pick_cheapest_model(self, models: list) -> str:
        """Select the smallest model by parameter count parsed from its name."""

        def _extract_size(name: str) -> float:
            for part in name.replace("-", " ").replace("_", " ").split():
                if part.endswith("B") and part[:-1].replace(".", "", 1).isdigit():
                    return float(part[:-1])
            return float("inf")

        return min(models, key=_extract_size)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_cheapest_model(env.models)

        # Spawn exactly one branch -- minimal cost.
        branch = env.spawn(model)

        # Run to natural stage completion so the model finishes its reasoning.
        while branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])

        answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)
