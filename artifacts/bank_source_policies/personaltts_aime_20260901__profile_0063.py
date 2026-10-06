from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Wide parallel probe with deep continuation and iterative refinement.

    Improvements over the round-0 incumbent (utility 0.949):

    1. Width 5 -> up to 40 parallel branches for robust majority voting.
       The accuracy requirement is only 0.1535, so even moderate per-branch
       accuracy (~20-30%) yields strong plurality with many independent votes.

    2. Depth ~4 -> ~16 continue rounds for deeper per-branch reasoning.
       Each branch has 4x more thinking time, improving per-branch accuracy.

    3. No premature early stop.  The round-0 policy stopped when every branch
       had a non-null answer; this policy lets branches keep reasoning until
       the latency budget is consumed or the stage terminates.

    4. Automatic refinement.  When all branches reach a terminal stage, they
       are refined (given feedback) and continued in a new stage.  This adds
       a second (or third) round of reasoning at no width cost.

    The latency budget (batch <= 167 s) is the binding constraint.  At ~7 s per
    parallel phase, 17 phases give ~120 s mean / ~130 s batch, well within budget.
    Weighted compute scales with width * depth but stays far under the 22.5 M
    mean limit (~2.3 M estimated).
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ---- helpers --------------------------------------------------------

    def _select_model(self, env) -> str:
        """Pick the smallest available model for lowest per-phase latency."""
        models = list(env.models)
        for keyword in ("4B", "7B", "8B", "14B", "32B", "72B"):
            for m in models:
                if keyword in m:
                    return m
        return models[0]

    def _extract_votes(self, branches: List[Dict[str, Any]]) -> List[str]:
        """Collect one best answer per unpruned branch.

        First tries latest_answer; on null, falls back to the most recent
        valid entry in answer_history so that transient-null checkpoints
        do not discard earlier observed answers.
        """
        votes: List[str] = []
        for b in branches:
            if b["pruned"]:
                continue
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                votes.append(ans)
                continue
            history = b.get("answer_history")
            if history is not None:
                for past in reversed(history):
                    ans = valid_answer(past)
                    if ans is not None:
                        votes.append(ans)
                        break
        return votes

    # ---- main solve -----------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # --- read latency budget from requirements -----------------------
        reqs = env.requirements
        latency_limit = 200.0
        if reqs is not None:
            latency_limit = float(reqs["latency_s"]["value"])

        # --- estimate per-phase latency ----------------------------------
        est_phase_s = 7.0  # default for small models (4B)
        if any(tag in model for tag in ("14B", "32B", "72B")):
            est_phase_s = 20.0
        elif any(tag in model for tag in ("7B", "8B")):
            est_phase_s = 10.0

        # Budget: use 75% of latency limit for safety margin
        total_budget_phases = max(3, int(latency_limit * 0.75 / est_phase_s))

        # --- spawn a wide parallel cohort --------------------------------
        opened = None
        for w in [40, 30, 20, 15, 10, 7, 5, 3, 1]:
            try:
                opened = env.spawn_many(model, w)
                break
            except Exception:
                continue

        if opened is None:
            return env.finish()

        phases_used = 1  # spawn counts as one phase

        # --- main loop: continue branches, refine when all terminal ------
        while phases_used < total_budget_phases:
            state = env.observe()
            branches = [b for b in state["branches"] if not b["pruned"]]

            continuable = [b["branch_id"] for b in branches if b["can_continue"]]
            if continuable:
                try:
                    env.continue_many(continuable)
                    phases_used += 1
                except Exception:
                    break
            else:
                # All branches are terminal or stopped -- try refinement
                refinable = [b["branch_id"] for b in branches
                             if b["can_refine"]]
                if refinable and phases_used + 2 <= total_budget_phases:
                    try:
                        env.refine_many(refinable)
                        phases_used += 1
                    except Exception:
                        break
                else:
                    break  # nothing more to do

        # --- vote and finish ---------------------------------------------
        state = env.observe()
        votes = self._extract_votes(state["branches"])
        winner = deterministic_plurality(votes) if votes else None
        return env.finish(winner)
