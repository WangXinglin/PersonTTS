from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn policy with depth cap tuned for the latency constraint.

    Round 3 spawned 7 Qwen3-4B branches and ran them to terminal (up to 33
    continue rounds), achieving 73% accuracy but violating the batch latency
    limit by ~15 s.  This policy keeps the same 7-branch parallel structure
    but caps depth based on the latency requirement, keeping the worst-case
    per-episode latency safely below the batch limit.

    With 20 max continue rounds (derived from the 390 s limit), most
    branches still reach terminal naturally, accuracy stays well above the
    33.2% threshold via 7-branch majority voting, and latency and compute
    remain comfortably within budget.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(env) -> str:
        """Select Qwen3-4B from the available model pool."""
        for m in env.models:
            if "4B" in m:
                return m
        # Fallback: pick the last model alphabetically (likely largest)
        models = sorted(env.models)
        return models[-1] if len(models) > 1 else models[0]

    @staticmethod
    def _compute_max_continues(latency_limit: float) -> int:
        """Derive max continue rounds from the batch latency limit.

        Conservative estimate: ~16 s per parallel phase for 7 Qwen3-4B
        branches (empirical range 12-16 s).  Target 90% of the limit to
        absorb per-episode variance, then subtract 1 for the spawn phase.
        """
        phases = int(latency_limit * 0.90 / 16.0)
        max_continues = phases - 1  # one phase is the spawn
        return max(10, min(max_continues, 30))

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width = 7

        # Determine depth cap from latency budget
        reqs = env.requirements
        if reqs is not None:
            latency_limit = reqs["latency_s"]["value"]
            max_continues = self._compute_max_continues(latency_limit)
        else:
            max_continues = 20

        # Phase 1: parallel spawn ----------------------------------------
        try:
            branches: List[Dict[str, Any]] = list(
                env.spawn_many(model, width)
            )
        except Exception:
            # Pool or budget insufficient for full cohort; try single branch
            try:
                branches = [env.spawn(model)]
            except Exception:
                return env.finish(None)

        # Track last valid answer per branch (robust against null flickers
        # at intermediate checkpoints).
        last_valid: Dict[int, str] = {}
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                last_valid[b["branch_id"]] = ans

        # Phase 2+: continue all active branches toward terminal ---------
        for _ in range(max_continues):
            alive = [b for b in branches if b["can_continue"]]
            if not alive:
                break
            try:
                updated = env.continue_many(
                    [b["branch_id"] for b in alive]
                )
                umap: Dict[int, Dict[str, Any]] = {
                    b["branch_id"]: b for b in updated
                }
                branches = [umap.get(b["branch_id"], b) for b in branches]
                for b in updated:
                    ans = valid_answer(b.get("latest_answer"))
                    if ans is not None:
                        last_valid[b["branch_id"]] = ans
            except Exception:
                break  # budget exhausted -- vote with what we have

        # Majority vote: prefer latest answer, fall back to last valid ---
        votes: List[str] = []
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is None:
                ans = last_valid.get(b["branch_id"])
            if ans is not None:
                votes.append(ans)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
