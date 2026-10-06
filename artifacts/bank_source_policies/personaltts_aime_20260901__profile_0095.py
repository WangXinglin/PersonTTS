from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Parallel-cohort majority-vote policy that minimises latency.

    The baseline fails the batch-latency constraint because it spawns branches
    sequentially (ASC).  This policy uses ``spawn_many`` and ``continue_many``
    to run a small cohort in parallel, collapsing N*K sequential steps into K
    parallel rounds.  With only ~10 % accuracy required, three parallel
    branches with majority voting are more than sufficient.
    """

    NAME = "candidate"

    # ---- tunables (class-level constants) ----
    WIDTH = 3          # parallel branches per problem
    MAX_CKPT = 200     # safety cap on continue rounds

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(env) -> str:
        """Choose the smallest available model to minimise per-token latency."""
        models = list(env.models)
        for name in ("Qwen3-4B",):
            if name in models:
                return name
        return models[0]

    @staticmethod
    def _collect_votes(env, branch_ids: List[int]) -> List[str]:
        """Gather valid answers from the listed branches."""
        state = env.observe()
        by_id: Dict[int, Dict[str, Any]] = {
            int(b["branch_id"]): b for b in state["branches"]
        }
        votes: List[str] = []
        for bid in branch_ids:
            branch = by_id.get(bid)
            if branch is None or branch["pruned"]:
                continue
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                votes.append(ans)
        return votes

    # ------------------------------------------------------------------
    # main solver
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width = self.WIDTH

        # --- spawn cohort in parallel ---
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            # Pool too small for full cohort – try single branch.
            try:
                opened = [env.spawn(model)]
            except Exception:
                return env.finish(None)

        branch_ids = [int(b["branch_id"]) for b in opened]

        # --- advance all branches to terminal in parallel rounds ---
        for _ in range(self.MAX_CKPT):
            state = env.observe()
            by_id: Dict[int, Dict[str, Any]] = {
                int(b["branch_id"]): b for b in state["branches"]
            }
            advancing = [
                bid for bid in branch_ids
                if not by_id[bid]["pruned"] and by_id[bid]["can_continue"]
            ]
            if not advancing:
                break
            try:
                env.continue_many(advancing)
            except Exception:
                break  # budget exhausted – vote with what we have

        # --- deterministic majority vote ---
        votes = self._collect_votes(env, branch_ids)
        answer = deterministic_plurality(votes)
        return env.finish(answer)
