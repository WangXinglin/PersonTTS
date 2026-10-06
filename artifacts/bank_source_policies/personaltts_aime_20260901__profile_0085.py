from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Low-latency parallel policy targeting profile_0085.

    Requirements
    ------------
    accuracy  >= 0.109   (mean)            – very low bar
    latency_s <= 347.08  (batch)             – binding constraint
    weighted_compute <= 24 016 097 (mean)  – generous

    Strategy
    --------
    The baseline (sequential ASC) fails latency because every branch runs
    serially.  Switching to parallel spawn_many / continue_many collapses
    the per-problem latency from hundreds of sequential steps to a handful
    of cohort phases whose cost is the single slowest branch in each round.

    With 5 parallel branches and up to 5 continue rounds (6 phases total),
    estimated batch per-problem latency is roughly 60-120 s, well within
    347 s.  Five branches with several checkpoints each yield far above
    10.9 percent accuracy via plurality vote.  Weighted compute stays
    minimal.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(env) -> str:
        """Prefer Qwen3-4B (smallest / fastest); fall back to first."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return env.models[0]

    @staticmethod
    def _collect_answers(env) -> List[str]:
        """Gather valid answers from all unpruned branches."""
        state = env.observe()
        answers: List[str] = []
        for branch in state["branches"]:
            if branch["pruned"]:
                continue
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                answers.append(ans)
        return answers

    # ------------------------------------------------------------------
    # main solve loop
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        width = 5           # parallel branches for voting
        max_continues = 5   # checkpoint rounds – keeps latency low

        # Phase 1: spawn all branches in one parallel cohort
        spawned = False
        try:
            env.spawn_many(model, width)
            spawned = True
        except Exception:
            pass

        if not spawned:
            # Fall back to a single branch if full width was rejected
            try:
                env.spawn_many(model, 1)
                spawned = True
            except Exception:
                pass

        if not spawned:
            return env.finish()

        # Phase 2: advance all continuable branches a few checkpoint rounds
        for _ in range(max_continues):
            state = env.observe()
            continuable = [
                b["branch_id"]
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not continuable:
                break
            try:
                env.continue_many(continuable)
            except Exception:
                break

        # Phase 3: collect answers and finish with deterministic plurality
        answers = self._collect_answers(env)
        if answers:
            return env.finish(deterministic_plurality(answers))
        return env.finish()
