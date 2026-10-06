from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel spawn-and-vote policy optimised for the latency-dominated
    profile_0087 constraint set.

    The baseline template runs sequential ASC (one branch at a time),
    accumulating latency as the sum of all branch lifetimes.  Switching to
    parallel cohort actions (spawn_many / continue_many) converts that sum
    into the *max* per round, reducing wall-clock time by roughly a factor
    of the width.

    Design rationale
    ----------------
    * Latency batch <= 5135 s.  With width-3 parallel execution a 66-round
      episode adds at most ~1650 s instead of the baseline's ~21000 s at p90.
    * Accuracy mean >= 0.204.  Three independent Qwen3-4B branches with
      deterministic plurality should achieve 50-70 pct.
    * Weighted compute mean <= 1861538.  Three branches use roughly one
      quarter of the baseline twelve-branch budget.
    """

    NAME = "candidate"

    def _pick_model(self, env) -> str:
        """Select Qwen3-4B for minimum per-token latency; fall back to
        the first available model if the pool has changed."""
        preferred = "Qwen3-4B"
        if preferred in env.models:
            return preferred
        return env.models[0]

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width = 3

        # --- parallel spawn ------------------------------------------------
        opened = env.spawn_many(model, width)

        # Track which branches can still advance and their latest answers.
        can_continue = set()
        latest_answers: Dict[int, str] = {}
        for branch in opened:
            bid = branch["branch_id"]
            if branch["can_continue"]:
                can_continue.add(bid)
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                latest_answers[bid] = ans

        # --- parallel continuation rounds ----------------------------------
        max_rounds = 500  # generous safety cap; real episodes finish earlier
        for _ in range(max_rounds):
            if not can_continue:
                break

            try:
                updated = env.continue_many(list(can_continue))
            except Exception:
                # Budget exhaustion surfaces as InvalidAction; any failure
                # means we should stop continuing and vote on what we have.
                break

            can_continue.clear()
            for branch in updated:
                bid = branch["branch_id"]
                if branch["can_continue"] and not branch["pruned"]:
                    can_continue.add(bid)
                ans = valid_answer(branch.get("latest_answer"))
                if ans is not None:
                    latest_answers[bid] = ans

        # --- vote and finish -----------------------------------------------
        votes = list(latest_answers.values())
        final = deterministic_plurality(votes) if votes else None
        return env.finish(final)
