from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    top_two_counts,
    usable_answers,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Parallel-branch policy optimised for latency-constrained profiles.

    Core idea: replace the sequential spawn-run-vote loop of ASC with
    spawn_many + continue_many so that every round is a single parallel
    latency step (max of the cohort) instead of N sequential steps.

    This shrinks per-episode wall-clock latency by roughly the width factor
    while preserving accuracy through majority voting and early consensus
    stopping.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _majority_winner(votes: list, threshold: int):
        """Return the answer if it has >= *threshold* votes, else None."""
        if len(votes) < threshold:
            return None
        winner, top1, _top2, tied = top_two_counts(votes)
        if tied or top1 < threshold:
            return None
        return winner

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        # --- Model selection -----------------------------------------
        model = "Qwen3-4B"
        if model not in env.models:
            model = env.models[0]

        # --- Tuning knobs -------------------------------------------
        width = 5
        consensus = 3          # 3 of 5 must agree for early stop
        warmup_rounds = 4      # let answers stabilise before checking
        max_continue_rounds = 20  # hard cap on parallel continue rounds

        # --- Spawn cohort in parallel (one latency step) ------------
        opened = self._safe_spawn_many(env, model, width)
        if opened is None:
            # Budget too tight for 5 -- try 3
            opened = self._safe_spawn_many(env, model, 3)
            width = 3
            consensus = 2
        if opened is None:
            # Absolute minimum: single branch
            return self._single_branch_fallback(env, model)

        # --- Parallel continue rounds --------------------------------
        for round_idx in range(max_continue_rounds):
            state = env.observe()
            advancing = [
                b["branch_id"]
                for b in state["branches"]
                if b["can_continue"]
            ]
            if not advancing:
                break  # every branch has reached stage terminal

            if not self._safe_continue_many(env, advancing):
                break  # compute budget exhausted

            # After a warmup period, check for majority consensus
            if round_idx >= warmup_rounds:
                state = env.observe()
                votes = usable_answers(state["branches"])
                winner = self._majority_winner(votes, consensus)
                if winner is not None:
                    return env.finish(winner)

        # --- Final plurality vote ------------------------------------
        state = env.observe()
        votes = usable_answers(state["branches"])
        answer = deterministic_plurality(votes)
        return env.finish(answer)

    # ------------------------------------------------------------------
    # Safe wrappers (catch budget / pool exhaustion without reflection)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_spawn_many(env, model: str, count: int):
        """Return opened branches or None on any spawn failure."""
        try:
            return env.spawn_many(model, count)
        except Exception:
            return None

    @staticmethod
    def _safe_continue_many(env, branch_ids: list) -> bool:
        """Return True if continue succeeded, False otherwise."""
        try:
            env.continue_many(branch_ids)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Fallback for extremely tight budgets
    # ------------------------------------------------------------------

    def _single_branch_fallback(self, env, model: str) -> Optional[str]:
        """Spawn one branch, run to terminal, return its answer."""
        try:
            branch = env.spawn(model)
        except Exception:
            return env.finish(None)
        while branch["can_continue"]:
            try:
                branch = env.continue_branch(branch["branch_id"])
            except Exception:
                break
        answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)
