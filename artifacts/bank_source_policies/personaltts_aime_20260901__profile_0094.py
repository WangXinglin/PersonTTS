from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    asc_posterior_confidence,
    deterministic_plurality,
    top_two_counts,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn latency-optimized policy with ASC early stopping.

    The incumbent fails entirely on latency (batch ~25,000 s vs limit ~1,934 s)
    because it runs branches sequentially.  This policy spawns a small cohort
    in parallel via spawn_many / continue_many so that total latency equals
    roughly *one* branch's execution time instead of the sum of many.

    Width-3 majority voting preserves accuracy well above the 13.9 % threshold
    while keeping compute under budget.  ASC-based early stopping lets easy
    problems finish in very few parallel rounds.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_answers(
        branches: List[Dict[str, Any]],
        branch_id_set: set,
        latest: Dict[int, Optional[str]],
    ) -> None:
        """Update *latest* with valid answers from *branches*."""
        for b in branches:
            bid = b["branch_id"]
            if bid not in branch_id_set:
                continue
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                latest[bid] = ans

    @staticmethod
    def _check_early_stop(
        latest: Dict[int, Optional[str]],
        confidence_threshold: float,
    ) -> Optional[str]:
        """Return the winning answer if ASC confidence is met, else None."""
        votes = list(latest.values())
        if len(votes) < 2:
            return None
        winner, top1, top2, tied = top_two_counts(votes)
        if tied:
            return None
        if asc_posterior_confidence(top1, top2) >= confidence_threshold:
            return winner
        return None

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        # Select fastest / smallest model available.
        model = "Qwen3-4B" if "Qwen3-4B" in env.models else env.models[0]

        width = 3
        confidence_threshold = 0.65  # stops on 2-of-3 agreement

        # --- spawn cohort in parallel --------------------------------
        spawn_ok = True
        opened = None
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            spawn_ok = False

        if not spawn_ok or opened is None:
            # Fallback: single branch to completion
            return self._single_branch_fallback(env, model)

        branch_ids = [b["branch_id"] for b in opened]
        branch_id_set = set(branch_ids)

        # Track the latest valid answer per branch.
        latest_answers: Dict[int, Optional[str]] = {}
        self._collect_answers(opened, branch_id_set, latest_answers)

        # --- parallel continue rounds --------------------------------
        for _ in range(200):  # generous safety cap
            state = env.observe()
            continuable = [
                b["branch_id"]
                for b in state["branches"]
                if b["branch_id"] in branch_id_set and b["can_continue"]
            ]
            if not continuable:
                break

            updated = None
            try:
                updated = env.continue_many(continuable)
            except Exception:
                break

            if updated is not None:
                self._collect_answers(updated, branch_id_set, latest_answers)

            # ASC-based early stopping
            winner = self._check_early_stop(latest_answers, confidence_threshold)
            if winner is not None:
                return env.finish(winner)

        # --- final majority vote -------------------------------------
        votes = list(latest_answers.values())
        final = deterministic_plurality(votes) if votes else None
        return env.finish(final)

    # ------------------------------------------------------------------
    # fallback for resource-constrained episodes
    # ------------------------------------------------------------------

    def _single_branch_fallback(self, env, model: str) -> Optional[str]:
        """Spawn one branch and run it to natural termination."""
        branch = env.spawn(model)
        while branch["can_continue"]:
            try:
                branch = env.continue_branch(branch["branch_id"])
            except Exception:
                break
        ans = valid_answer(branch.get("latest_answer"))
        return env.finish(ans)
