from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn majority-vote policy optimised for tight batch latency.

    Spawns a fixed-width cohort in parallel so that latency equals the
    slowest single branch (max) rather than the sequential sum.  After
    each parallel checkpoint round the policy checks for a clear
    majority; once *majority_threshold* branches agree the episode stops
    immediately, saving latency on easy problems while still running
    hard ones to terminal for better accuracy.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_votes(branches):
        """Return list of valid answer strings from unpruned branches."""
        votes = []
        for b in branches:
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                votes.append(a)
        return votes

    @staticmethod
    def _majority_answer(votes, threshold):
        """Return the answer if it has >= *threshold* votes, else None."""
        if len(votes) < threshold:
            return None
        counts = Counter(votes)
        top_answer, top_count = counts.most_common(1)[0]
        if top_count >= threshold:
            return top_answer
        return None

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    def solve(self, env) -> Optional[str]:
        # --- model selection ------------------------------------------------
        model = "Qwen3-4B"
        if model not in env.models:
            model = env.models[0]

        # --- tunables -------------------------------------------------------
        width = 5                 # parallel branches (no extra latency)
        max_continue_rounds = 50  # generous; loop exits when all terminal
        majority_threshold = 3    # 3-of-5 majority triggers early stop

        # --- parallel spawn -------------------------------------------------
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            # pool too small; fall back to single branch run-to-terminal
            return self._single_branch_fallback(env, model)

        branch_id_set = set(b["branch_id"] for b in opened)

        # --- checkpoint loop with early-stop --------------------------------
        for _ in range(max_continue_rounds):
            state = env.observe()
            my_branches = [
                b for b in state["branches"]
                if b["branch_id"] in branch_id_set and not b["pruned"]
            ]

            # check majority vote
            votes = self._collect_votes(my_branches)
            winner = self._majority_answer(votes, majority_threshold)
            if winner is not None:
                return env.finish(winner)

            # advance all branches that can still continue
            continuable = [b for b in my_branches if b["can_continue"]]
            if not continuable:
                break

            try:
                env.continue_many([b["branch_id"] for b in continuable])
            except Exception:
                break  # budget exhausted

        # --- final vote after all rounds ------------------------------------
        state = env.observe()
        my_branches = [
            b for b in state["branches"]
            if b["branch_id"] in branch_id_set and not b["pruned"]
        ]
        votes = self._collect_votes(my_branches)
        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish()

    # ------------------------------------------------------------------
    # fallback for tiny pools
    # ------------------------------------------------------------------
    def _single_branch_fallback(self, env, model: str) -> Optional[str]:
        """Spawn one branch, run to terminal, return its answer."""
        try:
            branch = env.spawn(model)
        except Exception:
            return env.finish()
        while branch["can_continue"]:
            try:
                branch = env.continue_branch(branch["branch_id"])
            except Exception:
                break
        a = valid_answer(branch.get("latest_answer"))
        return env.finish(a)
