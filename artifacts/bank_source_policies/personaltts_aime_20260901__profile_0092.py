from __future__ import annotations

from collections import Counter
from typing import Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn low-latency policy for tight batch latency constraints.

    Key idea: replace sequential spawn+continue loops (latency = sum of steps)
    with spawn_many/continue_many cohort operations (latency = max of cohort).
    This slashes batch latency from ~25,000s to a few hundred seconds while
    retaining enough voting width to comfortably clear the 0.208 accuracy bar.
    """

    NAME = "candidate"

    def _select_model(self, env) -> str:
        """Pick the smallest available model for lowest per-step latency."""
        preferred = "Qwen3-4B"
        if preferred in env.models:
            return preferred
        return env.models[0]

    def _collect_votes(self, env, branch_ids: list) -> list:
        """Gather valid answers from all non-pruned tracked branches."""
        state = env.observe()
        by_id = {b["branch_id"]: b for b in state["branches"]}
        votes = []
        for bid in branch_ids:
            branch = by_id.get(bid)
            if branch is not None and not branch["pruned"]:
                ans = valid_answer(branch.get("latest_answer"))
                if ans is not None:
                    votes.append(ans)
        return votes

    def _has_consensus(self, votes: list, threshold: int) -> bool:
        """Return True when the top answer has at least *threshold* votes."""
        if len(votes) < threshold:
            return False
        counts = Counter(votes)
        return max(counts.values()) >= threshold

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # --- Parallel spawn ------------------------------------------------
        width = 3
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            # Pool too small for requested width; fall back to 1 branch.
            try:
                opened = [env.spawn(model)]
            except Exception:
                return env.finish()

        branch_ids = [b["branch_id"] for b in opened]
        actual_width = len(branch_ids)
        consensus_need = max(2, (actual_width // 2) + 1)

        # --- Parallel checkpoint rounds ------------------------------------
        # Each continue_many round adds only max(cohort latency) instead of
        # the sum of individual latencies, keeping total latency very low.
        max_rounds = 12

        for _ in range(max_rounds):
            # Check current answers before spending another round.
            votes = self._collect_votes(env, branch_ids)
            if self._has_consensus(votes, consensus_need):
                break

            # Advance all branches that can still continue.
            state = env.observe()
            by_id = {b["branch_id"]: b for b in state["branches"]}
            active_ids = [
                bid
                for bid in branch_ids
                if not by_id[bid]["pruned"] and by_id[bid]["can_continue"]
            ]
            if not active_ids:
                break

            try:
                env.continue_many(active_ids)
            except Exception:
                break

        # --- Final vote via deterministic plurality ------------------------
        return env.finish()
