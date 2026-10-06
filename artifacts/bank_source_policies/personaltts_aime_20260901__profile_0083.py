from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-cohort policy optimized for tight batch latency constraints.

    The baseline uses sequential ASC (spawn one branch, continue all its
    checkpoints, check confidence, repeat), which creates additive latency
    across all branches.  This policy instead uses spawn_many/continue_many
    to run a fixed-width parallel cohort, reducing wall-clock latency from
    the SUM of per-branch latency to the MAX per parallel step.

    Early consensus stopping further minimizes latency for easy problems.
    With the very low accuracy threshold (0.2658), a width-5 cohort with
    moderate depth easily meets accuracy while keeping batch latency far
    below the limit.
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # Parallel width: 5 branches give reliable plurality voting
        width = 5
        # Max continuation rounds: each adds one parallel-max latency step.
        # With ~20-40s per step worst case, 15 rounds yields ~300-600s,
        # well under the 1142s batch limit.
        max_rounds = 15
        # Stop early when at least 2 branches agree on an answer.
        consensus_min = 2

        # Spawn all branches in one parallel step
        opened = self._safe_spawn(env, model, width)
        if not opened:
            return env.finish(None)

        branch_ids = [b["branch_id"] for b in opened]

        # Check initial answers from checkpoint 0
        answer = self._check_consensus(env, branch_ids, consensus_min)
        if answer is not None:
            return env.finish(answer)

        # Continue in parallel rounds with early consensus stopping
        for _ in range(max_rounds):
            advancing = self._get_advancing(env, branch_ids)
            if not advancing:
                break

            if not self._safe_continue(env, advancing):
                break

            answer = self._check_consensus(env, branch_ids, consensus_min)
            if answer is not None:
                return env.finish(answer)

        # No strong consensus reached; use plurality vote across all branches
        answer = self._get_plurality(env, branch_ids)
        return env.finish(answer)

    def _select_model(self, env) -> str:
        """Pick the smallest available model for fastest per-checkpoint latency."""
        models = list(env.models)
        for size_tag in ["4B", "7B", "8B", "14B", "32B", "72B"]:
            for m in models:
                if size_tag in m:
                    return m
        return models[0]

    def _safe_spawn(self, env, model: str, target_width: int) -> list:
        """Spawn a parallel cohort, falling back to smaller widths on failure."""
        for w in range(target_width, 0, -1):
            try:
                if w > 1:
                    return list(env.spawn_many(model, w))
                return [env.spawn(model)]
            except Exception:
                continue
        return []

    def _safe_continue(self, env, branch_ids: list) -> bool:
        """Continue branches in parallel. Returns False on failure."""
        try:
            env.continue_many(branch_ids)
            return True
        except Exception:
            return False

    def _check_consensus(self, env, branch_ids: list, min_votes: int) -> Optional[str]:
        """Return the top answer if it has min_votes or more, else None."""
        state = env.observe()
        by_id = {b["branch_id"]: b for b in state["branches"]}
        answers = []
        for bid in branch_ids:
            b = by_id.get(bid)
            if b is not None and not b["pruned"]:
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    answers.append(a)
        if not answers:
            return None
        counts = Counter(answers)
        best_answer, best_count = counts.most_common(1)[0]
        if best_count >= min_votes:
            return best_answer
        return None

    def _get_advancing(self, env, branch_ids: list) -> list:
        """Get branch IDs that can still be continued."""
        state = env.observe()
        by_id = {b["branch_id"]: b for b in state["branches"]}
        return [
            bid for bid in branch_ids
            if by_id.get(bid) is not None
            and not by_id[bid]["pruned"]
            and by_id[bid]["can_continue"]
        ]

    def _get_plurality(self, env, branch_ids: list) -> Optional[str]:
        """Get the plurality answer across active branches."""
        state = env.observe()
        by_id = {b["branch_id"]: b for b in state["branches"]}
        answers = []
        for bid in branch_ids:
            b = by_id.get(bid)
            if b is not None and not b["pruned"]:
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    answers.append(a)
        return deterministic_plurality(answers) if answers else None
