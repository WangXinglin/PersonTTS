from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Width-6 parallel voting with adaptive backup on uncertain problems.

    Improvement over the width-5+2 incumbent (utility 0.845):

    1. Increases initial parallel width from 5 to 6, giving every problem
       one extra independent vote. This directly boosts majority-vote
       accuracy on all problems, not just uncertain ones.

    2. Keeps the two-phase backup mechanism: when fewer than 4 branches
       agree after full depth, spawn 2 more backup branches (total 8)
       to improve accuracy on medium-difficulty problems.

    3. Compute budget: ~630k mean (6 branches at ~96k each = 576k base,
       plus ~28% of problems get 2 more at ~192k = +54k on average).
       Well under the 690k limit with ~60k margin.

    4. Latency: parallel execution keeps batch well under 3650s limit.
       Phase 1 max-of-6 + Phase 2 max-of-2 stays similar to incumbent.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def solve(self, env: Any) -> Optional[str]:
        model = "Qwen3-4B"
        if model not in env.models:
            model = env.models[0]

        initial_width = 6
        backup_width = 2
        strong_consensus = 4  # accept without backup if top >= 4 of 6

        # --- Phase 1: Spawn initial cohort and run all to terminal ---
        try:
            opened = env.spawn_many(model, initial_width)
        except Exception:
            return self._single_branch(env, model)

        branches: Dict[int, Dict[str, Any]] = {}
        for b in opened:
            branches[int(b["branch_id"])] = b

        self._run_to_terminal(env, branches)

        # Evaluate consensus after full depth
        top_ans, top_count = self._top_vote(branches)

        # Strong consensus: accept immediately (saves compute on easy problems)
        if top_count >= strong_consensus:
            return env.finish(top_ans)

        # --- Phase 2: Spawn backup branches for uncertain problems ---
        try:
            backup = env.spawn_many(model, backup_width)
            for b in backup:
                branches[int(b["branch_id"])] = b
            self._run_to_terminal(env, branches)
        except Exception:
            pass  # Budget/pool exhausted; vote on what we have

        # --- Final vote across all available branches ---
        return env.finish(self._plurality(branches))

    # ---- deterministic helpers ------------------------------------------------

    def _run_to_terminal(
        self, env: Any, branches: Dict[int, Dict[str, Any]]
    ) -> None:
        """Advance all continuable branches until every one reaches terminal."""
        for _ in range(300):  # safety cap well above natural max depth
            continuable = [
                bid
                for bid, b in branches.items()
                if b["can_continue"] and not b.get("pruned", False)
            ]
            if not continuable:
                break
            try:
                updated = env.continue_many(continuable)
            except Exception:
                break
            for b in updated:
                branches[int(b["branch_id"])] = b

    def _collect_valid_answers(
        self, branches: Dict[int, Dict[str, Any]]
    ) -> List[str]:
        """Return valid answers from non-pruned branches in insertion order."""
        result: List[str] = []
        for b in branches.values():
            if b.get("pruned", False):
                continue
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                result.append(ans)
        return result

    def _top_vote(
        self, branches: Dict[int, Dict[str, Any]]
    ) -> tuple:
        """Return (top_answer, top_count) or (None, 0)."""
        answers = self._collect_valid_answers(branches)
        if not answers:
            return None, 0
        counts = Counter(answers)
        top_ans, top_count = counts.most_common(1)[0]
        return top_ans, top_count

    def _plurality(
        self, branches: Dict[int, Dict[str, Any]]
    ) -> Optional[str]:
        """Deterministic plurality: highest count, first-occurrence tie-break."""
        answers = self._collect_valid_answers(branches)
        if not answers:
            return None
        counts = Counter(answers)
        top_count = counts.most_common(1)[0][1]
        for ans in answers:
            if counts[ans] == top_count:
                return ans
        return None

    def _single_branch(self, env: Any, model: str) -> Optional[str]:
        """Fallback: single branch to terminal."""
        branch = env.spawn(model)
        while branch["can_continue"]:
            try:
                branch = env.continue_branch(branch["branch_id"])
            except Exception:
                break
        return env.finish(valid_answer(branch.get("latest_answer")))
