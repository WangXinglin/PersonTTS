from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel deep-reasoning with terminal-aware majority vote.

    Spawns 3 branches in parallel (via spawn_many) and advances them
    synchronously (via continue_many) until branches reach stage_terminal.
    Returns early as soon as 2 terminal branches agree on the same answer,
    saving compute on the remaining branch.  This combines the accuracy
    benefit of deep reasoning (running to terminal) with the latency
    benefit of parallel execution (max-of-cohort, not sum) and the compute
    benefit of early stopping on consensus.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)
        width = 3  # 3 parallel branches for majority vote

        # Phase 1: Spawn all branches in a single parallel cohort.
        try:
            branches = env.spawn_many(model, width)
        except Exception:
            return self._single_branch_fallback(env, model)

        branch_id_set: Set[int] = set()
        branch_id_list: List[int] = []
        terminal_answers: Dict[int, Optional[str]] = {}
        latest_answers: Dict[int, str] = {}

        for b in branches:
            bid = int(b["branch_id"])
            branch_id_set.add(bid)
            branch_id_list.append(bid)
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                latest_answers[bid] = ans
            if b["stage_terminal"]:
                terminal_answers[bid] = ans

        # Check for immediate terminal consensus.
        result = self._terminal_majority(terminal_answers)
        if result is not None:
            return env.finish(result)

        # Phase 2: Advance branches synchronously until consensus or done.
        for _ in range(500):  # safety cap; typical episodes need < 100 rounds
            state = env.observe()
            to_continue = [
                int(b["branch_id"])
                for b in state["branches"]
                if int(b["branch_id"]) in branch_id_set
                and b["can_continue"]
                and not b["pruned"]
            ]

            if not to_continue:
                break

            try:
                updated = env.continue_many(to_continue)
            except Exception:
                break

            for b in updated:
                bid = int(b["branch_id"])
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    latest_answers[bid] = ans
                if b["stage_terminal"]:
                    # Use the terminal-checkpoint answer, or fall back to
                    # the last valid answer seen during this branch.
                    terminal_answers[bid] = (
                        ans if ans is not None else latest_answers.get(bid)
                    )

            # Early exit: 2+ terminal branches agree.
            result = self._terminal_majority(terminal_answers)
            if result is not None:
                return env.finish(result)

        # Phase 3: All branches exhausted — vote with the best answers.
        votes = self._collect_votes(
            branch_id_list, terminal_answers, latest_answers
        )
        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_model(env) -> str:
        """Pick the cheapest model available (Qwen3-4B preferred)."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return env.models[0]

    @staticmethod
    def _terminal_majority(
        terminal_answers: Dict[int, Optional[str]],
    ) -> Optional[str]:
        """Return the answer if 2+ terminal branches agree, else None."""
        valid = [v for v in terminal_answers.values() if v is not None]
        if len(valid) < 2:
            return None
        counts = Counter(valid)
        best, count = counts.most_common(1)[0]
        return best if count >= 2 else None

    @staticmethod
    def _collect_votes(
        branch_ids: List[int],
        terminal_answers: Dict[int, Optional[str]],
        latest_answers: Dict[int, str],
    ) -> List[str]:
        """Gather one vote per branch, preferring terminal answers."""
        votes: List[str] = []
        for bid in branch_ids:
            if bid in terminal_answers and terminal_answers[bid] is not None:
                votes.append(terminal_answers[bid])
            elif bid in latest_answers:
                votes.append(latest_answers[bid])
        return votes

    def _single_branch_fallback(self, env, model: str) -> Optional[str]:
        """Ultra-cheap fallback when spawn_many fails."""
        b = env.spawn(model)
        best = valid_answer(b.get("latest_answer"))
        while b["can_continue"]:
            b = env.continue_branch(int(b["branch_id"]))
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                best = ans
        return env.finish(best)
