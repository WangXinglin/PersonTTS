from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Low-latency parallel probe with early majority consensus.

    Spawns a small cohort in parallel (spawn_many) so latency per round is
    the *max* across branches, not the sum.  After each continue_many round
    we check for a 2-of-3 majority; most episodes terminate in 1-5 rounds
    (20-100 s), while a hard cap on rounds keeps worst-case latency well
    below the batch budget.
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # --- tunables derived from latency budget -------------------------
        width = 3
        min_agree = 2  # majority of 3
        # Conservative cap: ~40 rounds * ~15 s/round = ~600 s worst-case,
        # well within the 1223 s batch budget.
        max_continue_rounds = 40

        # 1. Parallel spawn (one latency round)
        opened = env.spawn_many(model, width)
        latest: Dict[int, Dict[str, Any]] = {}
        branch_ids: List[int] = []
        for b in opened:
            bid = b["branch_id"]
            latest[bid] = b
            branch_ids.append(bid)

        # Check for immediate consensus at checkpoint 0
        consensus = self._check_consensus(latest, branch_ids, min_agree)
        if consensus is not None:
            return env.finish(consensus)

        # 2. Continue rounds with early consensus stopping
        for _ in range(max_continue_rounds):
            active_ids = [
                bid for bid in branch_ids if latest[bid]["can_continue"]
            ]
            if not active_ids:
                break

            updated = env.continue_many(active_ids)
            for b in updated:
                latest[b["branch_id"]] = b

            consensus = self._check_consensus(latest, branch_ids, min_agree)
            if consensus is not None:
                return env.finish(consensus)

        # 3. Fallback: deterministic plurality over whatever answers exist
        answers = self._collect_answers(latest, branch_ids)
        return env.finish(deterministic_plurality(answers))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(env) -> str:
        """Prefer smallest model for minimal per-checkpoint latency."""
        for m in env.models:
            if "4B" in m:
                return m
        return env.models[0]

    @staticmethod
    def _collect_answers(
        latest: Dict[int, Dict[str, Any]], branch_ids: List[int]
    ) -> List[str]:
        result: List[str] = []
        for bid in branch_ids:
            a = valid_answer(latest[bid].get("latest_answer"))
            if a is not None:
                result.append(a)
        return result

    @staticmethod
    def _check_consensus(
        latest: Dict[int, Dict[str, Any]],
        branch_ids: List[int],
        min_agree: int,
    ) -> Optional[str]:
        """Return the consensus answer if >= min_agree branches agree."""
        answers: List[str] = []
        for bid in branch_ids:
            a = valid_answer(latest[bid].get("latest_answer"))
            if a is not None:
                answers.append(a)
        if len(answers) < min_agree:
            return None
        counts = Counter(answers)
        best, best_count = counts.most_common(1)[0]
        if best_count >= min_agree:
            return best
        return None
