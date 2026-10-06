from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel cohort with early majority stopping, optimised for tight latency.

    Strategy
    --------
    The binding constraint for profile_0091 is batch latency <= 3343 s while
    mean accuracy >= 0.2125 is very easy (baseline ~78 %).

    Sequential ASC (the incumbent) accumulates latency across every branch
    because each spawn/continue is a separate sequential step.  By contrast
    ``spawn_many`` and ``continue_many`` execute a full cohort in a single
    wall-clock step (latency = max of the cohort).  With width 3, total
    latency = (1 spawn round + N checkpoint rounds) x max-per-step latency,
    which is roughly 1/width of the sequential cost -- well inside the batch
    budget even for the hardest problems (~65 checkpoints).

    Early stopping: as soon as 2 of 3 branches agree on the same answer we
    finish immediately, saving many checkpoint rounds on easy problems.
    No refinement is used (unnecessary given the low accuracy threshold).
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pick_model(self, env) -> str:
        """Select the smallest available model to minimise per-step latency."""
        models = list(env.models)
        # Ordered by expected latency (smallest first).
        size_preference = [
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Qwen3-72B",
        ]
        for name in size_preference:
            if name in models:
                return name
        return models[0]

    def _majority_answer(
        self, answers: Dict[int, Optional[str]], threshold: int
    ) -> Optional[str]:
        """Return the leading answer if its vote count reaches *threshold*."""
        valid = [a for a in answers.values() if a is not None]
        if len(valid) < threshold:
            return None
        counts = Counter(valid)
        best, best_count = counts.most_common(1)[0]
        if best_count >= threshold:
            return best
        return None

    # ------------------------------------------------------------------
    # main solve loop
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width = 3
        majority = 2  # 2-of-3 agreement

        # --- spawn parallel cohort (one wall-clock step) ---------------
        opened = env.spawn_many(model, width)
        branch_id_set: set = set()
        answers: Dict[int, Optional[str]] = {}
        for b in opened:
            bid = b["branch_id"]
            branch_id_set.add(bid)
            answers[bid] = valid_answer(b.get("latest_answer"))

        # Check right after spawn (unlikely but free)
        winner = self._majority_answer(answers, majority)
        if winner is not None:
            return env.finish(winner)

        # --- advance in parallel checkpoint rounds ---------------------
        for _ in range(500):  # branches hit terminal well before this
            state = env.observe()
            active = [
                b
                for b in state["branches"]
                if b["branch_id"] in branch_id_set
                and b["can_continue"]
                and not b["pruned"]
            ]
            if not active:
                break

            updated = env.continue_many(
                [b["branch_id"] for b in active]
            )

            for b in updated:
                bid = b["branch_id"]
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    answers[bid] = a

            winner = self._majority_answer(answers, majority)
            if winner is not None:
                return env.finish(winner)

        # --- fallback: plurality of all collected answers --------------
        valid_list = [a for a in answers.values() if a is not None]
        if valid_list:
            return env.finish(deterministic_plurality(valid_list))
        return env.finish()
