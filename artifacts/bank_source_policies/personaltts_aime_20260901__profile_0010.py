from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Adaptive-depth single-branch policy with late-answer tiebreaker.

    Improvements over the round-3 incumbent (6 checkpoints, standard
    plurality with first-appearance tiebreak):

    1.  **Conditional 7th checkpoint**: after 6 checkpoints, if the
        collected valid answers have no unique majority winner (tie) or
        no valid answers at all, continue for one more checkpoint.
        This spends the ~422 WC margin only on uncertain problems,
        keeping mean WC safely under 3491.

    2.  **Last-appearance tiebreaker**: when plurality counts are tied,
        prefer the answer that appeared at the latest checkpoint.
        Later checkpoints have more reasoning context (more think
        tokens consumed) so their answers are statistically more
        accurate.  Non-tied majority decisions are unchanged.
    """

    NAME = "candidate"

    BASE_CHECKPOINTS = 6   # same depth as incumbent
    MAX_CHECKPOINTS = 7    # extend by 1 for uncertain problems

    # ------------------------------------------------------------------ #
    # deterministic helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _model_size_key(name: str) -> float:
        """Parse numeric billion-parameter indicator from a model name."""
        best = math.inf
        for part in name.replace("-", " ").replace("_", " ").split():
            cleaned = part.rstrip("Bb")
            try:
                val = float(cleaned)
                if 0 < val < best:
                    best = val
            except (ValueError, OverflowError):
                pass
        return best

    @staticmethod
    def _has_unique_winner(answers: List[str]) -> bool:
        """True when exactly one answer holds the strict plurality."""
        if not answers:
            return False
        counts = Counter(answers)
        top = max(counts.values())
        return sum(1 for c in counts.values() if c == top) == 1

    @staticmethod
    def _plurality_last_tiebreak(answers: List[str]) -> Optional[str]:
        """Plurality vote; ties broken by latest appearance."""
        if not answers:
            return None
        counts = Counter(answers)
        top = max(counts.values())
        # Walk backwards through the chronological answer list
        for a in reversed(answers):
            if counts[a] == top:
                return a
        return None  # unreachable but satisfies type checker

    # ------------------------------------------------------------------ #
    # solve                                                               #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        models = list(env.models)
        if not models:
            return env.finish(None)

        # Use cheapest model to maximise affordable depth
        model = min(models, key=self._model_size_key)

        # Spawn reveals checkpoint 0
        branch = env.spawn(model)
        cp_count = 1
        answers: List[str] = []

        ans = valid_answer(branch.get("latest_answer"))
        if ans is not None:
            answers.append(ans)

        # --- base phase: same 6 checkpoints as incumbent ---
        while branch["can_continue"] and cp_count < self.BASE_CHECKPOINTS:
            branch = env.continue_branch(branch["branch_id"])
            cp_count += 1
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                answers.append(ans)

        # --- conditional extension: 1 more checkpoint if uncertain ---
        if not self._has_unique_winner(answers):
            while branch["can_continue"] and cp_count < self.MAX_CHECKPOINTS:
                branch = env.continue_branch(branch["branch_id"])
                cp_count += 1
                ans = valid_answer(branch.get("latest_answer"))
                if ans is not None:
                    answers.append(ans)

        if not answers:
            return env.finish(None)

        # --- plurality with last-appearance tiebreaker ---
        return env.finish(self._plurality_last_tiebreak(answers))
