from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Wide parallel majority-vote with conservative supermajority early stop.

    Spawns 10 parallel Qwen3-4B branches (vs incumbent's 3) and runs them
    toward natural completion.  An 80 % supermajority gate (activated only
    after 5 checkpoint rounds) provides safe early stopping on easy problems
    while avoiding the incumbent's premature 2/3-agreement failures.
    Deterministic plurality vote decides the final answer.

    Budget arithmetic (Qwen3-4B, per-branch completion ~132 K weighted
    compute from baseline evidence):
      10 branches x 132 K = 1 320 K  <  1 446 K budget   (8 % margin)
    Parallel latency stays well under the batch = 3 224 s cap.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  Model / width selection                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_model_and_width(env) -> tuple:
        """Choose the smallest available model and the widest safe cohort."""
        models = env.models
        # Prefer Qwen3-4B: ~132 K wc per branch to completion
        if "Qwen3-4B" in models:
            return "Qwen3-4B", 10
        for m in models:
            if "4B" in m:
                return m, 10
        for m in models:
            if "8B" in m:
                return m, 5
        for m in models:
            if "14B" in m:
                return m, 3
        return models[0], 5

    # ------------------------------------------------------------------ #
    #  Supermajority check                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_supermajority(
        latest: Dict[int, str], threshold: int
    ) -> Optional[str]:
        """Return the answer that meets *threshold* votes, or None."""
        if len(latest) < threshold:
            return None
        counts = Counter(latest.values())
        best_ans, best_count = counts.most_common(1)[0]
        if best_count >= threshold:
            return best_ans
        return None

    # ------------------------------------------------------------------ #
    #  Main solve loop                                                    #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model, target_width = self._pick_model_and_width(env)

        # --- parallel spawn with graceful width reduction ----------------
        opened: Optional[list] = None
        width = target_width
        while width >= 1:
            try:
                opened = env.spawn_many(model, width)
                break
            except Exception:
                width -= 1

        if not opened:
            # absolute fallback: single sequential branch
            try:
                branch = env.spawn(model)
                while branch["can_continue"]:
                    branch = env.continue_branch(branch["branch_id"])
                return env.finish(valid_answer(branch.get("latest_answer")))
            except Exception:
                return env.finish(None)

        branch_ids: List[int] = [int(b["branch_id"]) for b in opened]
        bid_set: Set[int] = set(branch_ids)
        n = len(branch_ids)

        # 80 % supermajority rounded up; for n=10 -> 8, n=5 -> 4, n=3 -> 3
        supermajority_threshold: int = math.ceil(n * 0.8)

        # Minimum continue rounds before we allow supermajority stopping.
        # Prevents premature lock-in before branches reason sufficiently.
        min_rounds_before_stop: int = 5

        # --- collect checkpoint-0 answers --------------------------------
        latest: Dict[int, str] = {}
        for b in opened:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                latest[int(b["branch_id"])] = ans

        # --- parallel continue rounds ------------------------------------
        for round_num in range(500):
            state = env.observe()
            by_id = {int(b["branch_id"]): b for b in state["branches"]}
            advancing = [
                bid
                for bid in branch_ids
                if bid in bid_set
                and not by_id[bid]["pruned"]
                and by_id[bid]["can_continue"]
            ]
            if not advancing:
                break

            try:
                updated = env.continue_many(advancing)
            except Exception:
                break

            for b in updated:
                bid = int(b["branch_id"])
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    latest[bid] = ans

            # Conservative supermajority gate (only after enough rounds)
            if round_num >= min_rounds_before_stop:
                sm = self._check_supermajority(latest, supermajority_threshold)
                if sm is not None:
                    return env.finish(sm)

        # --- final deterministic plurality vote --------------------------
        votes: List[str] = list(latest.values())
        return env.finish(deterministic_plurality(votes) if votes else None)
