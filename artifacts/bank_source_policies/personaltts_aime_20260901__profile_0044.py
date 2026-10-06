from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Deep parallel majority-vote policy for latency-bounded accurate reasoning.

    Round-0 diagnosis
    -----------------
    The round-0 candidate spawned 5 parallel Qwen3-4B branches but allowed
    majority agreement (3/5) at checkpoint 0, before any branch had meaningful
    reasoning depth.  Premature consensus on wrong answers drove accuracy to
    30 % (need >= 42.17 %).  Latency was 67 s p99 against a 382 s batch budget
    and weighted compute was 105 K against an 8.7 M budget — huge headroom.

    The sequential baseline achieves 78 % accuracy by running every branch to
    its terminal checkpoint (3-65 checkpoints), but sequential execution blows
    the latency budget by 60x.

    Strategy
    --------
    1. spawn_many(5) opens a width-5 cohort in one parallel latency phase.
    2. continue_many advances all active branches each round; a round adds
       only the maximum single-branch checkpoint latency (~7-12 s Qwen3-4B).
    3. Branches run until all reach terminal (can_continue=False) or a safety
       cap computed from the batch latency budget is hit.
    4. Early agreement requires near-unanimous (4/5+) consensus after at
       least 5 continue rounds, preventing the premature-wrong-consensus bug.
    5. Final answer by deterministic plurality over each branch's latest
       valid answer.

    Easy problems terminate in 5-8 rounds (~60-100 s); medium in 10-20
    rounds (~120-240 s); hard problems hit the ~26-round cap (~290-330 s).
    All within the 382 s batch budget.  Expected accuracy 55-70 %, well above
    the 42.17 % threshold.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_model(env) -> str:
        """Select Qwen3-4B (data-backed, proven available), else first model."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return sorted(env.models)[0]

    @staticmethod
    def _max_continue_rounds(latency_budget: float) -> int:
        """Compute maximum parallel continue rounds that fit the batch budget.

        Conservative per-phase estimate of 12 s accounts for the parallel max
        of 5 Qwen3-4B branches, and a 15 % safety margin covers the spawn
        phase plus variability in later (deeper) checkpoints.
        """
        per_phase_est = 12.0
        safe_budget = latency_budget * 0.85
        max_phases = int(safe_budget / per_phase_est)
        # subtract 1 for the spawn_many phase; floor at 5, cap at 50
        return min(max(max_phases - 1, 5), 50)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # ---- budget-aware configuration ----
        width = 5
        latency_limit = 382.0
        if env.requirements is not None:
            latency_limit = float(env.requirements["latency_s"]["value"])
        max_rounds = self._max_continue_rounds(latency_limit)
        min_rounds_for_early = 5  # no early agreement before round 5

        # ---- phase 1: parallel spawn ----
        branches = None
        for w in (width, 3, 1):
            try:
                if w > 1:
                    branches = env.spawn_many(model, w)
                else:
                    branches = [env.spawn(model)]
                width = w
                break
            except Exception:
                continue
        if branches is None:
            return env.finish(None)

        branch_ids = set(b["branch_id"] for b in branches)

        # Track the latest valid answer per branch (survives transient nulls)
        best: Dict[Any, str] = {}
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                best[b["branch_id"]] = ans

        # ---- phase 2: parallel continue until terminal or cap ----
        for round_idx in range(max_rounds):
            state = env.observe()
            active_ids = [
                b["branch_id"]
                for b in state["branches"]
                if b["branch_id"] in branch_ids and b["can_continue"]
            ]
            if not active_ids:
                break  # every branch reached terminal

            try:
                updated = env.continue_many(active_ids)
            except Exception:
                break  # budget or pool exhausted

            for b in updated:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    best[b["branch_id"]] = ans

            # Conservative early agreement: near-unanimous after min depth
            if (round_idx >= min_rounds_for_early
                    and len(best) >= width
                    and width >= 3):
                answers = [best[bid] for bid in sorted(best)]
                counts = Counter(answers)
                top_count = max(counts.values())
                # Require all-but-one agreement (4/5 for width=5)
                if top_count >= width - 1:
                    return env.finish(deterministic_plurality(answers))

        # ---- phase 3: final plurality vote ----
        answers = [best[bid] for bid in sorted(best)]
        if not answers:
            return env.finish(None)
        return env.finish(deterministic_plurality(answers))
