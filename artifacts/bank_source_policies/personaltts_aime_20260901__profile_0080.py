from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn policy that uses spawn_many/continue_many for low latency.

    The baseline ASC runs branches sequentially, causing latency to scale
    linearly with the number of branches.  This policy spawns a fixed-width
    cohort in parallel and advances them synchronously, so latency equals
    max(single-branch time) instead of sum.  Early consensus stopping
    further reduces latency on easy problems.

    Profile 0080 constraints:
      accuracy (mean)           >= 0.37   -- easily met by 5-branch majority vote
      latency_s (batch)           <= 2099   -- met by parallel execution (~60 rounds)
      weighted_compute (mean)   <= 24.3M  -- met with Qwen3-4B at width 5
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def _pick_model(self, env: Any) -> str:
        """Choose the smallest available model for minimal per-checkpoint latency."""
        models = list(env.models)
        # Prefer Qwen3-4B (known small and fast); fall back to first available.
        for preferred in ("Qwen3-4B",):
            if preferred in models:
                return preferred
        return models[0]

    def solve(self, env: Any) -> Optional[str]:
        model = self._pick_model(env)

        # --- parallel spawn --------------------------------------------------
        width = 5
        opened = env.spawn_many(model, width)

        # Track latest branch state keyed by branch_id.
        latest: Dict[int, Dict[str, Any]] = {}
        for b in opened:
            latest[b["branch_id"]] = b

        branch_ids: List[int] = list(latest.keys())

        # --- parallel continue rounds ----------------------------------------
        # Each continue_many round adds max(cohort latency), not sum.
        max_rounds = 300  # generous safety ceiling
        for _ in range(max_rounds):
            # Identify branches that can still advance.
            active_ids = [
                bid for bid in branch_ids if latest[bid]["can_continue"]
            ]
            if not active_ids:
                break

            # Advance the whole cohort one checkpoint.
            try:
                updated = env.continue_many(active_ids)
            except Exception:
                # Budget or pool exhausted -- stop and vote with what we have.
                break

            for b in updated:
                latest[b["branch_id"]] = b

            # --- early consensus check ----------------------------------------
            # Collect the current valid answers across ALL branches
            # (including those already terminal).
            answers: List[str] = []
            for bid in branch_ids:
                a = valid_answer(latest[bid].get("latest_answer"))
                if a is not None:
                    answers.append(a)

            if len(answers) >= 3:
                counts = Counter(answers)
                top_count = max(counts.values())
                if top_count >= 3:
                    # Clear majority (>= 3 out of at most 5) -- stop early.
                    winner = deterministic_plurality(answers)
                    return env.finish(winner)

        # --- final vote -------------------------------------------------------
        final_answers: List[str] = []
        for bid in branch_ids:
            a = valid_answer(latest[bid].get("latest_answer"))
            if a is not None:
                final_answers.append(a)

        answer = deterministic_plurality(final_answers) if final_answers else None
        return env.finish(answer)
