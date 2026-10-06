from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel-probe policy that minimises latency via spawn_many/continue_many.

    The baseline runs branches sequentially (each spawn and continue adds to
    total latency).  This policy spawns a small cohort in parallel and advances
    them together with continue_many, so each phase only charges the maximum
    single-branch latency.  Early consensus stopping further reduces latency
    for easy problems.

    Design rationale for profile_0099:
      - accuracy  >= 0.018 (mean) — trivially met with any reasonable width
      - latency_s <= 3714  (batch)  — binding constraint; parallel execution
        converts O(W*C) sequential latency to O(C) parallel phases
      - weighted_compute <= 24.4M (mean) — easily met with Qwen3-4B
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pick_model(self, env) -> str:
        """Select the fastest available model (smallest first)."""
        preferred = "Qwen3-4B"
        if preferred in env.models:
            return preferred
        # Fallback: pick whatever is available
        return env.models[0]

    def _consensus_answer(
        self, branches: list, threshold: int = 2
    ) -> Optional[str]:
        """Return the answer that at least *threshold* unpruned branches agree
        on, or ``None`` if no such answer exists."""
        answers: list[str] = []
        for b in branches:
            if b["pruned"]:
                continue
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                answers.append(ans)
        if len(answers) < threshold:
            return None
        counts = Counter(answers)
        top_answer, top_count = counts.most_common(1)[0]
        if top_count >= threshold:
            return top_answer
        return None

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # Width 3 gives majority-voting robustness with small parallel
        # overhead (max of 3 per-branch latencies per phase).
        width = 3
        env.spawn_many(model, width)

        # Continue all active branches in parallel.  Check for early
        # consensus after the first continue round (branches need at
        # least one continuation to produce answers).
        max_rounds = 80          # safety cap  (~80 * 25s ≈ 2000s worst case)
        min_rounds_before_check = 1

        for round_idx in range(max_rounds):
            state = env.observe()

            # Early-consensus check: if ≥2 branches already agree, stop.
            if round_idx >= min_rounds_before_check:
                consensus = self._consensus_answer(state["branches"], threshold=2)
                if consensus is not None:
                    return env.finish(consensus)

            # Advance all branches that can still continue
            active = [
                b for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not active:
                break

            env.continue_many([b["branch_id"] for b in active])

        # All branches terminal (or cap hit) — fall through to plurality vote
        return env.finish()
