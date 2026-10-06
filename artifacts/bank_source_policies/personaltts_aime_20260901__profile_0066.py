from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Spawn-only parallel voting: high accuracy with minimal latency.

    Profile 0066 constraints
    ------------------------
      accuracy  (mean) >= 1/60   -- at least 1 correct per batch of 60
      latency   (batch)  <= 12.76s -- binding in prior rounds
      weighted_compute (mean) <= 6.12M -- easy

    Trace-informed strategy
    -----------------------
    The incumbent (round 1, utility 0.757) used 5 Qwen3-8B branches with
    early-stop at cp0; accuracy was the bottleneck because P0007 (AIME_24)
    only won plurality in 72.4% of seeds.

    Round 3 (rejected, utility 0.743) fixed accuracy by using 20 branches
    with continuation to cp1 (P0007 = 100%), but the second latency phase
    pushed batch above the 12.76 s budget in 25.7% of batches.

    This policy resolves both constraints simultaneously:
    1. Spawn 20 parallel branches of Qwen3-8B (one latency phase, ~6.1 s).
    2. Do NOT continue -- cp0 answers are sufficient for P0007 accuracy.
       Trace evidence shows ~12/20 branches give the correct answer at cp0,
       so plurality wins with overwhelming margin.
    3. By eliminating the continue phase, per-problem latency drops from
       ~12.3 s to ~6.1 s, keeping the batch batch well below 12.76 s.
    4. P0007 being correct in every batch guarantees mean accuracy = 1/60,
       exactly meeting the threshold.

    Expected outcome: ~100% satisfaction on all three constraints.
    """

    NAME = "candidate"
    TARGET_WIDTH = 20

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_size(name: str) -> float:
        """Extract approximate parameter count in billions from a model name."""
        parts = name.replace("_", "-").split("-")
        for part in parts:
            upper = part.upper()
            if upper.endswith("B") and len(upper) > 1:
                try:
                    return float(upper[:-1])
                except ValueError:
                    pass
        return 999.0

    def _pick_model(self, models: List[str]) -> str:
        """Select the largest available model with size <= 8B."""
        sized = [(self._parse_size(m), m) for m in models]
        sized.sort(key=lambda t: (-t[0], t[1]))  # largest first
        for size, model in sized:
            if size <= 8.0:
                return model
        # Fallback: smallest available model
        sized.sort(key=lambda t: (t[0], t[1]))
        return sized[0][1]

    # ------------------------------------------------------------------ #
    # main solve                                                           #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(list(env.models))

        # Check pool availability
        state = env.observe()
        pool = state["remaining_by_model"].get(model, 0)
        if pool < 1:
            return env.finish()

        width = min(self.TARGET_WIDTH, pool)

        # Phase 1: Spawn parallel branches (one latency phase only)
        branches: List[Dict[str, Any]] = []
        attempt = width
        while attempt >= 1:
            try:
                if attempt == 1:
                    branches = [env.spawn(model)]
                else:
                    branches = env.spawn_many(model, attempt)
                break
            except Exception:
                attempt = attempt // 2

        if not branches:
            return env.finish()

        # Collect cp0 answers and apply deterministic plurality voting
        votes: List[str] = []
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                votes.append(ans)

        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish()
