from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy


class CandidatePolicy(ControllerPolicy):
    """Low-latency parallel policy targeting profile_0039 constraints.

    Profile requirements:
      accuracy  >= 0.334  (mean)
      latency_s <= 216.29 (batch)
      weighted_compute <= 11,981,072 (mean)

    Design rationale
    ----------------
    Both prior rounds failed the batch latency constraint: round 0 had
    too-shallow reasoning (26.7% accuracy) and round 1 had too-deep
    reasoning with 7-wide cohorts (p50 latency already 235 s).

    Baseline trace analysis shows that per-phase (continue_many) latency
    is roughly 6-10 s for width=3 cohorts, and stage_terminal varies
    hugely (3 checkpoints for easy, 65+ for hard problems).

    This policy spawns 3 parallel Qwen3-4B branches and advances them
    up to 15 continue_many rounds.  At ~8 s per phase average:
      16 phases * 10 s (conservative batch) = 160 s << 216 s

    Three-branch majority vote with per-branch accuracy ~55% at 15
    checkpoints yields ~57% ensemble accuracy, well above the 33.4%
    threshold.  Weighted compute stays well under budget since
    3 branches * 15 rounds * ~500 tokens = ~22.5K tokens << 11.98M.
    """

    NAME = "candidate"

    # Maximum parallel checkpoint rounds before stopping.
    MAX_CONTINUE_ROUNDS = 15

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _select_model(self, env) -> str:
        """Prefer Qwen3-4B for best latency / accuracy tradeoff."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return env.models[0]

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # --- parallel spawn -----------------------------------------------
        # Use 3 branches for majority voting; fall back to 1 if pool is
        # too small.
        spawned = False
        for width in (3, 1):
            try:
                if width == 1:
                    env.spawn(model)
                else:
                    env.spawn_many(model, width)
                spawned = True
                break
            except Exception:
                continue
        if not spawned:
            return env.finish()

        # --- parallel continue rounds -------------------------------------
        # Advance all active branches toward stage_terminal.
        # Cap at MAX_CONTINUE_ROUNDS to keep batch latency safely below
        # the 216 s budget.  Easy problems terminate naturally in 3-5
        # rounds; hard problems are capped but still accumulate enough
        # reasoning for a useful vote.
        for _ in range(self.MAX_CONTINUE_ROUNDS):
            state = env.observe()
            active = [
                b for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not active:
                break
            try:
                env.continue_many([b["branch_id"] for b in active])
            except Exception:
                break

        # --- finish -------------------------------------------------------
        # Deterministic plurality vote across all branch answers.
        return env.finish()
