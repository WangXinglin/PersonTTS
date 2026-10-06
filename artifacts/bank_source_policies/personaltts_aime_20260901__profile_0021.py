from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Latency-calibrated parallel voting policy for profile_0021.

    Diagnosis
    ---------
    - Incumbent template (ASC sequential): accuracy 78 % OK, but batch latency
      ~25 000 s vs the 762.5 s hard limit.  Every batch fails latency.
    - Round 0 (parallel + aggressive majority early-stop at any depth):
      latency ~95 s OK, but accuracy only 52.6 % < 63.7 %.  Stopped way
      too early (~10 rounds avg vs ~27 needed).
    - Round 1 (parallel full-depth + selective refine): accuracy 76.3 % OK,
      but batch latency ~860 s, exceeding the 762.5 s limit by ~100 s.
      Root cause: per-phase latency was estimated at 11 s but the true
      value is ~16 s for 8× Qwen3-4B branches.  The resulting phase cap
      of 58 was far too generous.  Refinement on 14 % of episodes added
      extra latency without enough benefit.

    Fix
    ---
    1. Keep all work parallel (spawn_many / continue_many) so latency =
       sum-of-max-per-phase, not sequential sum.
    2. Recalibrate the phase cap using observed 16.5 s / phase (with
       margin).  For budget 762.5 s → cap = int(762.5/16.5) - 3 = 43
       continue rounds.  Max latency ≈ 44 × 16.5 = 726 s, safely under
       762.5 s even with per-phase variance.
    3. Remove all refinement to eliminate tail-latency spikes.
    4. Add conservative early stopping: only after ≥ 15 rounds AND a
       supermajority (≥ 75 % of valid votes agree).  This safely
       shortens easy/medium problems without the accuracy cliff that
       round 0 caused.
    5. Maintain 8 branches for reliable plurality voting.

    Expected:
      accuracy  ~73-76 %  (above 63.7 %)
      batch lat.  ~700-730 s  (below 762.5 s)
      compute   ~600-800 K  (below 20.79 M)
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(env) -> str:
        """Select the fastest available model."""
        preferred = [
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Qwen3-72B",
        ]
        for m in preferred:
            if m in env.models:
                return m
        return env.models[0]

    @staticmethod
    def _collect_votes(
        branch_state: Dict[int, Dict[str, Any]],
    ) -> List[str]:
        """Gather valid answers from all unpruned branches."""
        votes: List[str] = []
        for b in branch_state.values():
            if not b.get("pruned", False):
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    votes.append(a)
        return votes

    @staticmethod
    def _supermajority(votes: List[str], total_width: int) -> bool:
        """True when enough valid votes exist and ≥75 % agree."""
        # Need at least 75 % of branches to have voted
        min_votes = max(3, (total_width * 3 + 3) // 4)
        if len(votes) < min_votes:
            return False
        counts = Counter(votes)
        top_count = max(counts.values())
        # Top answer holds ≥ 75 % of all valid votes
        return top_count * 4 >= len(votes) * 3

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # Read latency budget from profile requirements
        reqs = env.requirements
        lat_budget = reqs["latency_s"]["value"] if reqs else 762.5

        target_width = 8

        # ---- Phase 1: parallel spawn --------------------------------
        opened = None
        for w in (target_width, 6, 4, 2, 1):
            try:
                opened = env.spawn_many(model, w)
                break
            except Exception:
                continue
        if opened is None:
            try:
                opened = [env.spawn(model)]
            except Exception:
                return env.finish()

        branch_state: Dict[int, Dict[str, Any]] = {}
        for b in opened:
            branch_state[b["branch_id"]] = b
        actual_width = len(opened)

        # ---- Phase cap: calibrated on observed ~16-16.5 s/phase -----
        # Total phases = 1 (spawn) + phase_cap (continues).
        # Budget equation: (phase_cap + 1) * per_phase_s <= lat_budget
        # Subtract 3 extra phases as safety margin for per-phase variance.
        per_phase_s = 16.5
        phase_cap = max(15, int(lat_budget / per_phase_s) - 3)

        # ---- Early stop parameters ----------------------------------
        # Only activate after sufficient depth to avoid round-0's mistake.
        # Require supermajority (75 %) so early stop is highly reliable.
        min_depth_for_stop = 15

        # ---- Phase 2: parallel continue with early stopping ---------
        for round_idx in range(phase_cap):
            # Check early stop after minimum depth
            if round_idx >= min_depth_for_stop:
                votes = self._collect_votes(branch_state)
                if self._supermajority(votes, actual_width):
                    break

            # Advance all branches that can still continue
            can_advance = [
                bid
                for bid, b in branch_state.items()
                if b.get("can_continue", False)
                and not b.get("pruned", False)
            ]
            if not can_advance:
                break

            try:
                updated = env.continue_many(can_advance)
            except Exception:
                break

            for b in updated:
                branch_state[b["branch_id"]] = b

        # ---- Phase 3: final plurality vote (no refinement) ----------
        final_votes = self._collect_votes(branch_state)
        answer = deterministic_plurality(final_votes) if final_votes else None
        return env.finish(answer)
