from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel deep-reasoning policy targeting profile_0035.

    The round-0 incumbent (W=5, D=15, K=3) achieves only 44.5% accuracy because:
      1. Consensus K=3 fires at checkpoint 0 on wrong initial guesses.
      2. Only 15 continue rounds limits per-branch reasoning quality.

    This policy fixes both issues by exploiting the enormous latency/compute
    headroom (73 s vs 487.5 s budget; 161 K vs 4.2 M compute budget):
      - Width 9: more independent votes for robust plurality.
      - Depth 30: nearly matches baseline per-branch depth (~37 checkpoints).
      - Warmup 5: suppresses consensus before branches have reasoned enough.
      - K=6/9: only early-stop on 67%+ agreement, avoiding false consensus.

    Worst-case resource estimates (all rounds, no early stop):
      weighted_compute ~ 9 x 31 x 530 x 6.67 ~ 985 K  (budget 4.2 M)
      latency          ~ 31 x ~10 s           ~ 310 s  (budget 487.5 s batch)
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  Main solve loop                                                     #
    # ------------------------------------------------------------------ #
    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # --- hyper-parameters tuned for profile_0035 ---
        width = 9           # parallel branches for majority voting
        max_rounds = 30     # checkpoint continue rounds after spawn
        warmup = 5          # suppress consensus check for first N rounds
        consensus_k = 6     # require 6/9 for early stop

        # ---- parallel spawn ----
        opened = env.spawn_many(model, width)

        # Per-branch answer tracking
        current_answer: Dict[int, Optional[str]] = {}
        best_answer: Dict[int, str] = {}          # most-recent valid

        for b in opened:
            bid = b["branch_id"]
            a = valid_answer(b.get("latest_answer"))
            current_answer[bid] = a
            if a is not None:
                best_answer[bid] = a

        # ---- parallel continue rounds ----
        active = [b for b in opened if b["can_continue"]]

        for round_idx in range(max_rounds):
            if not active:
                break

            advancing_ids = [b["branch_id"] for b in active]
            try:
                updated = env.continue_many(advancing_ids)
            except Exception:
                break  # budget or pool exhausted

            for b in updated:
                bid = b["branch_id"]
                a = valid_answer(b.get("latest_answer"))
                current_answer[bid] = a
                if a is not None:
                    best_answer[bid] = a

            # After warmup, allow early stop on strong consensus
            if round_idx >= warmup:
                winner = self._consensus(current_answer, consensus_k)
                if winner is not None:
                    return env.finish(winner)

            active = [b for b in updated if b["can_continue"]]

        # ---- fallback: plurality over best available answers ----
        votes = list(best_answer.values())
        final = deterministic_plurality(votes) if votes else None
        return env.finish(final)

    # ------------------------------------------------------------------ #
    #  Deterministic helpers                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _consensus(
        current_answer: Dict[int, Optional[str]],
        threshold: int,
    ) -> Optional[str]:
        """Return leading answer if *threshold*+ branches currently show it."""
        valid = [a for a in current_answer.values() if a is not None]
        if len(valid) < threshold:
            return None
        counts = Counter(valid)
        top_answer, top_count = counts.most_common(1)[0]
        if top_count >= threshold:
            return top_answer
        return None

    @staticmethod
    def _select_model(env) -> str:
        """Prefer Qwen3-4B for well-characterised latency/compute behaviour."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        # Fallback: lexicographic first available model
        return sorted(env.models)[0]
