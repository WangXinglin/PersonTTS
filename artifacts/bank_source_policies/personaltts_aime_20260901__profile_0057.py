from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn policy with early consensus, optimised for tight latency budgets.

    The incumbent ASC baseline spawns branches *sequentially*, so every spawn +
    continue adds to wall-clock latency.  This policy uses ``spawn_many`` and
    ``continue_many`` so that each round contributes only the *maximum* branch
    latency, cutting overall latency by an order of magnitude.

    Tuning knobs derived from profile_0057 constraints:
      - accuracy  >= 0.266  (very easy  -> width 3 is plenty)
      - latency   <= 298.9s (binding    -> parallel + shallow depth)
      - w-compute <= 15.9M  (very easy  -> no concern)
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  helpers                                                            #
    # ------------------------------------------------------------------ #
    def _pick_model(self, env) -> str:
        """Smallest / fastest model first."""
        for name in ("Qwen3-4B",):
            if name in env.models:
                return name
        return env.models[0]

    @staticmethod
    def _consensus(branches, min_agree: int = 2):
        """Return the majority answer when >= *min_agree* unpruned branches agree."""
        answers = []
        for b in branches:
            if b["pruned"]:
                continue
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                answers.append(ans)
        if len(answers) < min_agree:
            return None
        counts = Counter(answers)
        best, best_count = counts.most_common(1)[0]
        if best_count >= min_agree:
            return best
        return None

    # ------------------------------------------------------------------ #
    #  single-branch fallback (used when spawn_many is unavailable)       #
    # ------------------------------------------------------------------ #
    def _solve_single(self, env, model: str, max_depth: int) -> Optional[str]:
        branch = env.spawn(model)
        for _ in range(max_depth):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(branch["branch_id"])
        return env.finish()

    # ------------------------------------------------------------------ #
    #  main entry point                                                   #
    # ------------------------------------------------------------------ #
    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # --- budget-aware parameters -----------------------------------
        # Use requirements when available; fall back to safe defaults.
        reqs = env.requirements
        if reqs is not None:
            latency_limit = reqs["latency_s"]["value"]
        else:
            latency_limit = 300.0

        # Conservative per-parallel-round estimate (seconds).
        # Covers the max-of-width factor plus variance at batch per-step.
        est_round_s = 28.0
        width = 3
        # Reserve 1 round for spawn_many itself.
        max_continue_rounds = max(1, int(latency_limit / est_round_s) - 1)
        max_continue_rounds = min(max_continue_rounds, 20)
        consensus_threshold = 2  # stop as soon as 2 of 3 branches agree

        # --- spawn parallel cohort -------------------------------------
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            # Resource shortage -> fall back to single branch.
            return self._solve_single(env, model, max_continue_rounds)

        # --- parallel continue rounds with early consensus -------------
        for _ in range(max_continue_rounds):
            state = env.observe()
            active_ids = [
                b["branch_id"]
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not active_ids:
                break  # all branches terminated

            try:
                env.continue_many(active_ids)
            except Exception:
                break  # budget exhausted

            # Check for early consensus after this round.
            state = env.observe()
            winner = self._consensus(state["branches"], consensus_threshold)
            if winner is not None:
                return env.finish(winner)

        # --- no early consensus -> deterministic plurality -------------
        return env.finish()
