from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Ultra-wide parallel majority-vote policy for profile_0007.

    Requirements:
      accuracy   >= 0.369  (mean)
      latency_s  <= 61.24  (batch)
      weighted_compute <= 24.2M (mean)

    Key improvement over round 3 (15 branches, utility 0.989):
      - Increase width dramatically (target 31+ branches) to exploit the
        massive compute headroom (409K used vs 24.2M budget = 59x)
      - Evidence from rounds 2-3: latency barely changes between 7 and 15
        branches (both ~55-58s) because Qwen3-4B per-branch latency has
        extremely low variance, so max-of-N increases negligibly with N
      - More branches yield more valid answers at terminal depth and more
        robust plurality voting, pushing episode accuracy higher
      - Goal: convert the 11 failing accuracy batches by boosting per-
        episode accuracy another ~2-3 percentage points

    Latency model:
      spawn_many(N): 1 parallel round (~7s)
      continue_many x7: 7 parallel rounds (~7s each)
      Total: 8 rounds x ~7.2s = ~57.6s avg, well within 61.24s batch

    Compute model:
      31 branches x ~4100 raw tokens x 6.67 weight = ~847K per episode
      Even 63 branches = ~1.72M per episode, still 14x below budget
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _select_model(self, env: Any) -> str:
        """Pick the smallest available model for lowest per-step latency."""
        models = list(env.models)
        # Prefer Qwen3-4B (well-characterized in traces: low latency variance)
        for m in models:
            if "4B" in m:
                return m
        # Fallback: pick the model with smallest parameter count
        def _size(name: str) -> float:
            for part in name.replace("_", "-").split("-"):
                up = part.upper()
                if up.endswith("B"):
                    try:
                        return float(up[:-1])
                    except ValueError:
                        pass
            return float("inf")
        return min(models, key=_size)

    def _find_max_width(self, env: Any, model: str) -> int:
        """Find the largest affordable width using budget check.

        Tries descending target widths until can_afford_complete_candidates
        returns True. Uses checkpoints_per_stage=8 (spawn at cp0, 7 continues
        to cp7) and max_refine_rounds=0 (no refinement).
        """
        targets = [63, 55, 49, 45, 41, 37, 35, 33, 31, 29, 27, 25,
                   23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
        for w in targets:
            try:
                if env.can_afford_complete_candidates(
                    model,
                    count=w,
                    checkpoints_per_stage=8,
                    max_refine_rounds=0,
                ):
                    return w
            except Exception:
                continue
        return 0

    def _safe_spawn(self, env: Any, model: str, width: int) -> List:
        """Spawn branches with progressive fallback to smaller widths."""
        targets = [width]
        # Add fallback widths below the target
        fallbacks = [63, 55, 49, 45, 41, 37, 35, 33, 31, 29, 27, 25,
                     23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
        for f in fallbacks:
            if f < width:
                targets.append(f)
        for w in targets:
            try:
                if w > 1:
                    return list(env.spawn_many(model, w))
                else:
                    return [env.spawn(model)]
            except Exception:
                continue
        return []

    def _collect_votes(self, env: Any) -> List[str]:
        """Collect valid latest answers from all active (non-pruned) branches."""
        votes: List[str] = []
        for b in env.observe()["branches"]:
            if not b["pruned"]:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    votes.append(ans)
        return votes

    # ------------------------------------------------------------------ #
    # main                                                                 #
    # ------------------------------------------------------------------ #

    def solve(self, env: Any) -> Optional[str]:
        model = self._select_model(env)

        # --- Configuration ---
        # Maximize width to boost plurality vote accuracy.
        # Evidence: latency barely changes with width (7 vs 15 branches
        # had nearly identical latency distributions), compute budget is
        # 59x the current usage, and more branches = more correct votes.
        max_continues = 7  # 7 continue rounds = checkpoint 7 max depth

        # Find the widest affordable width
        width = self._find_max_width(env, model)
        if width == 0:
            return env.finish(None)

        # Phase 1: Parallel spawn
        branches = self._safe_spawn(env, model, width)
        if not branches:
            return env.finish(None)

        # Phase 2: Deepen ALL branches via parallel continue rounds
        # Continue every active branch that can_continue through all
        # 7 rounds. Branches that terminate early are automatically
        # excluded. Deeper reasoning produces better answers.
        for _ in range(max_continues):
            state = env.observe()
            continuable = [
                b["branch_id"]
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not continuable:
                break  # All branches terminal
            try:
                env.continue_many(continuable)
            except Exception:
                break  # Budget or pool exhausted

        # Phase 3: Deterministic plurality vote over all branches
        votes = self._collect_votes(env)
        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
