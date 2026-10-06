from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Wide parallel-spawn with deep continuation and plurality voting.

    Previous rounds achieved only 6-9 % accuracy with 3-5 branches at
    checkpoint 2, far below the 24.9 % threshold.  Latency (20 s) and
    compute (108 K) are at less than 3 % of their budgets.

    This policy exploits the massive headroom by spawning up to 50
    branches of Qwen3-14B and continuing 4 parallel rounds (reaching
    checkpoint 4).  Plurality voting over many independent branches
    amplifies weak per-branch accuracy: correct answers cluster on one
    value while wrong answers scatter across many distinct values.

    If the Qwen3-14B pool limits width to fewer than 10 branches, a
    supplementary cohort of another model is added (costing one spawn
    phase and therefore one fewer continue round).

    Latency: 5 phases x ~7 s = ~35 s  (batch well under 40.5 s)
    Compute: 50 x 5 x 12 K = 3 M     (under 6 M mean budget)
    """

    NAME = "candidate"

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _select_primary(env) -> str:
        """Pick the largest available model for best per-checkpoint accuracy."""
        for name in ("Qwen3-14B", "Qwen3-8B", "Qwen3-4B"):
            if name in env.models:
                return name
        return env.models[0]

    @staticmethod
    def _max_affordable(env, model: str, cps: int, cap: int) -> int:
        """Binary-search the largest width that fits remaining budget."""
        lo, hi, best = 1, cap, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                ok = env.can_afford_complete_candidates(
                    model, count=mid,
                    checkpoints_per_stage=cps, max_refine_rounds=0)
            except Exception:
                return cap  # API unavailable; optimistically return cap
            if ok:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    @staticmethod
    def _try_spawn(env, model: str, target: int) -> List[Dict[str, Any]]:
        """Spawn up to *target* branches with progressive width fallback."""
        widths = sorted(
            {w for w in [target, target * 3 // 4, target // 2,
                         target // 4, 10, 5, 3, 1]
             if 0 < w <= target},
            reverse=True,
        )
        for width in widths:
            try:
                if width > 1:
                    return list(env.spawn_many(model, width))
                return [env.spawn(model)]
            except Exception:
                continue
        return []

    # ----------------------------------------------------------------
    # main solve
    # ----------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        primary = self._select_primary(env)

        # Read hard-requirement thresholds
        reqs = env.requirements
        lat_limit = float(reqs["latency_s"]["value"]) if reqs else 40.0

        # Phase budget: ~7.5 s per phase, 2 s safety margin
        total_phases = max(2, int((lat_limit - 2.0) / 7.5))
        cps = total_phases  # checkpoints per stage (spawn cp0 + continues)

        # -- Spawn primary model -----------------------------------------
        primary_cap = self._max_affordable(env, primary, cps, 50)
        branches = self._try_spawn(env, primary, max(primary_cap, 1))

        # If primary model spawn failed, try every other model
        if not branches:
            for alt in env.models:
                if alt == primary:
                    continue
                branches = self._try_spawn(env, alt, 10)
                if branches:
                    break
        if not branches:
            return env.finish()

        spawn_phases = 1

        # -- Supplement with a secondary model if width is small ---------
        if len(branches) < 10:
            supp_cps = max(1, cps - 1)
            for other in env.models:
                if other == primary:
                    continue
                room = 50 - len(branches)
                if room < 1:
                    break
                supp_cap = self._max_affordable(
                    env, other, supp_cps, room)
                if supp_cap > 0:
                    supp = self._try_spawn(env, other, supp_cap)
                    if supp:
                        branches.extend(supp)
                        spawn_phases = 2
                    break

        # -- Continue rounds ---------------------------------------------
        max_continues = total_phases - spawn_phases
        for _ in range(max_continues):
            state = env.observe()
            ids = [b["branch_id"] for b in state["branches"]
                   if b["can_continue"] and not b["pruned"]]
            if not ids:
                break
            try:
                env.continue_many(ids)
            except Exception:
                break

        # -- Plurality vote ----------------------------------------------
        state = env.observe()
        votes: List[str] = []
        for b in state["branches"]:
            if b["pruned"]:
                continue
            # Use latest answer if valid
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                votes.append(ans)
                continue
            # Fallback: most recent valid answer from history
            history = b.get("answer_history")
            if history:
                for raw in reversed(history):
                    ans = valid_answer(raw)
                    if ans is not None:
                        votes.append(ans)
                        break

        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish()
