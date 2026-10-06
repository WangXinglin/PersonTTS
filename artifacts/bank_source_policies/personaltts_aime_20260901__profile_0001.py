from __future__ import annotations

from typing import Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Wide parallel majority-voting with terminal-weighted voting and one
    extra continue round versus the round-3 incumbent (utility 0.346).

    Root-cause diagnosis (round 3, 25 branches, 37 rounds, no early stop):
      * Overall accuracy 76.875% vs 77.02% threshold (gap = 0.145%).
      * Latency batch margin ~21 s; compute at 11% of budget.
      * AIME_25 accuracy 73.9% is the bottleneck; many branches still
        non-terminal at round 37, producing noisy intermediate answers that
        can outvote the more-reliable terminal answers in simple plurality.

    Improvements:
      1) max_continue_rounds 37 -> 38.  One extra round allows more branches
         to reach terminal state on medium-hard problems.  Latency impact:
         39 phases * 12.8 s/phase = 499 s at p99, well within the 500 s batch
         budget (estimated batch ~491 s, margin ~9 s).
      2) Terminal-weighted voting: terminal answers receive 2x weight, non-
         terminal answers 1x.  Terminal branches completed their full
         reasoning chain and are empirically more accurate.  This corrects
         cases where noisy intermediate answers from still-running branches
         outvote the correct terminal consensus.

    Budget analysis for profile_0001:
      accuracy  >= 0.7702 (mean)
        -- Same 25-branch width as round 3 (proven width)
        -- Extra depth + terminal weighting targets ~0.2-0.3% accuracy gain
        -- Expected overall ~77.1% (above threshold)
      latency   <= 499.89 s (batch)
        -- Hardest episodes: 39 phases * ~12.8 s = ~499 s (p99)
        -- Estimated batch ~491 s (9 s margin)
      compute   <= 19.4 M (mean)
        -- ~2.1 M per episode (11% of budget, unchanged)
    """

    NAME = "candidate"

    def _weighted_plurality(self, branches):
        """Deterministic weighted plurality: terminal answers count double."""
        counts = {}
        first_bid = {}
        for b in branches:
            if b["pruned"]:
                continue
            ans = valid_answer(b.get("latest_answer"))
            if ans is None:
                continue
            weight = 2 if b["stage_terminal"] else 1
            counts[ans] = counts.get(ans, 0) + weight
            if ans not in first_bid:
                first_bid[ans] = int(b["branch_id"])
        if not counts:
            return None
        top = max(counts.values())
        winners = [a for a, c in counts.items() if c == top]
        if len(winners) == 1:
            return winners[0]
        # Deterministic tie-break: answer that appeared on the lowest branch_id
        return min(winners, key=lambda a: first_bid[a])

    def solve(self, env) -> Optional[str]:
        # ---- model selection ------------------------------------------------
        model = "Qwen3-4B"
        if model not in env.models:
            model = sorted(env.models)[0]

        # ---- hyper-parameters for profile_0001 ------------------------------
        target_width = 25         # same proven width as round 3
        max_continue_rounds = 38  # +1 vs round 3 for deeper exploration

        # ---- phase 1: parallel spawn ----------------------------------------
        state = env.observe()
        pool = int(state["remaining_by_model"].get(model, 0))
        width = min(target_width, pool)
        if width < 1:
            return env.finish(None)

        spawned = False
        try:
            env.spawn_many(model, width)
            spawned = True
        except Exception:
            pass

        if not spawned:
            try:
                env.spawn(model)
                spawned = True
            except Exception:
                pass

        if not spawned:
            return env.finish(None)

        # ---- phase 2: run ALL branches toward natural termination -----------
        #   No early stopping: avoids premature consensus on wrong answer.
        #   Branches that reach terminal state naturally drop out of
        #   continue_many, reducing per-phase latency over time.
        for _ in range(max_continue_rounds):
            state = env.observe()
            active_ids = [
                b["branch_id"]
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not active_ids:
                break  # all branches reached terminal state naturally

            try:
                env.continue_many(active_ids)
            except Exception:
                break

        # ---- phase 3: terminal-weighted plurality vote ----------------------
        #   Terminal answers (completed reasoning) get 2x weight;
        #   non-terminal intermediate answers get 1x weight.
        state = env.observe()
        answer = self._weighted_plurality(state["branches"])
        return env.finish(answer)
