from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-probe policy using spawn_many/continue_many for low latency.

    The baseline ASC runs branches sequentially (latency = sum of all ops),
    causing batch latency far above the 1482s threshold.  This policy spawns
    a small cohort in parallel and advances them with continue_many so that
    latency = #rounds × max(cohort-per-round).  With width=3 and a
    checkpoint-depth cap of 30, worst-case phases ≈ 31 × max-of-3, keeping
    even p99 latency well under the constraint while majority voting
    maintains accuracy above the 0.352 requirement.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _pick_model(self, env) -> str:
        """Select model, preferring the smallest available for lowest latency."""
        for m in env.models:
            if "4B" in m:
                return m
        return env.models[0]

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # --- tunables --------------------------------------------------
        width = 3           # parallel branches (voting quorum)
        max_rounds = 30     # checkpoint-depth cap to bound latency
        min_rounds = 1      # rounds before early-stop eligibility
        streak_needed = 2   # consecutive majority rounds to stop
        # ---------------------------------------------------------------

        # Phase 1: spawn all branches in one parallel phase
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            # Pool or budget too small – fall back to a single branch
            branch = env.spawn(model)
            while branch["can_continue"]:
                branch = env.continue_branch(int(branch["branch_id"]))
            return env.finish(valid_answer(branch.get("latest_answer")))

        # Initialise per-branch tracking
        bids = [int(b["branch_id"]) for b in opened]
        bstate = {int(b["branch_id"]): b for b in opened}
        answers: Dict[int, str] = {}

        for b in opened:
            bid = int(b["branch_id"])
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                answers[bid] = ans

        prev_maj: Optional[str] = None
        streak = 0

        # Phase 2: parallel checkpoint rounds
        for rnd in range(max_rounds):
            active = [
                bid for bid in bids
                if not bstate[bid]["pruned"] and bstate[bid]["can_continue"]
            ]
            if not active:
                break

            try:
                updated = env.continue_many(active)
            except Exception:
                break  # budget exhausted

            for b in updated:
                bid = int(b["branch_id"])
                bstate[bid] = b
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    answers[bid] = ans

            # Early stopping on stable majority
            if rnd + 1 >= min_rounds and len(answers) >= 2:
                votes = [answers[bid] for bid in bids if bid in answers]
                counts = Counter(votes)
                best_answer, best_count = counts.most_common(1)[0]

                if best_count >= 2:  # ≥ 2 of 3 agree
                    if best_answer == prev_maj:
                        streak += 1
                    else:
                        prev_maj = best_answer
                        streak = 1

                    if streak >= streak_needed:
                        return env.finish(best_answer)
                else:
                    prev_maj = None
                    streak = 0

        # Phase 3: final plurality vote
        votes = [answers[bid] for bid in bids if bid in answers]
        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish()
