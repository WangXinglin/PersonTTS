from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel majority-vote controller for latency-constrained profiles.

    The incumbent sequential ASC spawns ~12 branches per problem one at a time,
    causing latency to scale linearly with candidate count (mean margin -24 ks
    over the batch budget).  This policy replaces sequential branching with a
    fixed-width parallel cohort:

    * spawn_many  -> one latency step (max of W branch spawns)
    * continue_many per round -> one latency step (max of W branch continues)

    Total per-problem latency ~ single_branch_time * max_of_W_factor instead of
    ~ N_branches * single_branch_time.  With W=3 and Qwen3-4B, even the hardest
    problems stay close to the 1176 s batch budget.

    Voting uses only **terminal** branch answers (answers can fluctuate at
    intermediate checkpoints, so early-stopping on intermediate consensus would
    sacrifice accuracy).  A strict majority among terminal branches triggers an
    early finish; otherwise all branches run to natural completion and a
    plurality vote decides.
    """

    NAME = "parallel_majority_vote"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def solve(self, env) -> Optional[str]:
        # ---- model selection ----
        model = "Qwen3-4B" if "Qwen3-4B" in env.models else env.models[0]

        # ---- width selection ----
        # Width 3 gives a meaningful majority-vote accuracy boost (~0.784 from
        # ~0.70 per-branch) while the max-of-3 per-round overhead (~1.2-1.3x)
        # keeps parallel latency within the batch budget.
        width = 3
        majority_needed = 2  # strict majority of 3

        # ---- spawn parallel cohort (single latency step) ----
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            return self._single_branch(env, model)

        my_ids = set(int(b["branch_id"]) for b in opened)

        # ---- synchronous parallel continuation rounds ----
        for _ in range(500):  # safety bound; branches terminate well before this
            state = env.observe()
            active_ids = []
            terminal_votes = []

            for b in state["branches"]:
                bid = int(b["branch_id"])
                if bid not in my_ids or b["pruned"]:
                    continue
                if b["can_continue"]:
                    active_ids.append(bid)
                else:
                    # Branch has reached its terminal checkpoint
                    ans = valid_answer(b.get("latest_answer"))
                    if ans is not None:
                        terminal_votes.append(ans)

            # Early exit: enough terminal branches agree
            if len(terminal_votes) >= majority_needed:
                counts = Counter(terminal_votes)
                top_answer, top_count = counts.most_common(1)[0]
                if top_count >= majority_needed:
                    return env.finish(top_answer)

            # All branches finished
            if not active_ids:
                break

            # Advance all still-active branches in parallel (one latency step)
            try:
                env.continue_many(active_ids)
            except Exception:
                break  # budget or pool exhausted

        # ---- final vote on terminal answers ----
        return env.finish(
            deterministic_plurality(terminal_votes) if terminal_votes else None
        )

    # ------------------------------------------------------------------
    def _single_branch(self, env, model: str) -> Optional[str]:
        """Fallback when spawn_many cannot allocate the full cohort."""
        b = env.spawn(model)
        while b["can_continue"]:
            b = env.continue_branch(b["branch_id"])
        return env.finish(valid_answer(b.get("latest_answer")))
