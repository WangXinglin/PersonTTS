from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Width-5 Qwen3-4B parallel majority vote with unbeatable-majority pruning.

    Strategy: spawn 5 Qwen3-4B branches in parallel and run them all to
    natural termination via synchronous checkpoint rounds.  After each round,
    check if enough naturally-terminated branches agree to form an unbeatable
    majority (3+ of 5).  If so, prune remaining live branches to save compute.
    Otherwise, continue until all branches finish and take a plurality vote.

    Key insights from prior rounds:
      - Round 0 (incumbent, utility=0.491): width-3 with early-stop got 62.4%
        accuracy (barely below the 62.9% threshold), compute 207K.
      - Round 1 (rejected): width-5 with aggressive early exit got only 51.3%
        accuracy because branches were stopped before thinking deeply.
      - Round 2 (rejected): width-7 at full depth achieved 76% accuracy but
        672K mean compute, exceeding the 548K budget.

    Width-5 at full depth is the sweet spot:
      - weighted_compute: 5/7 of round-2 cost = ~480K (budget 548K, 12% margin)
      - With majority pruning, saves additional 10-20% on easy problems
      - accuracy: ~70-74% majority-vote (threshold 62.9%)
      - latency: cohort-max semantics, well under 2100s limit
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _select_model(self, env) -> str:
        """Pick Qwen3-4B for its known cost-accuracy profile."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return sorted(env.models)[0]

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)
        width = 5
        majority_needed = (width // 2) + 1  # 3 for width=5

        # --- Phase 1: parallel spawn ---------------------------------- #
        opened = env.spawn_many(model, width)
        bid_set = set(b["branch_id"] for b in opened)

        # --- Phase 2: parallel checkpoint rounds with majority pruning - #
        for _ in range(500):  # generous safety cap
            state = env.observe()
            live_ids: List[int] = []
            terminated_answers: List[str] = []

            for b in state["branches"]:
                bid = b["branch_id"]
                if bid not in bid_set or b["pruned"]:
                    continue
                if b["can_continue"]:
                    live_ids.append(bid)
                else:
                    # Branch has naturally terminated
                    ans = valid_answer(b.get("latest_answer"))
                    if ans is not None:
                        terminated_answers.append(ans)

            # Check for unbeatable majority among terminated branches.
            # If the top answer among finished branches cannot be overtaken
            # even if all remaining live branches vote for the runner-up,
            # the result is decided: prune remaining branches to save compute.
            if len(terminated_answers) >= majority_needed:
                counts = Counter(terminated_answers)
                top_answer, top_count = counts.most_common(1)[0]
                second_max = max(
                    (c for a, c in counts.items() if a != top_answer),
                    default=0,
                )
                if top_count > len(live_ids) + second_max:
                    for bid in live_ids:
                        env.prune(bid)
                    return env.finish(top_answer)

            if not live_ids:
                break

            env.continue_many(live_ids)

        # --- Phase 3: plurality vote on all terminated answers -------- #
        votes: List[str] = []
        for b in env.observe()["branches"]:
            if b["branch_id"] in bid_set and not b["pruned"]:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    votes.append(ans)

        final = deterministic_plurality(votes) if votes else None
        return env.finish(final)
