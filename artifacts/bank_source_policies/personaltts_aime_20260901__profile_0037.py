from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from policies.base import ControllerPolicy, valid_answer


def _deterministic_plurality(answers: Sequence[str]) -> Optional[str]:
    """Return the most frequent answer; break ties by first appearance."""
    if not answers:
        return None
    counts = Counter(answers)
    top = max(counts.values())
    return next(a for a in answers if counts[a] == top)


class CandidatePolicy(ControllerPolicy):
    """Three-branch parallel plurality-voting policy.

    Spawns 3 independent Qwen3-4B branches in parallel, advances each
    until it produces a first valid answer (early stopping), then
    selects the final answer by deterministic plurality vote.

    Budget analysis for profile_0037
    --------------------------------
    * accuracy  >= 0.2403  (mean)       — voting improves reliability
    * latency   <= 3347.7  (batch)        — parallel branches ≈ single-branch latency
    * w_compute <= 108979  (mean)       — 3 × 28 163 ≈ 84 490, margin ≈ 24 500

    The baseline achieves 78 % accuracy with ≈ 11.6 sequential branches,
    proving that plurality voting over independent Qwen3-4B branches
    dramatically boosts accuracy even though single-branch accuracy
    (31.7 %) is below 50 %.  Three parallel branches is the largest
    width that safely fits the weighted-compute budget.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(models: List[str]) -> str:
        """Select Qwen3-4B if available, otherwise the first model."""
        preferred = "Qwen3-4B"
        if preferred in models:
            return preferred
        return models[0]

    @staticmethod
    def _desired_width(requirements: Optional[Dict[str, Any]]) -> int:
        """Choose branch width that fits the weighted-compute budget.

        With Qwen3-4B early-stopping, average per-branch weighted-compute
        is ≈ 28 163.  We allow a 20 % safety margin on the budget.
        """
        if requirements is None:
            return 3
        compute_limit = float(requirements["weighted_compute"]["value"])
        est_per_branch = 28200.0  # empirical average for 1 branch
        safe_limit = compute_limit * 0.80  # 20 % safety margin
        width = max(1, int(safe_limit / est_per_branch))
        # Cap at 5 — diminishing returns and latency growth
        return min(width, 5)

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env.models)
        width = self._desired_width(env.requirements)

        # ---- spawn ----
        opened: Optional[List[Dict[str, Any]]] = None
        while width >= 1:
            try:
                opened = env.spawn_many(model, width)
                break
            except Exception:
                width -= 1

        # absolute fallback: single sequential branch
        if opened is None or len(opened) == 0:
            return self._single_branch_fallback(env, model)

        my_bids = set()
        answers: Dict[int, str] = {}  # branch_id -> first valid answer

        for b in opened:
            bid = b["branch_id"]
            my_bids.add(bid)
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                answers[bid] = ans

        # ---- continue until every branch has a valid answer or is done ----
        max_rounds = 25
        for _ in range(max_rounds):
            if len(answers) >= len(my_bids):
                break

            state = env.observe()
            to_continue: List[int] = []
            for b in state["branches"]:
                bid = b["branch_id"]
                if (
                    bid in my_bids
                    and bid not in answers
                    and b["can_continue"]
                    and not b["pruned"]
                ):
                    to_continue.append(bid)

            if not to_continue:
                break

            try:
                updated = env.continue_many(to_continue)
            except Exception:
                # budget or pool exhausted
                break

            for b in updated:
                bid = b["branch_id"]
                if bid not in answers:
                    ans = valid_answer(b.get("latest_answer"))
                    if ans is not None:
                        answers[bid] = ans

        # ---- pick up any remaining valid answers from terminal branches ----
        if len(answers) < len(my_bids):
            state = env.observe()
            for b in state["branches"]:
                bid = b["branch_id"]
                if bid in my_bids and bid not in answers:
                    ans = valid_answer(b.get("latest_answer"))
                    if ans is not None:
                        answers[bid] = ans

        # ---- vote ----
        votes = list(answers.values())
        if votes:
            return env.finish(_deterministic_plurality(votes))
        return env.finish(None)

    # ------------------------------------------------------------------
    # fallback
    # ------------------------------------------------------------------

    def _single_branch_fallback(self, env, model: str) -> Optional[str]:
        """Identical to the round-0 incumbent: one branch, first answer."""
        branch = env.spawn(model)
        ans = valid_answer(branch.get("latest_answer"))
        rounds = 0
        while ans is None and branch["can_continue"] and rounds < 25:
            branch = env.continue_branch(branch["branch_id"])
            ans = valid_answer(branch.get("latest_answer"))
            rounds += 1
        return env.finish(ans)
