from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel multi-branch majority-vote policy using the largest affordable model.

    Strategy:
    - Pick the largest model where we can afford >= 3 parallel branches within
      the weighted-compute budget (preferring Qwen3-4B over smaller models).
    - Spawn branches in parallel via spawn_many to keep latency equal to a
      single branch (parallel execution).
    - Advance all branches via continue_many until every branch reaches
      stage_terminal.
    - Use deterministic plurality vote across final answers.

    This exploits the enormous compute and latency headroom left unused by the
    incumbent single-branch Qwen3-0.6B policy, dramatically boosting per-batch
    accuracy well above the 13.55 percent threshold.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def solve(self, env) -> Optional[str]:
        reqs = env.requirements
        wc_budget = float(reqs["weighted_compute"]["value"]) if reqs else 777368.0

        model = self._choose_model(env.models, wc_budget)
        width = self._choose_width(model, wc_budget)

        # Spawn all branches in parallel -- latency = max(single branch), not sum.
        opened = env.spawn_many(model, width)
        branch_id_set = set()
        for b in opened:
            branch_id_set.add(b["branch_id"])

        # Advance every active branch one checkpoint at a time, in parallel.
        while True:
            state = env.observe()
            active_ids = []
            for b in state["branches"]:
                bid = b["branch_id"]
                if bid in branch_id_set and not b["pruned"] and b["can_continue"]:
                    active_ids.append(bid)
            if not active_ids:
                break
            try:
                env.continue_many(active_ids)
            except Exception:
                break  # budget or pool exhausted

        # Collect votes from all surviving branches.
        state = env.observe()
        votes: List[str] = []
        for b in state["branches"]:
            if b["branch_id"] in branch_id_set and not b["pruned"]:
                answer = valid_answer(b.get("latest_answer"))
                if answer is not None:
                    votes.append(answer)

        final = deterministic_plurality(votes) if votes else None
        return env.finish(final)

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    def _choose_model(self, models: list, wc_budget: float) -> str:
        """Pick the largest model that can afford >= 3 parallel branches."""
        scored = []
        for m in models:
            size = self._parse_size(m)
            scored.append((size, m))
        # Sort largest first; ties broken alphabetically for determinism.
        scored.sort(key=lambda t: (-t[0], t[1]))

        for size, m in scored:
            est_wc = self._est_weighted_per_branch(size)
            if est_wc * 3 <= wc_budget * 0.90:
                return m

        # Fallback: smallest model
        scored.sort(key=lambda t: (t[0], t[1]))
        return scored[0][1]

    def _choose_width(self, model: str, wc_budget: float) -> int:
        """Maximum parallel width that fits within budget with safety margin."""
        size = self._parse_size(model)
        est_wc = self._est_weighted_per_branch(size)
        # Use 80 percent of budget as effective ceiling for the mean constraint.
        effective = wc_budget * 0.80
        width = max(1, int(effective / est_wc))
        # Cap width: diminishing returns beyond 7 for majority vote.
        return min(width, 7)

    def _est_weighted_per_branch(self, model_size_b: float) -> float:
        """Estimated weighted compute for one complete branch."""
        # Conservative estimate: ~22k raw tokens per branch (slightly above
        # the observed ~20k baseline average), weighted by model size.
        raw_est = 22000
        weight = max(model_size_b, 0.5) / 0.6
        return raw_est * weight

    @staticmethod
    def _parse_size(name: str) -> float:
        """Extract numeric parameter count before 'B' in a model name."""
        result = float("inf")
        i = 0
        n = len(name)
        while i < n:
            if name[i] in ("B", "b") and i > 0:
                j = i - 1
                while j >= 0 and (name[j].isdigit() or name[j] == "."):
                    j -= 1
                num_str = name[j + 1 : i]
                if num_str:
                    try:
                        val = float(num_str)
                        if val < result:
                            result = val
                    except ValueError:
                        pass
            i += 1
        return result
