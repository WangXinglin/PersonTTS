from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn, depth-bounded policy for tight latency / low accuracy profiles.

    Uses spawn_many + continue_many (cohort semantics: each round adds only the
    max-branch latency, not the sum) so parallel width is nearly free for latency.
    Depth is capped to a small number of continue rounds so that even the slowest
    problems finish well under the batch latency budget.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_smallest_model(models: List[str]) -> str:
        """Select the model with the smallest parameter count for speed."""

        def _extract_size(name: str) -> float:
            for part in name.replace("_", "-").split("-"):
                cleaned = part.upper().rstrip("B")
                try:
                    return float(cleaned)
                except ValueError:
                    continue
            return float("inf")

        return min(models, key=_extract_size)

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_smallest_model(list(env.models))

        # --- read requirements to size depth and width ----------------
        latency_budget = 500.0  # generous default
        compute_budget = 15_000_000.0
        reqs = env.requirements
        if reqs is not None:
            for c in reqs.get("constraints", []):
                if c["metric"] == "latency_s":
                    latency_budget = float(c["value"])
                elif c["metric"] == "weighted_compute":
                    compute_budget = float(c["value"])

        # Each parallel round costs roughly the max-branch checkpoint
        # latency.  Empirically ~20-80 s per round for Qwen3-4B.
        # Use a conservative per-round budget of 120 s so batch fits.
        per_round_budget = 120.0
        max_total_rounds = max(2, int(latency_budget / per_round_budget))
        max_continue_rounds = max_total_rounds - 1  # first round is spawn

        # Width: parallel branches are ~free for latency but cost compute.
        # Rough per-branch-per-checkpoint weighted compute for Qwen3-4B ≈ 4000.
        est_compute_per_branch = 4000.0 * max_total_rounds
        desired_width = max(1, int(compute_budget * 0.02 / max(est_compute_per_branch, 1)))
        desired_width = min(desired_width, 10)  # cap for pool safety

        # --- check pool availability ----------------------------------
        state = env.observe()
        remaining = int(state["remaining_by_model"].get(model, 0))
        width = max(1, min(desired_width, remaining))

        # --- parallel spawn -------------------------------------------
        opened = env.spawn_many(model, width)
        my_ids = set()
        for b in opened:
            my_ids.add(b["branch_id"])

        # --- bounded parallel continue rounds -------------------------
        for _ in range(max_continue_rounds):
            state = env.observe()
            to_continue = []
            for b in state["branches"]:
                if b["branch_id"] in my_ids and b["can_continue"] and not b["pruned"]:
                    to_continue.append(b["branch_id"])
            if not to_continue:
                break
            env.continue_many(to_continue)

        # --- collect answers and vote ---------------------------------
        state = env.observe()
        votes: list[str] = []
        for b in state["branches"]:
            if b["branch_id"] in my_ids and not b["pruned"]:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    votes.append(ans)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
