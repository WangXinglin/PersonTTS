from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn, full-completion, majority-vote policy.

    The round-0 candidate failed because it stopped on unreliable
    checkpoint-0 answers (partial reasoning before any continuation),
    achieving only 52% accuracy with 8 branches.  The baseline
    (sequential ASC) achieves 78% accuracy but takes ~5400 s avg
    latency because every branch-step is additive.

    This policy fixes both problems:
      1. Uses spawn_many / continue_many so each phase adds only the
         MAX cohort latency (parallel execution), keeping total
         latency well within the 1884 s batch budget.
      2. Runs every branch to full completion (stage_terminal) before
         voting, so only reliable final-reasoning answers are used.
      3. Spawns 32 parallel branches for robust majority voting,
         easily affordable within the 13.1 M weighted-compute budget.

    Expected metrics (Qwen3-4B, 32 branches):
      - Weighted compute: ~4 M   (budget: 13.1 M)
      - Latency avg:      ~600 s (budget batch: 1884 s)
      - Accuracy:         ~80 %  (threshold: 61.8 %)
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pick_model(self, env) -> str:
        """Select Qwen3-4B (well-characterised cost profile from traces).

        Fall back to the first available model if Qwen3-4B is absent.
        """
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return env.models[0]

    def _target_width(self, env) -> int:
        """Choose branch width that fits the weighted-compute budget.

        Conservative per-branch cost estimate for Qwen3-4B with full
        completion: ~150 K weighted compute (based on baseline trace
        data: 1.53 M avg compute / 11.6 avg branches ≈ 132 K, rounded
        up for safety).  We use at most 80 % of the budget for the
        initial stage, reserving headroom.
        """
        req = env.requirements
        if req is None:
            return 16
        compute_budget = float(req["weighted_compute"]["value"])
        per_branch_cost = 200000  # conservative
        max_by_budget = max(4, int(compute_budget * 0.8 / per_branch_cost))
        # Cap at 32: diminishing accuracy returns and gentle latency growth
        return min(32, max_by_budget)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width = self._target_width(env)

        # ---- pool check ------------------------------------------------
        state = env.observe()
        remaining = int(state.get("remaining_by_model", {}).get(model, 0))
        actual_width = min(width, remaining)
        if actual_width < 1:
            return env.finish()

        # ---- phase 1: parallel spawn -----------------------------------
        try:
            env.spawn_many(model, actual_width)
        except Exception:
            return env.finish()

        # ---- phase 2: continue all branches to completion --------------
        # Each continue_many round advances every active branch by one
        # checkpoint.  Latency per round = max across the cohort.
        # Qwen3-4B branches typically terminate within 50 rounds; the
        # generous cap of 500 handles outliers safely.
        for _ in range(500):
            state = env.observe()
            branches = state["branches"]
            active = [
                b for b in branches
                if b["can_continue"] and not b["pruned"]
            ]
            if not active:
                break
            try:
                env.continue_many([b["branch_id"] for b in active])
            except Exception:
                break  # budget exhausted; vote with what we have

        # ---- phase 3: majority vote on terminal answers ----------------
        state = env.observe()
        branches = state["branches"]

        terminal_votes: List[str] = []
        all_votes: List[str] = []
        for b in branches:
            if b["pruned"]:
                continue
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                all_votes.append(ans)
                if b["stage_terminal"]:
                    terminal_votes.append(ans)

        # Prefer terminal answers (complete reasoning); fall back to all.
        votes = terminal_votes if terminal_votes else all_votes
        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish()
