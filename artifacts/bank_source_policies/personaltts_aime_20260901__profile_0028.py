from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel majority-vote with calibrated depth and conservative early stopping.

    Profile constraints (profile_0028):
      accuracy  >= 0.4438  (mean)
      latency_s <= 315.86  (batch)
      weighted_compute <= 17.4 M  (mean)

    Diagnosis from prior rounds:
      Round 1: Qwen3-32B width=7, max_cp=22, warmup=3, agree=5
        => accuracy 58.4% (great), latency p90=338s (too high, ~15.3s/phase)
      Round 2: Qwen3-32B width=7, max_cp=17, warmup=1, agree=4
        => accuracy 39.4% (too low), latency p90=239s (lots of margin)

    Root cause in round 2: warmup=1 triggers premature consensus on wrong
    answers — many episodes exit at round 2-3 before branches have developed
    enough to reliably identify the correct solution.

    Fix for round 3:
      1. Increase warmup to 6 — no consensus check until round 6, giving
         branches ample time to develop their reasoning.
      2. Keep agree_needed at 5/7 (strong 71% supermajority) to avoid
         false consensus even after warmup.
      3. Set max_cp=19 to cap tail latency:
         20 phases × 15.4s/phase ≈ 308s < 316s budget.
         (Round 1 had 23 phases × 15.3s ≈ 352s, which exceeded budget.)

    Expected outcome:
      accuracy  ~ 55-60% (similar to round 1, well above 0.4438)
      latency_s ~ 308s at batch (under 315.86)
      compute   ~ 2.2M (far under 17.4M)
    """

    NAME = "candidate"

    MODEL_PREFERENCE = [
        "Qwen3-32B", "Qwen3-14B", "Qwen3-8B", "Qwen3-4B",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _select_model(self, env) -> str:
        """Pick the largest available model for best per-branch accuracy."""
        available = set(env.models)
        for m in self.MODEL_PREFERENCE:
            if m in available:
                return m
        return env.models[0]

    def _collect_answers(self, env, branch_set):
        """Gather valid answers from all unpruned branches in the set."""
        answers = []
        for b in env.observe()["branches"]:
            if int(b["branch_id"]) in branch_set and not b["pruned"]:
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    answers.append(a)
        return answers

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # --- Parameters calibrated from round 1/2 trace analysis ----------
        width = 7           # parallel branches for robust voting
        max_cp = 19         # max continue_many rounds (cap tail latency)
        warmup = 6          # no consensus check before round 6
        agree_needed = 5    # 5/7 = 71% supermajority for early stop

        # Adjust width to what the pool can supply
        state = env.observe()
        pool = state["remaining_by_model"].get(model, 0)
        width = min(width, pool)
        if width < 1:
            return env.finish()

        # Scale consensus threshold to actual width
        if width <= 3:
            agree_needed = width
        elif width <= 5:
            agree_needed = max(3, width - 1)
        else:
            agree_needed = max(3, (width * 5 + 6) // 7)  # ~71% threshold

        # --- Parallel spawn (one cohort phase) ----------------------------
        opened = env.spawn_many(model, width)
        branch_ids = [int(b["branch_id"]) for b in opened]
        branch_set = set(branch_ids)

        # --- Parallel continue rounds with conservative early stopping ----
        for rnd in range(max_cp):
            state = env.observe()
            by_id = {int(b["branch_id"]): b for b in state["branches"]}

            # Advance every branch that still has checkpoints remaining
            active = [
                bid for bid in branch_ids
                if not by_id[bid]["pruned"] and by_id[bid]["can_continue"]
            ]
            if not active:
                break  # all branches reached stage-terminal

            try:
                env.continue_many(active)
            except Exception:
                break  # budget or pool exhausted

            # After warmup rounds, check for strong supermajority consensus
            if rnd >= warmup:
                answers = self._collect_answers(env, branch_set)
                if len(answers) >= agree_needed:
                    counts = Counter(answers)
                    top_answer, top_count = counts.most_common(1)[0]
                    if top_count >= agree_needed:
                        return env.finish(top_answer)

        # --- Final deterministic plurality vote ---------------------------
        answers = self._collect_answers(env, branch_set)
        if answers:
            return env.finish(deterministic_plurality(answers))
        return env.finish()
