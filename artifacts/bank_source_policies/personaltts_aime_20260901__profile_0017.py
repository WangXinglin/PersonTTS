from __future__ import annotations

from typing import Any, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Single-branch deep-thinking policy for budget-constrained profiles.

    Round 0 used the cheapest model (Qwen3-0.6B) which scored <1 %
    accuracy despite needing only ~11.2 %.  The baseline achieved 78 %
    accuracy but overran the weighted-compute budget by 65x.

    This revision picks the largest model whose per-step weighted
    compute still allows at least 3 continues within the mean WC
    budget, then runs a single branch without early-stopping so the
    model can refine its answer across multiple checkpoints.

    For profile 0017 (WC budget ~23 426):
      - Qwen3-4B is selected (multiplier 6.667).
      - 5 continues => 6 generation steps * ~550 tokens * 6.667 ~ 22 000 WC.
      - Comfortably within budget even if every problem reaches the cap.
      - Expected accuracy ~40-60 %, far above the 11.2 % threshold.
    """

    NAME = "candidate"

    MULTIPLIERS = {
        "Qwen3-0.6B": 1.0,
        "Qwen3-1.7B": 2.833,
        "Qwen3-4B": 6.667,
        "Qwen3-8B": 13.333,
        "Qwen3-14B": 23.333,
        "Qwen3-32B": 53.333,
    }

    TOKENS_PER_STEP = 550

    MIN_CONTINUES = 3

    MAX_CONTINUES_CAP = 12

    def select_model_and_depth(self, models, wc_budget):
        """Return (model_name, max_continues) fitting the WC budget."""
        available = set(models)
        for name in (
            "Qwen3-32B",
            "Qwen3-14B",
            "Qwen3-8B",
            "Qwen3-4B",
            "Qwen3-1.7B",
            "Qwen3-0.6B",
        ):
            if name not in available:
                continue
            mult = self.MULTIPLIERS.get(name, 10.0)
            wc_per_step = self.TOKENS_PER_STEP * mult
            affordable_steps = wc_budget / wc_per_step
            max_cont = int(affordable_steps) - 1
            if max_cont >= self.MIN_CONTINUES:
                return name, min(max_cont, self.MAX_CONTINUES_CAP)

        for name in ("Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"):
            if name in available:
                return name, self.MAX_CONTINUES_CAP
        return list(models)[0], 5

    def solve(self, env) -> Optional[str]:
        req = env.requirements
        if req is not None:
            wc_budget = float(req["weighted_compute"]["value"])
        else:
            wc_budget = 23426.0

        model, max_continues = self.select_model_and_depth(env.models, wc_budget)

        branch = env.spawn(model)

        for _ in range(max_continues):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(branch["branch_id"])

        return env.finish(valid_answer(branch.get("latest_answer")))
