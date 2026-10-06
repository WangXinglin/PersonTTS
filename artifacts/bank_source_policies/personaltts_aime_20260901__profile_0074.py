from __future__ import annotations

from typing import List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Budget-aware single-branch policy that upgrades model quality.

    Profile 0074 requires only ~1.13% mean accuracy but has generous
    weighted-compute (mean <= 56 030) and latency (batch <= 1659 s) headroom.
    The incumbent uses the cheapest model (Qwen3-0.6B) and achieves only
    0.935% accuracy -- failing the accuracy threshold in 55% of batches.

    This policy:
      1. Selects the largest model whose per-checkpoint cost still allows
         at least 5 checkpoints within 80% of the compute budget.
      2. Spawns a single branch and continues until stage_terminal or the
         checkpoint cap, whichever comes first.
      3. Returns the latest valid answer.

    Even modest accuracy (~10-40%) from a stronger model makes batch
    satisfaction near-certain (P(>=1 correct in 60) > 99.99%), giving
    utility close to 1.0 versus the incumbent's 0.449.
    """

    NAME = "candidate"

    # -----------------------------------------------------------------
    # Deterministic helpers
    # -----------------------------------------------------------------

    @staticmethod
    def extract_size(name: str) -> float:
        """Extract numeric model-size hint (billions) from a name like 'Qwen3-4B'."""
        parts = name.replace("_", "-").split("-")
        for part in parts:
            upper = part.upper()
            if upper.endswith("B"):
                try:
                    return float(upper[:-1])
                except ValueError:
                    continue
        return float("inf")

    def select_model_and_depth(
        self, models: List[str], compute_budget: float
    ) -> tuple:
        """Return (model_name, max_checkpoints) maximising expected accuracy.

        Uses conservative estimates:
          - tokens_per_checkpoint = 600 (baseline avg is ~500)
          - budget_safety = 0.80 (target 80% of budget to absorb variance)
          - min_useful_depth = 5 (need enough checkpoints for AIME reasoning)
          - max_checkpoint_cap = 20 (diminishing returns beyond this)
        """
        tokens_per_ckpt = 600
        budget_safety = 0.80
        min_useful_depth = 5
        max_checkpoint_cap = 20

        sizes = [(m, self.extract_size(m)) for m in models]
        min_size = min(s for _, s in sizes)
        if min_size <= 0:
            min_size = 1.0

        usable = compute_budget * budget_safety

        best_model: Optional[str] = None
        best_depth = 0
        best_score = -1.0

        for m, size in sizes:
            weight = size / min_size
            raw_budget = usable / weight
            max_ckpts = int(raw_budget / tokens_per_ckpt)

            if max_ckpts < min_useful_depth:
                continue

            effective_depth = min(max_ckpts, max_checkpoint_cap)

            # Heuristic: larger model with adequate depth is better.
            # Cap depth contribution so we prefer quality over sheer depth.
            score = size * min(effective_depth, 12.0)

            if score > best_score:
                best_score = score
                best_model = m
                best_depth = effective_depth

        if best_model is None:
            # Fallback: smallest model with whatever depth we can afford
            fallback = min(models, key=lambda m: self.extract_size(m))
            weight = self.extract_size(fallback) / min_size
            raw_budget = usable / max(weight, 1.0)
            depth = max(1, min(int(raw_budget / tokens_per_ckpt), max_checkpoint_cap))
            return fallback, depth

        return best_model, best_depth

    # -----------------------------------------------------------------
    # Main solve
    # -----------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        models = list(env.models)
        reqs = env.requirements
        compute_budget = reqs["weighted_compute"]["value"]

        model, max_ckpts = self.select_model_and_depth(models, compute_budget)

        # Spawn a single branch (reveals checkpoint 0 -- counts as 1 checkpoint)
        branch = env.spawn(model)
        bid = branch["branch_id"]

        # Continue until the branch reaches stage_terminal, or we exhaust
        # our checkpoint budget.  (max_ckpts - 1 continues after the spawn.)
        for _ in range(max_ckpts - 1):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(bid)

        answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)
