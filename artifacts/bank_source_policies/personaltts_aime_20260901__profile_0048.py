from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import math

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Accuracy-optimised policy using a larger model with parallel branches.

    Profile 0048 requires accuracy >= 26.6 % (mean) with generous latency
    (2328 s batch) and compute (274 754 mean weighted-compute) budgets.

    Round 1 showed Qwen3-0.6B x17 branches reaches only 23 % accuracy and
    cannot meet the threshold regardless of width.  The baseline template
    proves Qwen3-4B achieves ~78 % accuracy with many branches but blows
    the compute budget.

    Strategy: spawn 2 parallel Qwen3-4B branches (est. ~264 K weighted-
    compute mean, under 274 K budget).  Two branches give a confirmation
    signal on easy problems and 50 % odds on hard ones, yielding well
    above the 26.6 % accuracy target.  If only a small model is available,
    fall back to maximum-width parallel majority voting.
    """

    NAME = "candidate"

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _parse_size(name: str) -> float:
        """Extract parameter-count in billions from a model name."""
        best = 999999.0
        for part in name.replace("-", "_").split("_"):
            if len(part) > 1 and part[-1].upper() == "B":
                try:
                    val = float(part[:-1])
                    if 0 < val < best:
                        best = val
                except ValueError:
                    pass
        return best

    def _choose_config(
        self, models: List[str], wc_budget: float
    ) -> Tuple[str, int]:
        """Pick model and width that maximise expected accuracy in budget.

        Heuristic: single-branch accuracy scales steeply with model size
        (0.6B ~14 %, 4B ~50 %).  Majority-voting gain has diminishing
        returns.  Prefer the largest model that allows width >= 2.
        Fall back to width 1 of the largest affordable model, then to
        maximum width of the smallest model.
        """
        sizes = {m: self._parse_size(m) for m in models}
        smallest_size = min(sizes.values())

        # Estimated per-branch weighted-compute by model size.
        # 0.6B: ~14 K raw x1 weight = 14 K wc.
        # 4B: ~20 K raw x6.67 weight ≈ 133 K wc.
        def _est_wc(model: str) -> float:
            sz = sizes[model]
            wf = sz / smallest_size if smallest_size > 0 else 1.0
            raw = 14000.0 * (1.0 + 0.06 * (wf - 1.0))
            return raw * wf

        # Sort models from largest to smallest.
        ranked = sorted(models, key=lambda m: sizes[m], reverse=True)

        # Phase 1 – largest model that allows width >= 2.
        for model in ranked:
            est = _est_wc(model)
            max_w = max(1, int(wc_budget * 0.92 / est))
            if max_w >= 2:
                return model, min(max_w, 32)

        # Phase 2 – largest model at width 1 (if its single-branch
        # accuracy is likely high enough).
        for model in ranked:
            est = _est_wc(model)
            if est <= wc_budget * 0.92:
                wf = sizes[model] / smallest_size if smallest_size > 0 else 1.0
                if wf >= 3.0:          # only prefer big model at width 1
                    return model, 1

        # Phase 3 – fall back to smallest model, max width.
        model = ranked[-1]
        est = _est_wc(model)
        max_w = max(1, int(wc_budget * 0.92 / est))
        return model, min(max_w, 32)

    # -----------------------------------------------------------------
    # main solve
    # -----------------------------------------------------------------
    def solve(self, env) -> Optional[str]:
        reqs = env.requirements
        wc_budget = 274754.0
        if reqs is not None:
            wc_budget = float(reqs["weighted_compute"]["value"])

        models = list(env.models)
        model, width = self._choose_config(models, wc_budget)

        # --- single branch fast path ---
        if width <= 1:
            branch = env.spawn(model)
            while branch["can_continue"]:
                branch = env.continue_branch(branch["branch_id"])
            return env.finish(valid_answer(branch.get("latest_answer")))

        # --- parallel spawn + full completion ---
        opened = env.spawn_many(model, width)
        branch_ids = [b["branch_id"] for b in opened]

        while True:
            state = env.observe()
            by_id = {b["branch_id"]: b for b in state["branches"]}
            advancing = [
                bid for bid in branch_ids
                if not by_id[bid]["pruned"] and by_id[bid]["can_continue"]
            ]
            if not advancing:
                break
            try:
                env.continue_many(advancing)
            except Exception:
                break

        # --- deterministic plurality vote ---
        state = env.observe()
        votes: List[str] = []
        for branch in state["branches"]:
            if branch["pruned"]:
                continue
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                votes.append(ans)

        answer: Optional[str] = None
        if votes:
            counts = Counter(votes)
            top = max(counts.values())
            answer = next(v for v in votes if counts[v] == top)

        return env.finish(answer)
