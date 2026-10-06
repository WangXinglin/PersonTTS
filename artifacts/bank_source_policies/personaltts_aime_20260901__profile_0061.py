from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Adaptive model+width selection for maximal accuracy under latency budget.

    Profile 0061 constraints:
      accuracy  >= 1.13%  (mean)  -- binding constraint
      latency   <= 11.05s (batch)
      compute   <= 20.7M  (mean)

    Incumbent (round 3): Qwen3-8B, width=5, CP0 only -> 72.3% utility.
    All correct answers come from AIME_24 (2.41%/problem), AIME_25 = 0%.
    Latency (~6s) and compute (~34K vs 345K budget) have huge headroom.

    Improvements over incumbent:
    1. Model selection: prefer the largest model available (up to ~14B)
       to boost per-branch accuracy, especially for AIME_25.
       A larger model is far more likely to solve hard competition
       problems than a smaller model with more branches.
    2. Width 7 for models <= 8B (up from 5), width 5 for 8-14B.
       More branches improve plurality voting on borderline problems.
    3. Single spawn_many phase only: guarantees latency stays ~6-8s,
       well under the 11.05s batch threshold.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_param_size(name: str) -> float:
        """Extract numeric parameter-count proxy from model name (e.g. '8B').

        Returns math.inf if no size pattern is found.
        """
        upper = name.upper()
        best = math.inf
        i = 0
        while i < len(upper):
            if upper[i] == "B" and i > 0:
                j = i - 1
                dot_seen = False
                while j >= 0:
                    ch = upper[j]
                    if ch.isdigit():
                        j -= 1
                    elif ch == "." and not dot_seen:
                        dot_seen = True
                        j -= 1
                    else:
                        break
                start = j + 1
                if start < i:
                    try:
                        val = float(upper[start:i])
                        if val < best:
                            best = val
                    except ValueError:
                        pass
            i += 1
        return best

    def _select_model_and_width(self, models: list) -> tuple:
        """Pick the best (model, width) for accuracy under latency budget.

        Prefer the largest model whose single-phase latency fits the budget.
        For models up to 8B: width 7 (cheap per checkpoint, more voting).
        For models 8B-14B: width 5 (heavier per checkpoint, still affordable).
        For models >14B: width 3 (expensive but highest per-branch accuracy).
        """
        sized = [(m, self._model_param_size(m)) for m in models]
        sized.sort(key=lambda x: x[1])

        # Prefer largest model that fits latency in a single phase.
        # With ~6s base for 8B, estimate ~0.75s per additional 1B.
        # Budget 11.05s => max safe model ~ 14B for single phase.
        # But be conservative: cap at 14B to avoid latency surprises.
        best_model = None
        best_size = 0.0
        for m, s in sized:
            if s <= 14.0:
                best_model = m
                best_size = s

        if best_model is None:
            # All models are >14B or have no parseable size; use smallest
            best_model = sized[0][0]
            best_size = sized[0][1]

        # Width based on model size
        if best_size <= 4.0:
            width = 9
        elif best_size <= 8.0:
            width = 7
        elif best_size <= 14.0:
            width = 5
        else:
            width = 3

        return best_model, width

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model, target_width = self._select_model_and_width(env.models)

        # Single parallel spawn phase -- latency = max-of-cohort.
        # Try decreasing widths if the pool cannot satisfy the target.
        opened = None
        for w in range(target_width, 0, -1):
            try:
                if w == 1:
                    opened = [env.spawn(model)]
                else:
                    opened = env.spawn_many(model, w)
                break
            except Exception:
                continue

        if not opened:
            return env.finish()

        # Collect valid answers from checkpoint 0
        answers: List[str] = []
        for branch in opened:
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                answers.append(ans)

        # Finish with deterministic plurality of CP0 answers.
        # No continues: keeps every episode in one latency phase,
        # guaranteeing batch latency stays well under 11.05s.
        if answers:
            return env.finish(deterministic_plurality(answers))

        # All branches returned null at CP0 (very rare at width >= 5).
        # Accept zero accuracy rather than adding a continue phase.
        return env.finish()
