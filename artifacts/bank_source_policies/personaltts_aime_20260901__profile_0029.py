from __future__ import annotations

from typing import Any, Optional

from policies.base import (
    ControllerPolicy,
    asc_posterior_confidence,
    deterministic_plurality,
    top_two_counts,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Budget-aware ASC policy with mid-size model and moderate width/depth.

    Profile 0029 requires accuracy >= 34.28 %, latency batch <= 5 179 s,
    and weighted-compute mean <= 147 489.

    Round 0 used the smallest model (Qwen3-0.6B, width 1) and achieved
    only 10.5 % accuracy with huge compute headroom.  The template
    baseline uses Qwen3-4B with ~12 wide branches and ~40 checkpoints
    each, achieving 78 % accuracy but blowing the compute budget by 10x.

    This policy:
    - Selects a ~4 B model (best accuracy-per-compute for our budget).
    - Runs up to 3 sequential branches with ASC early stopping
      (confidence threshold 0.85; stops after 2 matching votes).
    - Caps each branch at ~11 checkpoints so worst-case weighted
      compute stays within budget (~132 k < 147 k).
    - On easy problems the first two branches agree and ASC stops
      early, saving ~33 % of the budget.
    - Latency is ~36 sequential steps x ~12 s = ~430 s, well within
      the 5 179 s batch limit.
    """

    NAME = "candidate"

    # ------------------------------------------------------------------ #
    # Deterministic helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_size(name: str) -> float:
        """Extract approximate parameter count (billions) from a model name.

        Handles names like ``Qwen3-4B``, ``Qwen3-0.6B``, ``Qwen3-14B``.
        Returns 999.0 when no size token is found.
        """
        upper = name.upper()
        best = 999.0
        for i, c in enumerate(upper):
            if c == "B" and i > 0:
                j = i - 1
                while j >= 0 and (upper[j].isdigit() or upper[j] == "."):
                    j -= 1
                tok = upper[j + 1 : i]
                if tok:
                    try:
                        v = float(tok)
                        if 0.0 < v < best:
                            best = v
                    except ValueError:
                        pass
        return best

    def _pick_config(
        self, models: list[str]
    ) -> tuple[str, int, int]:
        """Choose (model, width, max_continues) within the compute budget.

        Prefers a model near 4 B for the best accuracy / weighted-compute
        trade-off.  Tries width 3 first (good majority voting), then
        falls back to width 2 or 1 with proportionally more depth.
        """
        reqs = self.requirements
        wc_lim = float(reqs["weighted_compute"]["value"]) if reqs else 150000.0

        # Conservative raw-tokens-per-action estimate (baseline average
        # is ~519 for Qwen3-4B; use 550 for safety).
        tok_step = 550
        min_depth = 5

        # Sort models by closeness to 4 B (ideal), breaking ties toward
        # the larger model.
        sized = [(self._parse_size(m), m) for m in models]
        sized.sort(key=lambda x: (abs(x[0] - 4.0), -x[0]))

        for _sz, m in sized:
            sz = self._parse_size(m)
            # Weight factor mirrors replay: params / 0.6
            wt = max(sz / 0.6, 1.0)
            wc_step = tok_step * wt
            # 0.95 safety margin against per-step token variance
            step_budget = wc_lim * 0.95 / wc_step

            # Try widths in preference order
            for w in (3, 2, 4, 1):
                # -1 because spawn already uses one step
                d = int(step_budget / w) - 1
                if d >= min_depth:
                    return m, w, min(d, 25)

        # Absolute fallback: smallest model, single deep branch
        fallback = min(models, key=self._parse_size)
        return fallback, 1, 20

    def _can_spawn(self, env: Any, model: str, max_continues: int) -> bool:
        """Check whether one complete branch can still be afforded."""
        return bool(
            env.can_afford_complete_candidates(
                model,
                count=1,
                checkpoints_per_stage=max_continues + 1,
                max_refine_rounds=0,
            )
        )

    # ------------------------------------------------------------------ #
    # Main solve loop                                                      #
    # ------------------------------------------------------------------ #

    def solve(self, env: Any) -> Optional[str]:
        model, width, max_cont = self._pick_config(list(env.models))

        votes: list[str] = []

        for _ in range(width):
            # Budget / pool guard
            if not self._can_spawn(env, model, max_cont):
                break

            # Spawn one branch and advance it
            branch = env.spawn(model)
            bid = branch["branch_id"]

            for _ in range(max_cont):
                if not branch["can_continue"]:
                    break
                branch = env.continue_branch(bid)

            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                votes.append(ans)

            # ASC early stopping: two matching votes give confidence
            # 0.875 >= 0.85, so we can stop one branch early on easy
            # problems while retaining high reliability.
            if len(votes) >= 2:
                winner, top1, top2, tied = top_two_counts(votes)
                if not tied:
                    conf = asc_posterior_confidence(top1, top2)
                    if conf >= 0.85:
                        return env.finish(winner)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
