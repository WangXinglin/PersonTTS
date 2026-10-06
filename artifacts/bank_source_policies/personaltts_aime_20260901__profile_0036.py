from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    asc_posterior_confidence,
    deterministic_plurality,
    top_two_counts,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Wide parallel majority-vote with deep execution for profile_0036.

    Root-cause of round 3 accuracy shortfall (72.75% vs 73.10% target)
    ===================================================================
    Round 3 used Qwen3-4B with 10 parallel branches, ASC early stopping
    at conf >= 0.95 with min_depth = 5.  Trace evidence shows:
      - Average depth only ~19 checkpoints per branch (vs baseline's ~37).
      - Many problems early-stopped at rounds 6-10 when 8/10 branches
        reached false consensus on a wrong answer.
      - Fastest incorrect episodes had only 71 trace steps (~6 rounds),
        confirming premature exit is the primary accuracy loss.

    Baseline (sequential ASC, Qwen3-4B, ~37 checkpoints) achieves 78.2%
    accuracy, proving deeper execution directly increases accuracy.

    Fix applied in round 4
    ======================
    1. Increase width from 10 to 14 for more robust majority voting.
       With per-branch accuracy ~0.65, P(correct majority|14) ~ 0.90
       vs ~0.85 for 10 branches.
    2. Raise min_depth from 5 to 15 so branches develop reliable answers
       before any consensus check.  Eliminates the ~6-round false-exit
       failure mode.
    3. Raise confidence threshold from 0.95 to 0.999.  With 14 branches
       this requires 13/14 or 14/14 agreement (conf >= 0.9995), making
       false consensus nearly impossible.
    4. Set max_depth = 60 as a safety cap.  Most branches terminate
       naturally at 15-50 checkpoints; the cap protects batch latency.

    Expected performance for profile_0036
    ======================================
    - Accuracy  ~76-78% mean (budget >= 73.1%)   -- much wider margin
    - Latency   batch ~1200-1500 s (budget 1712.5 s)
    - Compute   mean ~1.5-2.0 M (budget 10.35 M)
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # Wider branches + much deeper execution + very strict early stop
        width = 14
        min_depth = 15
        conf_thresh = 0.999
        max_depth = 60

        # ── Parallel spawn ────────────────────────────────────────────
        opened = self._safe_spawn(env, model, width)
        if opened is None:
            return env.finish(None)

        branch_map: Dict[Any, Dict[str, Any]] = {
            b["branch_id"]: b for b in opened
        }

        # ── Deep parallel execution ──────────────────────────────────
        for round_idx in range(max_depth):
            active_ids = [
                bid
                for bid, b in branch_map.items()
                if b["can_continue"] and not b.get("pruned")
            ]
            if not active_ids:
                break  # All branches reached natural terminal

            try:
                updated = env.continue_many(active_ids)
            except Exception:
                break  # Budget or pool exhausted

            for b in updated:
                branch_map[b["branch_id"]] = b

            # Very conservative early stopping after sufficient depth
            if round_idx >= min_depth:
                answers = self._collect_answers(branch_map)
                if len(answers) >= 2:
                    winner, top1, top2, tied = top_two_counts(answers)
                    if not tied:
                        conf = asc_posterior_confidence(top1, top2)
                        if conf >= conf_thresh:
                            return env.finish(winner)

        # ── Final majority vote ──────────────────────────────────────
        answers = self._collect_answers(branch_map)
        if not answers:
            return env.finish(None)
        return env.finish(deterministic_plurality(answers))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(env: Any) -> str:
        """Prefer Qwen3-4B: cheap enough for wide + deep parallel execution.

        Baseline evidence: deep Qwen3-4B achieves 78.2% accuracy via
        majority voting over ~12 branches at ~37 checkpoints each.
        Larger models (Qwen3-32B) give higher per-branch accuracy but
        blow the batch latency budget at any useful depth.
        """
        for m in ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]:
            if m in env.models:
                return m
        return env.models[0]

    @staticmethod
    def _collect_answers(
        branch_map: Dict[Any, Dict[str, Any]],
    ) -> List[str]:
        """Gather valid answers from all non-pruned branches."""
        answers: List[str] = []
        for b in branch_map.values():
            if b.get("pruned"):
                continue
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                answers.append(a)
        return answers

    @staticmethod
    def _safe_spawn(
        env: Any, model: str, width: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Spawn a parallel cohort, falling back to smaller widths."""
        for w in (
            width,
            max(width * 3 // 4, 4),
            max(width // 2, 2),
            1,
        ):
            try:
                return env.spawn_many(model, w)
            except Exception:
                continue
        try:
            return [env.spawn(model)]
        except Exception:
            return None
