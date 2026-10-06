from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Wide parallel spawn with calibrated depth for accuracy-latency balance.

    Targets profile_0008 constraints:
      accuracy  >= 76.28% (mean)  -> 96 parallel branches + depth 40 reasoning
      latency   batch <= 593.78s    -> depth 40 caps batch at ~568s (margin +26s)
      wcompute  mean <= 12.08M    -> 96 * ~25ckpt * ~520tok * 6.67 ~ 7.3M

    Key calibration from prior rounds:
      - Round 2 (depth=36, width=48): accuracy 75.73%, latency p99=467s
        -> accuracy barely missed 76.28% by 0.55%; latency had +134s margin
      - Round 3 (depth=48, width=96): accuracy 78.33%, latency batch~680s
        -> accuracy well above threshold; latency exceeded budget by 86s
      - Per-round latency at p99: 701/49 ~ 14.3s/round

    This candidate splits the difference: depth=40 (vs 36 and 48) yields:
      - batch latency: ~41 phases * 14.0s ~ 574s (within 594s budget)
      - Accuracy: boosted above 76.28% by both deeper reasoning (+4 depth
        units vs round 2) and wider voting (96 vs 48 branches)
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _select_model(self, models: Any) -> str:
        """Pick Qwen3-4B for well-characterized latency and compute profile."""
        model_set = set(models)
        if "Qwen3-4B" in model_set:
            return "Qwen3-4B"
        for m in ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]:
            if m in model_set:
                return m
        return list(models)[0]

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env.models)

        # ---- parameters calibrated from round 2/3 trace data ----------------
        # Depth 40: 4 more than round 2 (boosts AIME_25 accuracy from 71.55%
        # toward 76.66%), but 8 fewer than round 3 (saves ~112s at batch,
        # bringing it from ~680s down to ~568s, well within 594s budget).
        max_continue_rounds = 40

        # Width 96: parallel branches for robust majority voting.
        # In cohort semantics, each continue_many adds only the max single-
        # branch latency, so more branches improve accuracy "for free" in
        # terms of latency phases (minor extreme-value inflation per round).
        target_width = 96
        # ---------------------------------------------------------------------

        # Spawn parallel cohort, gracefully falling back to smaller widths
        opened = None
        for w in [target_width, 80, 64, 48, 32, 16, 8, 4]:
            try:
                opened = env.spawn_many(model, w)
                break
            except Exception:
                continue

        if opened is None:
            # Last-resort single-branch fallback
            branch = env.spawn(model)
            for _ in range(max_continue_rounds):
                if not branch["can_continue"]:
                    break
                branch = env.continue_branch(branch["branch_id"])
            return env.finish(valid_answer(branch.get("latest_answer")))

        branch_ids = set()
        latest_answer: Dict[int, str] = {}

        for b in opened:
            bid = int(b["branch_id"])
            branch_ids.add(bid)
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                latest_answer[bid] = a

        # ---- deep parallel continue without early stopping ------------------
        # No early stopping: avoids premature wrong-consensus failures
        # (round 1's main accuracy failure mode) and ensures every branch
        # reasons to its natural depth (or the safety cap), maximizing
        # per-branch answer quality before the final majority vote.
        for rnd in range(max_continue_rounds):
            state = env.observe()
            continuable = [
                int(b["branch_id"])
                for b in state["branches"]
                if int(b["branch_id"]) in branch_ids
                and not b["pruned"]
                and b["can_continue"]
            ]
            if not continuable:
                break

            try:
                updated = env.continue_many(continuable)
            except Exception:
                break  # budget boundary

            for b in updated:
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    latest_answer[int(b["branch_id"])] = a

        # ---- final majority vote at full depth ------------------------------
        votes = list(latest_answer.values())
        if not votes:
            return env.finish()
        return env.finish(deterministic_plurality(votes))
