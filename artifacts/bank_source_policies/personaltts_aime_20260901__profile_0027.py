from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Wide parallel plurality with budget-calibrated model/width/depth.

    Evidence from prior rounds:
    - Round 02: Qwen3-4B, 11 branches, ~4.2 continues -> 28% accuracy
      but 200K weighted compute (over 190,793 budget).
    - Round 01: Qwen3-14B, 1 branch, 3 continues -> 12.4% accuracy,
      39K compute (massive headroom but insufficient accuracy).
    - Round 03: Qwen3-14B, 4 branches, 2 continues -> 11% accuracy
      (wider but shallower hurt accuracy vs single deep branch).

    Strategy: use the cheapest model (Qwen3-4B) with maximum width
    that fits the compute budget at moderate depth. This maximises
    the number of independent samples for plurality voting. AIME
    wrong answers are diverse, so P(correct wins) ~ P(>=1 branch
    correct) ~ 1-(1-p)^W, which scales well with width W.

    Uses answer_history fallback to recover valid answers from
    earlier checkpoints when latest_answer is null.
    """

    NAME = "candidate"

    @staticmethod
    def _model_size(name: str) -> float:
        for part in name.replace("_", "-").split("-"):
            upper = part.upper()
            if upper.endswith("B"):
                try:
                    return float(upper[:-1])
                except ValueError:
                    pass
        return float("inf")

    def _choose_model(self, models: List[str]) -> str:
        """Pick the smallest usable model (Qwen3-4B preferred)."""
        by_size = sorted(models, key=self._model_size)
        for m in by_size:
            s = self._model_size(m)
            if 0 < s < float("inf"):
                return m
        return models[0]

    @staticmethod
    def _best_answer(branch: Dict[str, Any]) -> Optional[str]:
        """Return latest valid answer, falling back to answer_history."""
        ans = valid_answer(branch.get("latest_answer"))
        if ans is not None:
            return ans
        history = branch.get("answer_history")
        if history:
            for past in reversed(history):
                ans = valid_answer(past)
                if ans is not None:
                    return ans
        return None

    def _collect_votes(self, branches: Dict[int, Dict[str, Any]]) -> List[str]:
        votes: List[str] = []
        for b in branches.values():
            ans = self._best_answer(b)
            if ans is not None:
                votes.append(ans)
        return votes

    def _get_config(self, model_size_b: float) -> tuple:
        """Return (width, base_continues) for the given model size."""
        reqs = self.requirements or {}
        constraints = reqs.get("constraints", [])
        wc_budget = 200000.0
        lat_budget = 60.0
        for c in constraints:
            metric = c.get("metric")
            if metric == "weighted_compute":
                wc_budget = float(c["value"])
            elif metric == "latency_s":
                lat_budget = float(c["value"])

        # Weight factor and per-checkpoint cost for the chosen model
        wf = max(model_size_b / 0.6, 1.0)
        raw_per_cp = 530  # conservative estimate including probe
        wc_per_cp = raw_per_cp * wf

        # Use 92% of compute budget as safety margin
        wc_target = wc_budget * 0.92

        # Latency: each phase ~7s for small models, ~8s for larger
        phase_lat = 5.0 + model_size_b * 0.3
        # Max sequential phases within 90% of latency budget
        max_phases = max(2, int(lat_budget * 0.90 / phase_lat))

        best = (1, 3)
        best_score = 0.0
        for width in range(16, 0, -1):
            for cont in range(min(max_phases - 1, 6), 2, -1):
                total_cps = width * (cont + 1)
                est_wc = total_cps * wc_per_cp
                if est_wc > wc_target:
                    continue
                # Phases = 1 (spawn) + cont (continues)
                if cont + 1 > max_phases:
                    continue
                # Need at least 3 branches for meaningful plurality
                if width < 3:
                    continue
                # Score: balance width and depth
                # Width gives more independent samples (primary)
                # Depth gives more reliable per-branch answers
                score = width * 1.0 + cont * 0.5
                if score > best_score:
                    best_score = score
                    best = (width, cont)

        return best

    def solve(self, env) -> Optional[str]:
        models = list(env.models)
        model = self._choose_model(models)
        size = self._model_size(model)
        width, base_continues = self._get_config(size)

        # Phase 1: spawn parallel cohort
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            try:
                opened = env.spawn_many(model, max(1, width // 2))
            except Exception:
                opened = [env.spawn(model)]

        branches: Dict[int, Dict[str, Any]] = {
            b["branch_id"]: b for b in opened
        }

        # Phases 2..N: base continue rounds (all branches)
        for _ in range(base_continues):
            continuable = [
                bid for bid, b in branches.items()
                if b["can_continue"] and not b.get("pruned", False)
            ]
            if not continuable:
                break
            try:
                updated = env.continue_many(continuable)
            except Exception:
                break
            for b in updated:
                branches[b["branch_id"]] = b

        # Collect votes using best available answer per branch
        votes = self._collect_votes(branches)

        # Selective adaptive: if too few votes, continue ONLY branches
        # without valid answers (1 extra round, minimal compute)
        if len(votes) < 3:
            need_more = [
                bid for bid, b in branches.items()
                if b["can_continue"]
                and not b.get("pruned", False)
                and self._best_answer(b) is None
            ]
            if need_more:
                try:
                    updated = env.continue_many(need_more)
                except Exception:
                    pass
                else:
                    for b in updated:
                        branches[b["branch_id"]] = b
                    votes = self._collect_votes(branches)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
