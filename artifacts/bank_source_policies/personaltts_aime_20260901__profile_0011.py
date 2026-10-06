from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel Qwen3-4B with correctly calibrated depth cap.

    Round 3 achieved 70.5% accuracy (well above 49.75%) but failed
    batch latency (258s vs 214s budget).  Root cause: per-quantum latency
    was estimated at 7.9s but the actual max-of-8 Qwen3-4B value is
    ~10.5s, so 24 continue rounds overshot.

    Fix: recalibrate per_quantum to 11.0s (conservative), giving 16
    max continues (17 quanta).  Worst-case latency: 17 * 10.5 = 178s,
    well under the 214s budget.  Also lower the early-stop threshold
    from 75% supermajority to simple majority (5/8) starting from
    round 3, so more problems exit early, further compressing the tail.

    Expected: accuracy ~65-68% (>= 49.75%), batch latency ~160-178s
    (<= 214.2s), mean compute ~470K (<= 1.09M).
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _parse_size(name: str) -> float:
        """Extract model size in billions from name string."""
        for part in name.replace("_", "-").split("-"):
            u = part.upper()
            if u.endswith("B"):
                try:
                    return float(u[:-1])
                except ValueError:
                    pass
        return 0.0

    def _pick_model(self, models: List[str]) -> str:
        """Prefer ~4B model for best accuracy-per-compute."""
        parsed = [(m, self._parse_size(m)) for m in models]
        for m, sz in parsed:
            if 3.5 <= sz <= 4.5:
                return m
        above = sorted(
            [(m, sz) for m, sz in parsed if sz >= 1.0], key=lambda x: x[1]
        )
        if above:
            return above[0][0]
        valid = sorted(
            [(m, sz) for m, sz in parsed if sz > 0], key=lambda x: x[1]
        )
        return valid[0][0] if valid else models[0]

    @staticmethod
    def _collect_votes(br: Dict[int, Dict[str, Any]]) -> List[str]:
        """Valid latest answers from non-pruned branches."""
        out: List[str] = []
        for b in br.values():
            if b.get("pruned", False):
                continue
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                out.append(a)
        return out

    @staticmethod
    def _history_fallback(br: Dict[int, Dict[str, Any]]) -> List[str]:
        """Most-recent valid answer per non-pruned branch from history."""
        out: List[str] = []
        for b in br.values():
            if b.get("pruned", False):
                continue
            hist = b.get("answer_history")
            if isinstance(hist, list):
                for ans in reversed(hist):
                    a = valid_answer(ans)
                    if a is not None:
                        out.append(a)
                        break
        return out

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        reqs = env.requirements
        lat_budget = float(reqs["latency_s"]["value"]) if reqs else 200.0
        wc_budget = float(reqs["weighted_compute"]["value"]) if reqs else 1_000_000.0

        model = self._pick_model(list(env.models))
        model_sz = max(self._parse_size(model), 0.5)

        # --- Calibrated per-quantum latency ---
        # Round 3 traces: max-of-8 Qwen3-4B worst-case ~10.5s per quantum.
        # Use 11.0s with model-size scaling for conservative headroom.
        per_quantum_s = 11.0 * (model_sz / 4.0)

        width = 8
        wt = model_sz / 0.6
        tok_per_cp = 520

        # Derive max continues from latency budget with 0.90 safety factor
        max_quanta = max(int(lat_budget * 0.90 / per_quantum_s), 5)
        max_continues = max(max_quanta - 1, 4)

        # Compute guard: reduce depth if weighted compute would overshoot
        est_wc = width * (max_continues + 1) * tok_per_cp * wt
        while est_wc > wc_budget * 0.90 and max_continues > 4:
            max_continues -= 1
            est_wc = width * (max_continues + 1) * tok_per_cp * wt

        # Reduce width if still over compute budget
        while est_wc > wc_budget * 0.90 and width > 3:
            width -= 1
            est_wc = width * (max_continues + 1) * tok_per_cp * wt

        need_majority = width // 2 + 1
        warmup = 3  # Start checking majority consensus from round 3

        # --- Phase 1: Parallel spawn (1 cohort quantum) ---
        opened = None
        for w in sorted(set([width, max(width - 2, 3), 3, 2, 1]), reverse=True):
            try:
                opened = env.spawn_many(model, w) if w > 1 else [env.spawn(model)]
                width = len(opened)
                need_majority = width // 2 + 1
                break
            except Exception:
                continue
        if opened is None:
            return env.finish(None)

        br: Dict[int, Dict[str, Any]] = {b["branch_id"]: b for b in opened}

        # --- Phase 2: Continue rounds with majority early stop ---
        for rnd in range(max_continues):
            active = [
                bid
                for bid, b in br.items()
                if b["can_continue"] and not b.get("pruned", False)
            ]
            if not active:
                break
            try:
                updated = env.continue_many(active)
                for b in updated:
                    br[b["branch_id"]] = b
            except Exception:
                break

            # Majority consensus check after warmup
            if rnd + 1 >= warmup:
                votes = self._collect_votes(br)
                if len(votes) >= need_majority:
                    counts = Counter(votes)
                    top_ans, top_n = counts.most_common(1)[0]
                    if top_n >= need_majority:
                        return env.finish(top_ans)

        # --- Phase 3: Final plurality vote ---
        votes = self._collect_votes(br)
        if not votes:
            votes = self._history_fallback(br)
        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
