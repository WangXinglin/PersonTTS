from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Latency-aware parallel Qwen3-4B policy with adaptive checkpoint cap.

    Spawns a cohort of 3 parallel Qwen3-4B branches and advances them
    synchronously checkpoint-by-checkpoint.  Stops as soon as 2/3
    branches agree (majority), or when the adaptive round cap is
    reached to stay within the latency batch budget.

    The round cap is derived from env.requirements["latency_s"]["value"]
    using a conservative per-round latency estimate of 22 s for
    Qwen3-4B width-3 (observed worst ~17 s, with safety margin).
    One round's budget is reserved for the parallel spawn phase.

    For profile_0030 (latency_batch <= 687 s):
      max_rounds = floor((687 - 22) / 22) = 30
      worst-case latency ~ 31 * 22 = 682 s < 687 s

    Accuracy is preserved well above the 0.327 requirement because
    (a) most problems reach majority within a few rounds and are
    unaffected by the cap, and (b) hard problems that run long are
    often wrong regardless of additional rounds.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    # ---- public entry point ------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width = 3
        max_rounds = self._compute_max_rounds()

        # Phase 1: spawn parallel cohort
        branches = self._safe_spawn(env, model, width)
        if not branches:
            return env.finish(None)

        branch_map: Dict[int, Dict[str, Any]] = {}
        for b in branches:
            branch_map[b["branch_id"]] = b

        actual_width = len(branch_map)
        majority_threshold = max(2, (actual_width + 1) // 2)

        # Phase 2: iterate checkpoint rounds with early majority stopping
        for round_idx in range(max_rounds + 1):
            # Gather valid answers from current branch states
            answers: Dict[int, str] = {}
            for bid, b in branch_map.items():
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    answers[bid] = a

            # Majority agreement -> return immediately
            majority_answer = self._find_majority(answers, majority_threshold)
            if majority_answer is not None:
                return env.finish(majority_answer)

            # Identify branches that can still advance
            continuable = [
                bid for bid, b in branch_map.items() if b["can_continue"]
            ]

            # If no branches can continue or we hit the round cap, stop
            if not continuable or round_idx >= max_rounds:
                break

            # Advance all continuable branches in parallel
            updated = self._safe_continue(env, continuable)
            if updated is None:
                break
            for b in updated:
                branch_map[b["branch_id"]] = b

        # Phase 3: fall-back deterministic plurality vote
        votes: List[str] = []
        for b in branch_map.values():
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                votes.append(a)
        return env.finish(deterministic_plurality(votes) if votes else None)

    # ---- internal helpers --------------------------------------------------

    def _compute_max_rounds(self) -> int:
        """Derive the maximum number of continue-many rounds from the
        latency constraint so the worst-case episode stays within budget.

        Uses a conservative per-round estimate (22 s) that exceeds the
        observed worst-case (~17 s) by ~28 % to absorb variance.
        """
        if self.requirements is None:
            return 30
        lat_req = self.requirements.get("latency_s")
        if not isinstance(lat_req, dict):
            return 30
        latency_limit = float(lat_req.get("value", 700.0))
        # Conservative per-round latency for Qwen3-4B width-3.
        # Covers both the spawn phase (1 round) and continue phases.
        per_round_s = 22.0
        # Reserve one round for the spawn phase; the rest for continues.
        max_rounds = max(1, int((latency_limit - per_round_s) / per_round_s))
        return max_rounds

    def _pick_model(self, env) -> str:
        """Select model: prefer Qwen3-4B for proven accuracy on AIME."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        # Fall back to smallest available model
        best_name = env.models[0]
        best_size = self._extract_model_size(best_name)
        for name in env.models[1:]:
            size = self._extract_model_size(name)
            if size < best_size:
                best_size = size
                best_name = name
        return best_name

    def _safe_spawn(
        self, env, model: str, width: int
    ) -> List[Dict[str, Any]]:
        """Spawn *width* branches in parallel, falling back to smaller widths."""
        for w in range(width, 0, -1):
            try:
                if w > 1:
                    return list(env.spawn_many(model, w))
                return [env.spawn(model)]
            except Exception:
                continue
        return []

    def _safe_continue(
        self, env, branch_ids: List[int]
    ) -> Optional[List[Dict[str, Any]]]:
        """Advance branches one checkpoint round in parallel."""
        try:
            if len(branch_ids) == 1:
                return [env.continue_branch(branch_ids[0])]
            return list(env.continue_many(branch_ids))
        except Exception:
            return None

    def _find_majority(
        self, answers: Dict[int, str], threshold: int
    ) -> Optional[str]:
        """Return the answer held by >= threshold branches, or None."""
        if len(answers) < threshold:
            return None
        counts = Counter(answers.values())
        top_answer, top_count = counts.most_common(1)[0]
        return top_answer if top_count >= threshold else None

    def _extract_model_size(self, name: str) -> float:
        """Extract the parameter-count number before 'B' in a model name.

        Examples: 'Qwen3-4B' -> 4.0, 'Qwen3-14B' -> 14.0
        Falls back to inf when no pattern is found.
        """
        upper = name.upper()
        best = float("inf")
        i = 0
        while i < len(upper):
            if upper[i] == "B" and i > 0:
                j = i - 1
                while j >= 0 and (upper[j].isdigit() or upper[j] == "."):
                    j -= 1
                if j < i - 1:
                    try:
                        val = float(name[j + 1 : i])
                        if val < best:
                            best = val
                    except ValueError:
                        pass
            i += 1
        return best
