from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn policy that uses a capable model with bounded depth.

    The round-0 candidate selected the smallest model (Qwen3-0.6B),
    achieving only 8.7 % accuracy (well below the 27.2 % threshold).
    The template baseline used Qwen3-4B sequentially, achieving 78 %
    accuracy but with 5 377 s average latency (far above the 637 s batch cap).

    This policy bridges the gap:
      - Picks the largest model up to ~4-5 B for strong per-branch accuracy.
      - Spawns a parallel cohort via spawn_many (single latency step).
      - Advances with continue_many for a bounded number of rounds,
        keeping batch latency safely under 637 s.
      - Takes deterministic plurality vote across branches.

    Expected results for Qwen3-4B, width 5, depth 10:
      - Accuracy: 35-55 % (well above 27.2 % threshold)
      - Avg latency: ~180-250 s, batch < 600 s (under 637.4 s threshold)
      - Avg weighted compute: ~180 K (under 8.7 M threshold)
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_size_b(name: str) -> float:
        """Extract the largest *B size number from a model name string."""
        best = 0.0
        i = 0
        while i < len(name):
            if name[i].isdigit() or (
                name[i] == "."
                and i + 1 < len(name)
                and name[i + 1].isdigit()
            ):
                j = i
                while j < len(name) and (name[j].isdigit() or name[j] == "."):
                    j += 1
                if j < len(name) and name[j] in ("B", "b"):
                    try:
                        val = float(name[i:j])
                        if val > best:
                            best = val
                    except ValueError:
                        pass
                i = max(j, i + 1)
            else:
                i += 1
        return best

    def _select_model(self, env: Any) -> str:
        """Pick the largest model whose size is at most ~5 B.

        Qwen3-4B is the known sweet-spot: high per-branch accuracy with
        manageable per-step latency for parallel execution.  Models > 8 B
        are too slow per step to meet the batch latency budget.
        """
        models = list(env.models)
        if len(models) == 1:
            return models[0]

        sizes: Dict[str, float] = {m: self._parse_size_b(m) for m in models}

        # 1. Prefer models in the 3-5 B range (known best tradeoff)
        preferred = [m for m in models if 3.0 <= sizes[m] <= 5.0]
        if preferred:
            return max(preferred, key=lambda m: sizes[m])

        # 2. Largest model <= 8 B
        feasible = [m for m in models if 0 < sizes[m] <= 8.0]
        if feasible:
            return max(feasible, key=lambda m: sizes[m])

        # 3. Fallback: smallest parseable (for latency safety)
        parsed = [m for m in models if sizes[m] > 0]
        if parsed:
            return min(parsed, key=lambda m: sizes[m])

        return models[0]

    def _choose_width_and_depth(self, model_size: float) -> tuple:
        """Return (width, max_continue_rounds) adapted to the model size.

        Larger models need fewer parallel branches and shallower depth
        because each checkpoint step takes longer.
        """
        if model_size <= 1.0:
            # Small model: many branches, more rounds for answer quality
            return 8, 15
        elif model_size <= 5.0:
            # Medium model (incl Qwen3-4B): balanced width + depth
            return 5, 10
        elif model_size <= 8.0:
            # Larger model: reduce to fit latency budget
            return 3, 6
        else:
            # Very large model: single branch, minimal depth
            return 1, 4

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self, env: Any) -> Optional[str]:
        model = self._select_model(env)
        model_size = self._parse_size_b(model)
        width, max_rounds = self._choose_width_and_depth(model_size)

        # 1. Spawn cohort in parallel (single latency step)
        opened: Optional[List[Dict[str, Any]]] = None
        for w in range(width, 0, -1):
            try:
                opened = env.spawn_many(model, w)
                break
            except Exception:
                continue
        if opened is None:
            return env.finish(None)

        branch_ids: List[Any] = [b["branch_id"] for b in opened]
        actual_width = len(branch_ids)
        answers: Dict[Any, str] = {}

        # Collect answers from spawn checkpoint
        for b in opened:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                answers[b["branch_id"]] = ans

        # 2. Continue rounds (each continue_many is one parallel latency step)
        for round_idx in range(max_rounds):
            state = env.observe()
            by_id: Dict[Any, Dict[str, Any]] = {
                b["branch_id"]: b for b in state["branches"]
            }

            advancing = [
                bid
                for bid in branch_ids
                if not by_id[bid]["pruned"] and by_id[bid]["can_continue"]
            ]
            if not advancing:
                break

            try:
                updated = env.continue_many(advancing)
            except Exception:
                break

            for b in updated:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    answers[b["branch_id"]] = ans

            # Conservative early stop: unanimous agreement after min rounds
            if (
                round_idx >= 4
                and len(answers) >= actual_width
                and len(set(answers.values())) == 1
            ):
                break

        # 3. Deterministic plurality vote
        votes = list(answers.values())
        final = deterministic_plurality(votes) if votes else None
        return env.finish(final)
