from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Parallel wide majority vote running all branches to completion.

    Trace analysis of prior rounds revealed:

    1. Qwen3-4B is the optimal model for AIME: the baseline achieves 78.2%
       accuracy when branches run to completion. Qwen3-32B (round 0, 47.9%)
       and Qwen3-0.6B (round 1, 20.1%) both failed dramatically.

    2. The round 2 candidate spawned 48 parallel Qwen3-4B branches but only
       achieved 67.2% accuracy because ASC early stopping exited with wrong
       consensus on intermediate (incomplete) answers. Average trace steps
       were 809 (only ~16 parallel checkpoint rounds) vs the baseline's ~37
       checkpoints per branch needed for full reasoning completion.

    3. Latency and compute budgets are massively under-utilised: round 2 had
       mean latency margin of +2408s and mean compute margin of +17.9M.

    This policy fixes the accuracy gap by:
    - Spawning 48 branches of Qwen3-4B in parallel via spawn_many
    - Running ALL branches to stage-terminal completion via continue_many
      with NO early stopping whatsoever
    - Voting only on final terminal answers via deterministic plurality

    Expected performance:
    - Accuracy: >= 79.8% (baseline's per-branch quality + wider 48-vote)
    - Latency batch: ~2000s (parallel execution, well under 3286s limit)
    - Weighted compute mean: ~6.1M (48 * 19K tokens * 6.667 weight, under 20.7M)
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_size_b(name: str) -> float:
        """Parse approximate parameter count in billions from model name."""
        upper = name.upper()
        best = 0.0
        i = 0
        n = len(upper)
        while i < n:
            if upper[i].isdigit() or (
                upper[i] == "." and i + 1 < n and upper[i + 1].isdigit()
            ):
                j = i
                while j < n and (upper[j].isdigit() or upper[j] == "."):
                    j += 1
                if j < n and upper[j] == "B":
                    try:
                        val = float(upper[i:j])
                        if val > best:
                            best = val
                    except ValueError:
                        pass
                i = max(i + 1, j)
            else:
                i += 1
        return best if best > 0.0 else 4.0

    def _select_model(self, env: Any) -> str:
        """Select the model closest to 4B for optimal AIME accuracy.

        Qwen3-4B is empirically proven: the baseline achieves 78.2% accuracy.
        Smaller (0.6B: 20.1%) and larger (32B: 47.9%) models both fail
        because 4B has the best accuracy-to-depth ratio for math reasoning.
        """
        models = list(env.models)
        if not models:
            return ""
        sizes = {m: self._model_size_b(m) for m in models}
        return min(models, key=lambda m: abs(sizes[m] - 4.0))

    def _determine_width(self, env: Any, model: str) -> int:
        """Determine the largest affordable parallel width.

        Target 48 branches for robust majority voting. The compute budget
        (20.7M) easily affords 48 Qwen3-4B branches running to completion
        (~6.1M estimated weighted compute).
        """
        target_widths = [48, 40, 32, 24, 20, 16, 12, 8, 4, 2, 1]
        for w in target_widths:
            try:
                ok = env.can_afford_complete_candidates(
                    model,
                    count=w,
                    checkpoints_per_stage=80,
                    max_refine_rounds=0,
                )
                if ok:
                    return w
            except (AttributeError, TypeError):
                return 48
        return 1

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env: Any) -> Optional[str]:
        model = self._select_model(env)
        width = self._determine_width(env, model)

        # --- Spawn the entire cohort in one parallel phase ---------------
        opened: Optional[List[Dict[str, Any]]] = None
        attempts = [width]
        for fallback in [width * 3 // 4, width // 2, width // 4, 12, 8, 4, 2, 1]:
            if fallback < 1 or fallback >= width:
                continue
            attempts.append(fallback)
        for w in attempts:
            try:
                opened = env.spawn_many(model, w)
                break
            except Exception:
                continue
        if opened is None:
            return env.finish(None)

        branch_ids: List[int] = [int(b["branch_id"]) for b in opened]

        # --- Run ALL branches to completion via parallel checkpoint rounds --
        # NO early stopping: every branch must reach stage_terminal=true
        # before we collect votes. This is the critical fix over round 2,
        # which lost 11% accuracy by exiting on wrong intermediate consensus.
        for _ in range(500):
            state = env.observe()
            by_id: Dict[int, Dict[str, Any]] = {
                int(b["branch_id"]): b for b in state["branches"]
            }
            can_cont: List[int] = [
                bid
                for bid in branch_ids
                if bid in by_id
                and not by_id[bid]["pruned"]
                and by_id[bid]["can_continue"]
            ]
            if not can_cont:
                break

            try:
                env.continue_many(can_cont)
            except Exception:
                break

        # --- Final deterministic plurality vote on terminal answers ------
        state = env.observe()
        by_id = {int(b["branch_id"]): b for b in state["branches"]}
        votes: List[str] = []
        for bid in branch_ids:
            if bid in by_id and not by_id[bid]["pruned"]:
                ans = valid_answer(by_id[bid].get("latest_answer"))
                if ans is not None:
                    votes.append(ans)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
