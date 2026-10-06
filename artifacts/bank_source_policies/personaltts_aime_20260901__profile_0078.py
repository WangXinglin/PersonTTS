from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Single-branch largest-model run to terminal for minimal latency.

    Round 3 (width-3 Qwen3-32B with locked-majority exit) achieved 0.078
    utility: 100% accuracy and compute satisfaction but only 7.8% latency
    satisfaction.  The batch batch latency exceeded the 4107 s budget by
    ~418 s on average because each continue_many round pays max-of-3
    per-checkpoint latency, inflating total latency -- especially on the
    hard AIME_25 tail (p90 = 4674 s, p99 = 9001 s with width 3).

    This policy eliminates the max-of-N overhead entirely:

    1. Spawn ONE branch of the largest available model.
    2. Run it to terminal via sequential continue_branch calls.
    3. Return the terminal answer.

    Latency improvement: each continue_branch adds only that single
    branch's checkpoint latency (no max-of-N amplification).  For an
    i.i.d. per-checkpoint distribution, E[max(3)] / E[1] ~ 1.5--1.8,
    so total per-episode latency drops by ~35--45 %.  Additionally,
    width 1 draws a SINGLE branch checkpoint count N instead of max(N0,
    N1, N2) rounds, further reducing the number of sequential phases.

    Estimated per-episode latency distribution (width 1 vs width 3):
      - p50:  ~180 s  (was 295 s)
      - p90:  ~1800 s (was 2975 s)
      - batch:  ~3000 s (was ~4525 s)  -- below 4107 s budget
      - p99:  ~5500 s (was 8947 s)

    Accuracy: per-branch Qwen3-32B terminal accuracy ~ 70--72 %, far
    above the 46.2 % requirement.  All 1000 batches should satisfy
    accuracy (binomial mean 43/60 with std 3.5, threshold 28/60).

    Compute: ~670 K per episode (1/3 of width 3), well below 20.87 M.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _select_model(self, env: Any) -> str:
        """Pick the largest available model for highest per-branch accuracy."""
        best_model: Optional[str] = None
        best_size: float = -1.0
        for m in env.models:
            size = 0.0
            for part in m.replace("-", " ").replace("_", " ").split():
                upper = part.upper()
                if upper.endswith("B"):
                    try:
                        size = float(upper[:-1])
                    except ValueError:
                        pass
            if size > best_size:
                best_size = size
                best_model = m
        return best_model if best_model is not None else sorted(env.models)[0]

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env: Any) -> Optional[str]:
        model = self._select_model(env)

        # Phase 1: Single spawn -------------------------------------------
        try:
            branch = env.spawn(model)
        except Exception:
            return env.finish()

        bid = branch["branch_id"]
        best_answer = valid_answer(branch.get("latest_answer"))

        # Phase 2: Run to terminal ----------------------------------------
        while branch["can_continue"]:
            try:
                branch = env.continue_branch(bid)
            except Exception:
                break
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                best_answer = ans

        # Phase 3: Return terminal answer ---------------------------------
        final_answer = valid_answer(branch.get("latest_answer"))
        if final_answer is None:
            final_answer = best_answer

        return env.finish(final_answer)
