from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Sequential-branch majority-vote with tighter depth budgeting.

    Uses up to 3 sequential Qwen3-4B branches with 2-of-3 majority early
    stopping.  The previous round used headroom_factor=1.4 giving 14 ckpts
    per branch and mean weighted compute ~119K (over the 109,651 budget).
    Reducing headroom to 1.15 gives 12 ckpts per branch.  Estimated mean
    ~103K comfortably fits the budget while the accuracy loss from 2 fewer
    checkpoints is minimal (current 0.52 vs required 0.405).

    Additionally, a global step budget caps the total spawn+continue actions
    across all branches, and the last branch adaptively uses only the
    remaining step allowance.
    """

    NAME = "candidate"

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_size_b(name: str) -> float:
        """Extract parameter count (billions) from a model name like 'Qwen3-4B'."""
        buf = ""
        for ch in name:
            if ch.isdigit() or (ch == "." and buf and buf[-1].isdigit()):
                buf += ch
            else:
                if buf and ch in ("B", "b"):
                    try:
                        return float(buf)
                    except ValueError:
                        pass
                buf = ""
        return float("inf")

    @staticmethod
    def _get_wc_budget(reqs: Optional[Dict[str, Any]]) -> float:
        """Extract weighted-compute budget from a requirements dict."""
        if reqs is None:
            return 200000.0
        wc = reqs.get("weighted_compute")
        if isinstance(wc, dict) and "value" in wc:
            return float(wc["value"])
        for c in reqs.get("constraints", []):
            if c.get("metric") == "weighted_compute":
                return float(c["value"])
        return 200000.0

    def _choose_model(self, env: Any) -> str:
        """Pick the largest model whose estimated mean cost fits the budget.

        Formula: 2.5 effective branches x 8 avg ckpts x 520 tok/step x weight.
        Iterates largest-first, returning the first model that fits.
        """
        models = list(env.models)
        if not models:
            raise ValueError("no models available in env.models")

        wc_budget = self._get_wc_budget(env.requirements)

        sized: List[tuple] = [(m, self._parse_size_b(m)) for m in models]
        sized.sort(key=lambda x: x[1])

        chosen = sized[0][0]  # default: smallest model
        for name, size_b in reversed(sized):
            if size_b == float("inf"):
                continue
            weight = max(size_b / 0.6, 1.0)
            estimated_mean_wc = 2.5 * 8 * 520 * weight
            if estimated_mean_wc <= wc_budget:
                chosen = name
                break

        return chosen

    # ------------------------------------------------------------------ #
    # solve                                                                #
    # ------------------------------------------------------------------ #

    def solve(self, env: Any) -> Optional[str]:
        model = self._choose_model(env)

        # Read budget from env.requirements (authoritative runtime source)
        wc_budget = self._get_wc_budget(env.requirements)

        # Estimate per-step weighted compute for the chosen model
        model_size = self._parse_size_b(model)
        if model_size == float("inf"):
            model_size = 4.0
        weight = max(model_size / 0.6, 1.0)
        cost_per_step = 520.0 * weight

        max_branches = 3
        agreement = 2  # 2-of-3 early stop

        # Global step budget: 1.15x headroom for the mean constraint.
        # Easy problems early-stop well below this, keeping the batch mean
        # under the actual budget even when hard problems use the full cap.
        max_total_steps = int(wc_budget * 1.15 / cost_per_step)

        # Per-branch depth cap (checkpoints including the spawn checkpoint)
        max_ckpts = max(5, min(20, max_total_steps // max_branches))

        steps_used = 0
        votes: List[str] = []

        for _ in range(max_branches):
            # Remaining step allowance for this and any subsequent branches
            remaining = max_total_steps - steps_used
            if remaining < 3:
                break  # not enough budget for a meaningful branch

            # This branch can use up to max_ckpts, but no more than remaining
            branch_depth = min(max_ckpts, remaining)

            # Spawn a new branch (checkpoint 0)
            try:
                branch: Dict[str, Any] = env.spawn(model)
            except Exception:
                break  # pool or budget exhausted
            steps_used += 1

            # Advance to natural terminal or depth cap
            for _ in range(branch_depth - 1):  # spawn was checkpoint 0
                if not branch["can_continue"]:
                    break
                try:
                    branch = env.continue_branch(branch["branch_id"])
                except Exception:
                    break
                steps_used += 1

            # Collect vote from this branch
            ans = valid_answer(branch.get("latest_answer"))
            if ans is not None:
                votes.append(ans)

            # Early stop: enough agreement across branches
            if len(votes) >= agreement:
                counts = Counter(votes)
                top_ans, top_count = counts.most_common(1)[0]
                if top_count >= agreement:
                    return env.finish(top_ans)

        # Fallback: deterministic plurality of whatever votes we have
        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish()
