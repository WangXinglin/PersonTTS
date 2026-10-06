from __future__ import annotations

from typing import Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Minimal single-branch policy targeting profiles with a low accuracy
    floor and tight resource ceilings.

    The baseline ASC spawns ~12 sequential branches to reach 78 % accuracy,
    far exceeding the compute and latency budgets.  Since the accuracy
    requirement is only ~25 %, a single complete branch (per-branch accuracy
    ~55 %) easily clears it while cutting weighted compute to ~132 k (budget
    456 k) and keeping per-problem latency well under 2 264 s.
    """

    NAME = "candidate"

    def _pick_model(self, env) -> str:
        """Select the smallest available model to minimise weighted compute."""
        models = list(env.models)
        # Qwen3-4B has the lowest per-token weight multiplier (~6.67).
        if "Qwen3-4B" in models:
            return "Qwen3-4B"
        # Fallback: lexicographic sort generally puts smaller models first.
        return sorted(models)[0]

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # Single branch keeps both compute and latency at ~1/12 of baseline.
        branch = env.spawn(model)

        # Run the branch to natural termination (stage_terminal → no more
        # checkpoints).  A complete single attempt is cheap and yields the
        # model's best-effort answer.
        while branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])

        answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)
