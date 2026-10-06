from __future__ import annotations

from typing import Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Minimal single-branch policy for tight compute and latency budgets.

    For profile_0026 the accuracy bar is only 0.391 while weighted-compute
    mean must stay under ~191 K and latency batch under ~2 146 s.  The baseline
    ASC spawns ~11.6 Qwen3-4B branches per problem, consuming ~1.53 M weighted
    compute on average and producing tail latencies far above the threshold.

    This policy spawns exactly one Qwen3-4B branch, runs it to its natural
    terminal checkpoint, and returns the answer.  Expected per-problem weighted
    compute is ~132 K (well within budget); per-problem latency even for hard
    cases is ~1 000 s (well within the batch cap); single-branch accuracy for
    Qwen3-4B on AIME is estimated at 55-70 %, far above the 39.1 % floor.
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        # Pick the lightest model to minimise weighted-compute cost.
        model = "Qwen3-4B"
        if model not in env.models:
            model = env.models[0]

        # Spawn one branch and advance it to natural termination.
        branch = env.spawn(model)
        while branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])

        # Prefer the terminal answer; fall back to last valid observed answer.
        answer = valid_answer(branch.get("latest_answer"))
        if answer is None:
            for past in reversed(branch.get("answer_history", [])):
                answer = valid_answer(past)
                if answer is not None:
                    break

        return env.finish(answer)
