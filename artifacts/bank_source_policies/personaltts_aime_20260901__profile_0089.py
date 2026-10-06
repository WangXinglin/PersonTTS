from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel majority-vote policy optimised for tight latency constraints.

    The incumbent ASC runs each candidate branch sequentially, so latency
    equals the *sum* of every spawn + continue action.  This policy uses
    ``spawn_many`` and ``continue_many`` to execute a small cohort in
    parallel, reducing latency to ``rounds * max(per-branch step)``
    instead of ``rounds * sum(per-branch step)``.

    Design targets for profile_0089
    --------------------------------
    * accuracy  mean >= 0.30  --> easily met with 3-branch majority vote
    * latency   batch  <= 3776 --> parallel execution keeps worst-case low
    * w-compute mean <= 13.2M --> 3 lightweight Qwen3-4B branches << budget
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_model(models: List[str]) -> str:
        """Return the smallest model available (minimises per-step latency)."""
        for preferred in ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"):
            if preferred in models:
                return preferred
        return models[0]

    @staticmethod
    def _plurality(branches: List[Dict[str, Any]]) -> Optional[str]:
        """Deterministic plurality over active branch answers."""
        answers = []
        for b in branches:
            if b.get("pruned"):
                continue
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                answers.append(a)
        return deterministic_plurality(answers)

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env.models)
        width = 3                     # keep small for low max-per-round latency
        max_checkpoint_rounds = 200   # generous safety net; most branches finish earlier

        # ---- parallel spawn (one latency step for the whole cohort) ----
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            # budget too tight even for 1 branch – finish immediately
            return env.finish()

        actual_width = len(opened)

        # ---- parallel continue rounds -----------------------------------
        for _ in range(max_checkpoint_rounds):
            state = env.observe()
            branches = state["branches"]

            # Check unanimous early-stop *before* paying for another round
            non_pruned = [b for b in branches if not b["pruned"]]
            valid_answers = [
                a
                for b in non_pruned
                if (a := valid_answer(b.get("latest_answer"))) is not None
            ]
            if (
                len(valid_answers) == actual_width
                and len(set(valid_answers)) == 1
            ):
                return env.finish(valid_answers[0])

            # Advance every branch that has not yet terminated
            active = [b for b in non_pruned if b["can_continue"]]
            if not active:
                break

            try:
                env.continue_many([b["branch_id"] for b in active])
            except Exception:
                break  # budget or pool exhausted – vote with what we have

        # ---- final majority vote ----------------------------------------
        return env.finish()
