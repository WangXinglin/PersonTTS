from __future__ import annotations

from typing import Any, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Deep-reasoning policy with smallest model, max depth, and refinement.

    Combines the incumbent's proven Qwen3-0.6B deep-checkpoint strategy
    with refinement for problems whose initial stage terminates early.

    Key design choices:

    1.  Model: Always selects the smallest available model to minimise
        the per-token weight factor (e.g. 0.6B → factor 1.0), which
        maximises affordable checkpoint depth within the weighted-compute
        budget.  This is what the incumbent (round 1) uses successfully.

    2.  Max depth: Continues all affordable checkpoints in the initial
        stage, collecting every valid answer for self-consistency voting.

    3.  Refinement on short stages: When the initial stage terminates
        before exhausting the budget (short-stage problems use only
        ~20-25% of the budget), the policy refines to enter additional
        stages.  Each refine round gives the model self-correction
        feedback and a fresh reasoning attempt.  This converts wasted
        budget headroom into additional answer diversity, targeting the
        few batches where all 60 problems are wrong.

    4.  Self-consistency voting: All valid answers across all stages
        and checkpoints participate in deterministic plurality voting.
        More evidence from refinement stages improves the chance of
        recovering the correct answer on marginal problems.
    """

    NAME = "candidate"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_size_b(name: str) -> float:
        """Parse approximate parameter count (billions) from a model name."""
        upper = name.upper()
        pos = upper.rfind("B")
        if pos <= 0:
            return 999.0
        j = pos - 1
        while j >= 0 and (name[j].isdigit() or name[j] == "."):
            j -= 1
        fragment = name[j + 1 : pos]
        if not fragment:
            return 999.0
        try:
            return float(fragment)
        except ValueError:
            return 999.0

    def _weight_factor(self, name: str) -> float:
        """Weighted-compute multiplier for a given model."""
        return max(self._model_size_b(name) / 0.6, 0.5)

    def _read_wc(self, env: Any) -> float:
        """Read episode accumulated weighted compute; -1.0 if unavailable."""
        try:
            cost = env.cost
            if isinstance(cost, dict):
                wc = cost.get("weighted_compute")
                if wc is not None:
                    return float(wc)
        except Exception:
            pass
        return -1.0

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def solve(self, env: Any) -> Optional[str]:
        reqs = env.requirements
        if reqs is None:
            return env.finish()

        wc_limit = float(reqs["weighted_compute"]["value"])

        # ---- model selection: smallest available for weight factor ≈ 1.0 ----
        model = min(env.models, key=self._model_size_b)
        weight = self._weight_factor(model)
        est_wc_per_cp = weight * 550.0  # conservative per-checkpoint estimate

        # Static fallback when live cost tracking is unavailable.
        max_cps_static = max(1, int(wc_limit / est_wc_per_cp))

        all_answers: list = []
        cps_used = 0

        # ---- Phase 1: spawn and continue all affordable checkpoints ----
        branch = env.spawn(model)
        cps_used += 1
        a = valid_answer(branch.get("latest_answer"))
        if a is not None:
            all_answers.append(a)

        while branch["can_continue"]:
            wc = self._read_wc(env)
            if wc >= 0.0:
                if wc + est_wc_per_cp > wc_limit:
                    break
            else:
                if cps_used >= max_cps_static:
                    break
            branch = env.continue_branch(branch["branch_id"])
            cps_used += 1
            a = valid_answer(branch.get("latest_answer"))
            if a is not None:
                all_answers.append(a)

        # ---- Phase 2: refine when stage terminated early with headroom ----
        # Only activates for short-stage problems where substantial budget
        # remains (e.g. stage terminated at checkpoint 1-2 with ~75% budget
        # unspent).  Long-stage problems that hit the budget first have
        # can_refine=False or insufficient headroom and skip this block.
        for _ in range(10):  # safety cap on refine iterations
            if not branch.get("can_refine", False):
                break

            # Pre-refine budget check: need room for feedback overhead
            # plus at least one new checkpoint to justify the refine.
            wc = self._read_wc(env)
            if wc < 0.0:
                break  # cannot track cost safely; stop
            if wc + est_wc_per_cp * 2.0 > wc_limit:
                break

            try:
                branch = env.refine(branch["branch_id"])
            except Exception:
                break

            # Collect any answer revealed at refine entry.
            a = valid_answer(branch.get("latest_answer"))
            if a is not None:
                all_answers.append(a)

            # Post-refine: verify feedback cost did not exhaust the budget
            # before continuing into the new stage.
            wc = self._read_wc(env)
            if wc >= 0.0 and wc + est_wc_per_cp > wc_limit:
                break

            # Continue through the new stage's checkpoints.
            while branch["can_continue"]:
                wc = self._read_wc(env)
                if wc >= 0.0:
                    if wc + est_wc_per_cp > wc_limit:
                        break
                else:
                    break  # lost cost tracking; stop
                branch = env.continue_branch(branch["branch_id"])
                cps_used += 1
                a = valid_answer(branch.get("latest_answer"))
                if a is not None:
                    all_answers.append(a)

        # ---- answer selection: self-consistency via plurality ----
        if all_answers:
            return env.finish(deterministic_plurality(all_answers))
        return env.finish()
