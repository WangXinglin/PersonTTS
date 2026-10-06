from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Width-5 parallel probe with full-completion plurality voting.

    The width-3 incumbent stops the moment two branches agree (majority
    2/3), even at checkpoint 0 before any branch has finished reasoning.
    Trace evidence shows this locks in wrong answers on many easy-looking
    problems (4-step incorrect traces where two branches immediately
    produce the same wrong answer).

    This candidate fixes both failure modes:

    1.  **Wider cohort (5 branches)** – with diverse wrong answers typical
        of AIME integer problems, the correct answer is more likely to be
        the plurality winner when sampled five times rather than three.
    2.  **Full completion, no premature early-stop** – every branch runs
        to its terminal checkpoint, giving the model its full reasoning
        budget.  Final answers are higher quality than intermediate ones.

    Resource feasibility (Qwen3-4B, width 5, no early stop):

    *   Weighted compute ≈ 5 × 133 k ≈ 665 k  vs budget 4 044 k (6× margin).
    *   Latency: cohort-parallel rounds; max-branch checkpoints ≈ 65,
        batch episode latency ≈ 800–1 200 s  vs budget 3 884 s.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _select_model(self, env) -> str:
        """Prefer Qwen3-4B for known-safe latency/compute profile."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        # Fallback: first available model
        return env.models[0]

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        # --- parallel spawn (width 5, fallback to 3, then 1) ----------
        opened = None
        for w in (5, 3):
            try:
                opened = env.spawn_many(model, w)
                break
            except Exception:
                continue
        if opened is None:
            return self._single_branch_fallback(env, model)

        # --- collect initial answers at checkpoint 0 -------------------
        branch_answers: Dict[int, str] = {}
        for b in opened:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                branch_answers[b["branch_id"]] = ans

        # --- advance ALL branches to completion -----------------------
        for _ in range(300):
            state = env.observe()
            continuable = [
                b
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not continuable:
                break

            try:
                updated = env.continue_many(
                    [b["branch_id"] for b in continuable]
                )
            except Exception:
                break

            for b in updated:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    branch_answers[b["branch_id"]] = ans

        # --- deterministic plurality vote on final answers ------------
        if branch_answers:
            return env.finish(
                deterministic_plurality(list(branch_answers.values()))
            )
        return env.finish()

    # ------------------------------------------------------------------
    # single-branch fallback
    # ------------------------------------------------------------------

    def _single_branch_fallback(self, env, model: str) -> Optional[str]:
        """Run one branch to completion when cohort spawn fails."""
        branch = env.spawn(model)
        while branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])
        ans = valid_answer(branch.get("latest_answer"))
        return env.finish(ans)
