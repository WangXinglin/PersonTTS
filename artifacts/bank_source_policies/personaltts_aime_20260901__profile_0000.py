from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Wide-and-deep parallel policy for high accuracy within latency budget.

    Key improvements over round 0 (17.6% accuracy, 26.7s latency):
    1. Width 16 (vs 4): far more branches for reliable plurality voting.
    2. Depth 11 checkpoints (vs 6): nearly double thinking time per branch,
       converting many null-answer branches into valid-answer branches.
    3. No premature early stopping: round 0 stopped at consensus-of-2 which
       frequently locked in wrong answers on hard problems.
    4. Pure plurality vote at the end: avoids bias toward early (shallow) answers.

    Latency estimate: 11 parallel rounds * ~8s/round = ~88s batch, within the
    92.33s budget.  Weighted compute: ~16*11*600*6.67 = ~704K, well within
    the 10.78M budget.
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        # Model selection: use Qwen3-4B for known latency characteristics
        model = self._select_model(env)

        # Configuration tuned for ~92s batch latency budget with Qwen3-4B
        width = 16
        max_continues = 10  # 11 total parallel rounds (spawn + 10 continues)

        # Verify pool availability
        remaining = int(env.observe()["remaining_by_model"].get(model, 0))
        actual_width = min(width, remaining)
        if actual_width < 1:
            return env.finish(None)

        # Phase 1: Parallel spawn — creates all branches in one latency round
        try:
            opened = env.spawn_many(model, actual_width)
        except Exception:
            return env.finish(None)
        branch_ids = [int(b["branch_id"]) for b in opened]

        # Phase 2: Parallel continue rounds — each advances all branches by 1 checkpoint
        for _ in range(max_continues):
            state = env.observe()
            by_id = {int(b["branch_id"]): b for b in state["branches"]}
            continuable = [
                bid for bid in branch_ids
                if by_id[bid]["can_continue"] and not by_id[bid]["pruned"]
            ]
            if not continuable:
                break
            try:
                env.continue_many(continuable)
            except Exception:
                break

        # Phase 3: Gather final answers from all branches and vote
        state = env.observe()
        answers = self._gather_votes(state, branch_ids)

        if answers:
            return env.finish(deterministic_plurality(answers))
        return env.finish(None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_model(env) -> str:
        """Select model, preferring Qwen3-4B for known latency profile."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        # Fallback: pick the first available model (lexicographic)
        return sorted(env.models)[0]

    @staticmethod
    def _gather_votes(
        state: Dict[str, Any], branch_ids: List[int]
    ) -> List[str]:
        """Collect the latest valid answer from each unpruned branch."""
        answers: List[str] = []
        branch_set = set(branch_ids)
        for b in state["branches"]:
            if int(b["branch_id"]) in branch_set and not b["pruned"]:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    answers.append(ans)
        return answers
