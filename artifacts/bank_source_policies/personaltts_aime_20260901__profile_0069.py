from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Wide parallel spawn with full-depth completion for maximum accuracy.

    The incumbent uses width=5 with early consensus stopping and an 11-round
    depth cap, yielding only 0.481 per-problem accuracy.  The binding
    constraint is accuracy (965/1000 batches satisfied); latency and compute
    have 12x and 108x headroom respectively.

    Key improvements:
    1. Width 9 (vs 5) for stronger majority voting.
    2. All branches run to natural stage completion -- no premature consensus
       lock-in and no artificial depth cap.  Baseline evidence shows Qwen3-4B
       branches need ~37 checkpoints on average (up to 65+) to finish
       reasoning; the incumbent truncates at 11.
    3. With per-branch accuracy ~0.60 at full depth (inferred from baseline
       0.782 accuracy with ~12 sequential branches), majority-of-9 voting
       should yield ~0.73 per-problem accuracy, pushing batch satisfaction
       well above 0.965.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_model(self, models: list) -> str:
        """Select model -- prefer Qwen3-4B for proven, predictable latency."""
        preference = [
            "Qwen3-4B",
            "Qwen3-8B",
            "Qwen3-14B",
            "Qwen3-32B",
            "Qwen3-72B",
        ]
        for name in preference:
            if name in models:
                return name
        return sorted(models)[0]

    def _collect_votes(self, env) -> List[str]:
        """Gather valid latest answers from all unpruned branches."""
        state = env.observe()
        votes: List[str] = []
        for branch in state["branches"]:
            if not branch["pruned"]:
                ans = valid_answer(branch.get("latest_answer"))
                if ans is not None:
                    votes.append(ans)
        return votes

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env.models)
        width = 9

        # Phase 1: Spawn all branches in parallel (one latency step)
        try:
            env.spawn_many(model, width)
        except Exception:
            # Graceful fallback if full width is unavailable
            try:
                env.spawn_many(model, 5)
            except Exception:
                env.spawn(model)

        # Phase 2: Run every branch to natural stage completion.
        # Each continue_many round adds only the max cohort latency.
        # Branches drop out as they reach stage_terminal, so later
        # rounds are progressively cheaper.
        while True:
            state = env.observe()
            active = [
                b for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not active:
                break
            try:
                env.continue_many([b["branch_id"] for b in active])
            except Exception:
                break

        # Phase 3: Deterministic plurality vote on final answers
        votes = self._collect_votes(env)
        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
