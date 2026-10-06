from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn policy: spawn a small Qwen3-4B cohort via spawn_many,
    advance all branches in parallel via continue_many, and take majority vote.

    The baseline's latency bottleneck comes from sequential spawning: latency is
    the SUM of all branch latencies.  By using spawn_many and continue_many the
    latency becomes the MAX of branch latencies within each parallel phase,
    cutting batch latency by an order of magnitude while retaining high accuracy
    through majority voting over 3 independent branches.
    """

    NAME = "candidate"

    def _pick_model(self, env) -> str:
        """Select smallest available model for minimal per-checkpoint latency."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        # Fallback: first listed model
        return env.models[0]

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)
        width = 3  # small parallel cohort, ample for majority vote

        # ---- parallel spawn ------------------------------------------------
        opened = env.spawn_many(model, width)
        branch_ids = [b["branch_id"] for b in opened]

        # Track the latest valid answer per branch (checkpoint 0 may already
        # have one).
        latest = {}  # type: Dict[int, str]
        for b in opened:
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                latest[b["branch_id"]] = a

        # ---- parallel continue until all branches terminal -----------------
        for _ in range(500):  # generous upper bound; branches typically finish < 70
            state = env.observe()
            by_id = {b["branch_id"]: b for b in state["branches"]}
            advancing = [
                bid
                for bid in branch_ids
                if by_id[bid]["can_continue"] and not by_id[bid]["pruned"]
            ]
            if not advancing:
                break
            try:
                updated = env.continue_many(advancing)
            except Exception:
                break
            for b in updated:
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    latest[b["branch_id"]] = a

        # ---- deterministic plurality vote ----------------------------------
        votes = [latest[bid] for bid in branch_ids if bid in latest]
        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
