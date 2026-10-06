from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel full-execution majority-vote policy for Qwen3-4B.

    The baseline (ASC with Qwen3-4B) reaches 78 % accuracy by running
    each branch to its natural terminal checkpoint, but fails latency
    because every spawn/continue is *sequential* (each adds its own
    wall-clock time).  This policy fixes that with three changes:

    1. **spawn_many** – opens all branches in one parallel phase so the
       spawn latency is max(N) instead of sum(N).
    2. **continue_many** – advances every active branch in one
       synchronous round; each round charges only the max branch time.
    3. **No early stopping** – lets every branch reach its natural
       terminal checkpoint so each vote reflects the model's
       fully-reasoned answer.  Round 0 failed because majority-2-of-3
       at checkpoint 0 locked in unreliable shallow answers.

    With width 7 and parallel semantics the per-episode latency is
    roughly  max_terminal_checkpoint × max(7 branch-round latencies),
    which stays well under the 1 042 s batch budget while delivering
    accuracy above the 60.83 % threshold through plurality voting over
    7 independently-reasoned terminal answers.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _select_model(self, env) -> str:
        """Pick the model with proven terminal accuracy from baseline data."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        return env.models[0]

    # ------------------------------------------------------------------ #
    # main solve loop                                                      #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)

        width = 7          # parallel branches – good accuracy/latency balance
        max_rounds = 200   # generous safety cap; branches terminate much sooner

        # ---------- spawn cohort (single parallel phase) ----------
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            return env.finish(None)

        branch_ids = [b["branch_id"] for b in opened]
        bstate: Dict[int, Dict[str, Any]] = {
            b["branch_id"]: b for b in opened
        }
        latest: Dict[int, str] = {}

        # Collect any answers already visible at checkpoint 0
        for b in opened:
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                latest[b["branch_id"]] = a

        # ---------- run all branches to terminal ----------
        for _ in range(max_rounds):
            active = [
                bid for bid in branch_ids
                if bstate[bid]["can_continue"] and not bstate[bid]["pruned"]
            ]
            if not active:
                break

            try:
                updated = env.continue_many(active)
            except Exception:
                break  # budget exhausted – vote on what we have

            for b in updated:
                bid = b["branch_id"]
                bstate[bid] = b
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    latest[bid] = a

        # ---------- plurality vote on terminal answers ----------
        votes = list(latest.values())
        if votes:
            return env.finish(deterministic_plurality(votes))
        return env.finish(None)
