from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel 3-branch majority vote with terminal-branch early stopping.

    Spawns 3 branches in parallel via spawn_many, advances them concurrently
    with continue_many, and returns the majority answer as soon as 2 terminal
    branches agree.  This cuts both latency (parallel vs sequential) and
    compute (3 branches vs ~12 in the baseline ASC) while maintaining
    sufficient accuracy through majority voting.
    """

    NAME = "parallel_majority_3"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _select_model(self, env) -> str:
        """Pick the cheapest available model (Qwen3-4B preferred)."""
        models = env.models
        if "Qwen3-4B" in models:
            return "Qwen3-4B"
        return sorted(models)[0]

    def _terminal_majority(
        self, terminal: Dict[int, Optional[str]], threshold: int
    ) -> Optional[str]:
        """Return the answer with >= *threshold* terminal votes, or None."""
        valid = [a for a in terminal.values() if a is not None]
        if len(valid) < threshold:
            return None
        counts = Counter(valid)
        best, cnt = counts.most_common(1)[0]
        return best if cnt >= threshold else None

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)
        width = 3
        threshold = 2  # majority = 2 out of 3

        # --- spawn all branches in parallel ---
        try:
            opened = env.spawn_many(model, width)
        except Exception:
            return self._solve_single(env, model)

        branch_ids: List[int] = [int(b["branch_id"]) for b in opened]
        terminal: Dict[int, Optional[str]] = {}

        # record any branches already terminal at spawn
        for b in opened:
            bid = int(b["branch_id"])
            if not b["can_continue"]:
                terminal[bid] = valid_answer(b.get("latest_answer"))

        # immediate check (unlikely but correct)
        ans = self._terminal_majority(terminal, threshold)
        if ans is not None:
            return env.finish(ans)

        # --- advance in parallel rounds ---
        for _ in range(500):
            # identify branches still running
            state = env.observe()
            by_id: Dict[int, Dict[str, Any]] = {
                int(b["branch_id"]): b for b in state["branches"]
            }
            active = [
                bid
                for bid in branch_ids
                if bid not in terminal
                and not by_id[bid]["pruned"]
                and by_id[bid]["can_continue"]
            ]
            if not active:
                break

            # advance one checkpoint round
            try:
                updated = env.continue_many(active)
            except Exception:
                break

            # record newly-terminal branches
            for b in updated:
                bid = int(b["branch_id"])
                if not b["can_continue"]:
                    terminal[bid] = valid_answer(b.get("latest_answer"))

            # check terminal majority
            ans = self._terminal_majority(terminal, threshold)
            if ans is not None:
                return env.finish(ans)

        # --- fallback: final terminal majority check ---
        ans = self._terminal_majority(terminal, threshold)
        if ans is not None:
            return env.finish(ans)

        # plurality over every branch that has a non-null answer
        state = env.observe()
        bid_set = set(branch_ids)
        votes: List[str] = []
        for b in state["branches"]:
            if int(b["branch_id"]) in bid_set and not b["pruned"]:
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    votes.append(a)

        return env.finish(deterministic_plurality(votes) if votes else None)

    # ------------------------------------------------------------------
    # fallback for insufficient pool
    # ------------------------------------------------------------------

    def _solve_single(self, env, model: str) -> Optional[str]:
        """Single-branch fallback when spawn_many is unavailable."""
        branch = env.spawn(model)
        while branch["can_continue"]:
            try:
                branch = env.continue_branch(int(branch["branch_id"]))
            except Exception:
                break
        ans = valid_answer(branch.get("latest_answer"))
        return env.finish(ans)
