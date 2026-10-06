from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Enhanced wide-parallel majority-vote with answer normalization
    and adaptive reinforcement for weak pluralities.

    Builds on the proven round-02 approach (80 Qwen3-4B branches,
    2 universal refine rounds, ~240 votes per problem, utility 0.999)
    with three targeted improvements aimed at flipping the single
    remaining failing batch:

    1. Answer normalization: merges equivalent string representations
       (e.g. "042" -> "42", "42.0" -> "42", "$42$" -> "42") before
       counting plurality votes, preventing correct-answer vote splits
       that can arise from formatting variation in AIME integer answers.

    2. Answer-history fallback: when a branch's terminal-checkpoint
       answer is invalid, recovers the most recent valid answer from
       its answer_history, reclaiming votes that would otherwise be
       lost.

    3. Adaptive reinforcement: when the main-phase plurality is thin
       (leading answer < 55 % of total votes), spawns a secondary
       cohort of 16 branches with one refine round, adding up to 32
       extra votes on hard problems.  This invests extra compute only
       where the majority signal is weakest, with negligible impact
       on batch-average compute and latency.
    """

    NAME = "candidate"

    MAIN_WIDTH = 80
    REINFORCE_WIDTH = 16
    REINFORCE_THRESHOLD_PCT = 55  # plurality % below which to reinforce
    FALLBACK_WIDTHS = (64, 48, 32, 16, 8, 4, 2)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _pick_model(self, env) -> str:
        """Select the smallest available model for maximum parallelism."""
        if "Qwen3-4B" in env.models:
            return "Qwen3-4B"
        for m in ("Qwen3-8B", "Qwen3-14B", "Qwen3-32B"):
            if m in env.models:
                return m
        return env.models[0]

    def _normalize(self, raw: Optional[str]) -> Optional[str]:
        """Normalize an answer string to a canonical integer form when
        possible, preserving the original string otherwise."""
        if raw is None:
            return None
        s = raw.strip()
        if not s:
            return None
        # Strip common LaTeX wrappers
        if len(s) > 2 and s[0] == "$" and s[-1] == "$":
            s = s[1:-1].strip()
        if s.startswith("\\boxed{") and s.endswith("}"):
            s = s[7:-1].strip()
        # Try to convert to canonical integer string
        try:
            v = float(s)
            if v == int(v) and -1e15 < v < 1e15:
                return str(int(v))
        except (ValueError, OverflowError):
            pass
        return s

    def _spawn_cohort(
        self, env, model: str, target: int
    ) -> List[Dict[str, Any]]:
        """Spawn *target* branches, falling back to smaller sizes."""
        for w in [target] + [f for f in self.FALLBACK_WIDTHS if f < target]:
            if w < 1:
                continue
            try:
                return list(env.spawn_many(model, w))
            except Exception:
                continue
        try:
            return [env.spawn(model)]
        except Exception:
            return []

    def _continue_to_terminal(
        self, env, latest: Dict[Any, Dict[str, Any]]
    ) -> None:
        """Advance every non-pruned, continuable branch until all are
        terminal or the budget is exhausted."""
        while True:
            ids = [
                bid
                for bid, b in latest.items()
                if b.get("can_continue") and not b.get("pruned")
            ]
            if not ids:
                break
            try:
                for b in env.continue_many(ids):
                    latest[b["branch_id"]] = b
            except Exception:
                break

    def _branch_answer(self, b: Dict[str, Any]) -> Optional[str]:
        """Extract the best normalized answer from a branch, falling
        back to the most recent valid answer in its history when the
        terminal checkpoint answer is invalid."""
        a = self._normalize(valid_answer(b.get("latest_answer")))
        if a is not None:
            return a
        # Fallback: scan answer_history from most recent to oldest
        history = b.get("answer_history")
        if isinstance(history, (list, tuple)):
            for h in reversed(history):
                a = self._normalize(valid_answer(h))
                if a is not None:
                    return a
        return None

    def _collect_votes(
        self, latest: Dict[Any, Dict[str, Any]]
    ) -> List[str]:
        """Collect normalized valid answers from every non-pruned branch."""
        out: List[str] = []
        for b in latest.values():
            if not b.get("pruned"):
                a = self._branch_answer(b)
                if a is not None:
                    out.append(a)
        return out

    def _refine_and_collect(
        self, env, latest: Dict[Any, Dict[str, Any]]
    ) -> List[str]:
        """Refine all refinable branches, run to terminal, and collect
        new votes.  Returns an empty list if refinement is unavailable
        or fails."""
        refinable = [
            bid
            for bid, b in latest.items()
            if b.get("can_refine") and not b.get("pruned")
        ]
        if not refinable:
            return []
        try:
            refined = env.refine_many(refinable)
            for b in refined:
                latest[b["branch_id"]] = b
            self._continue_to_terminal(env, latest)
            return self._collect_votes(latest)
        except Exception:
            return []

    def _plurality(self, votes: List[str]) -> Optional[str]:
        """Deterministic plurality: highest-count answer wins, with
        first-appearance as the tiebreaker."""
        if not votes:
            return None
        counts = Counter(votes)
        top = max(counts.values())
        return next(v for v in votes if counts[v] == top)

    # ------------------------------------------------------------------ #
    # main solve                                                           #
    # ------------------------------------------------------------------ #

    def solve(self, env) -> Optional[str]:
        model = self._pick_model(env)

        # --- Phase 1: main parallel cohort --------------------------------
        opened = self._spawn_cohort(env, model, self.MAIN_WIDTH)
        if not opened:
            return env.finish(None)

        main_latest: Dict[Any, Dict[str, Any]] = {
            b["branch_id"]: b for b in opened
        }

        # Stage 0: run all branches to terminal
        self._continue_to_terminal(env, main_latest)
        all_votes: List[str] = self._collect_votes(main_latest)

        # Stage 1: first refinement round
        s1_votes = self._refine_and_collect(env, main_latest)
        all_votes.extend(s1_votes)

        # Stage 2: second refinement round
        s2_votes = self._refine_and_collect(env, main_latest)
        all_votes.extend(s2_votes)

        # --- Phase 2: adaptive reinforcement for weak pluralities ---------
        if all_votes:
            counts = Counter(all_votes)
            top_count = max(counts.values())
            total = len(all_votes)
            if top_count * 100 < total * self.REINFORCE_THRESHOLD_PCT:
                extra_opened = self._spawn_cohort(
                    env, model, self.REINFORCE_WIDTH
                )
                if extra_opened:
                    extra_latest: Dict[Any, Dict[str, Any]] = {
                        b["branch_id"]: b for b in extra_opened
                    }
                    # Run extra branches through stage 0
                    self._continue_to_terminal(env, extra_latest)
                    all_votes.extend(self._collect_votes(extra_latest))
                    # One refinement round for extra branches
                    r1_votes = self._refine_and_collect(env, extra_latest)
                    all_votes.extend(r1_votes)

        # --- Finish with deterministic plurality over normalized votes ----
        answer = self._plurality(all_votes)
        return env.finish(answer)
