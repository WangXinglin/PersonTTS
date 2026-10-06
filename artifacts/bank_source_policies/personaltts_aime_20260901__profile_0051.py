from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Latency-optimised parallel majority voting for profile_0051.

    Targets the binding batch-latency constraint (<= 818 s) by capping the
    continue-round budget and tightening early-stop thresholds, while
    keeping mean accuracy safely above the 47.3 % floor.

    Key changes vs round-02 candidate (utility 0.582):
      - max_continue_rounds: 65 -> 45   (caps worst-case latency ~730 s)
      - warmup_rounds:       5  -> 3    (consensus eligibility 2 rounds earlier)
      - stable_agree_rounds: 3  -> 2    (faster consensus lock-in)
      - NEW immediate majority stop after round 20 (no consecutive requirement)
      - NEW late pruning of branches that never produced a valid answer

    With round 02's per-round latency of 12-16 s, capping at 45 rounds
    brings the worst-case (currently ~1043 s) down to ~730 s and the
    former batch (~848 s) down to ~587 s, both well under the 818 s limit.
    The accuracy margin (63.5 % -> ~60 %) stays above the 47.3 % floor.
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        # ── model selection ────────────────────────────────────────
        model = "Qwen3-4B"
        if model not in env.models:
            model = env.models[0]

        # ── configuration tuned for batch latency <= 818 s ──────────
        desired_width = 5
        max_continue_rounds = 45      # hard cap (was 65)
        warmup_rounds = 3             # min rounds before consensus (was 5)
        stable_agree_rounds = 2       # consecutive rounds for early consensus (was 3)
        immediate_majority_round = 20 # after this, any majority stops immediately
        prune_no_answer_round = 25    # prune branches with no valid answer

        # ── phase 1: parallel spawn ───────────────────────────────
        opened = self._try_spawn(env, model, desired_width)
        if not opened:
            return env.finish(None)

        actual_width = len(opened)
        maj = actual_width // 2 + 1   # 3 for width=5

        branch_ids: List[int] = []
        last_valid: Dict[int, str] = {}
        terminal_answers: Dict[int, Optional[str]] = {}

        for b in opened:
            bid = int(b["branch_id"])
            branch_ids.append(bid)
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                last_valid[bid] = ans
            if b.get("stage_terminal"):
                terminal_answers[bid] = ans

        # immediate terminal majority (unlikely right after spawn)
        winner = self._check_majority(terminal_answers, maj)
        if winner is not None:
            return env.finish(winner)

        bid_set = set(branch_ids)
        agree_streak = 0
        last_consensus: Optional[str] = None

        # ── phase 2: parallel continue rounds ─────────────────────
        for round_idx in range(max_continue_rounds):
            state = env.observe()
            to_continue: List[int] = [
                int(b["branch_id"])
                for b in state["branches"]
                if int(b["branch_id"]) in bid_set
                and not b["pruned"]
                and b["can_continue"]
            ]

            if not to_continue:
                break

            try:
                updated = env.continue_many(to_continue)
            except Exception:
                break

            for b in updated:
                bid = int(b["branch_id"])
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    last_valid[bid] = ans
                if b.get("stage_terminal"):
                    terminal_answers[bid] = ans

            # ── early stop 1: terminal majority ───────────────────
            winner = self._check_majority(terminal_answers, maj)
            if winner is not None:
                return env.finish(winner)

            # ── early stop 2: consensus-based ─────────────────────
            if round_idx >= warmup_rounds:
                consensus = self._get_consensus(last_valid, maj)
                if consensus is not None:
                    # After enough rounds, any majority is reliable
                    if round_idx >= immediate_majority_round:
                        return env.finish(consensus)
                    # Before that, require consecutive agreement
                    if consensus == last_consensus:
                        agree_streak += 1
                    else:
                        agree_streak = 1
                        last_consensus = consensus
                    if agree_streak >= stable_agree_rounds:
                        return env.finish(consensus)
                else:
                    agree_streak = 0
                    last_consensus = None

            # ── late pruning of answer-less branches ──────────────
            if round_idx == prune_no_answer_round:
                self._prune_no_answer(env, bid_set, last_valid,
                                      terminal_answers)

        # ── phase 3: fallback plurality vote ──────────────────────
        votes: List[str] = []
        for bid in branch_ids:
            if bid in terminal_answers and terminal_answers[bid] is not None:
                votes.append(terminal_answers[bid])
            elif bid in last_valid:
                votes.append(last_valid[bid])

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)

    # ── helpers ────────────────────────────────────────────────────

    def _try_spawn(
        self, env: Any, model: str, desired: int
    ) -> List[Dict[str, Any]]:
        """Spawn *desired* branches in parallel, falling back gracefully."""
        for w in (desired, 3, 1):
            if w < 1 or w > desired:
                continue
            try:
                if w > 1:
                    return list(env.spawn_many(model, w))
                return [env.spawn(model)]
            except Exception:
                continue
        return []

    def _check_majority(
        self, answers: Dict[int, Optional[str]], threshold: int
    ) -> Optional[str]:
        """Return majority answer among terminal branches, or None."""
        valid = [a for a in answers.values() if a is not None]
        if len(valid) < threshold:
            return None
        counts = Counter(valid)
        best, count = counts.most_common(1)[0]
        return best if count >= threshold else None

    def _get_consensus(
        self, answers: Dict[int, str], threshold: int
    ) -> Optional[str]:
        """Return the consensus answer if enough branches currently agree."""
        if len(answers) < threshold:
            return None
        counts = Counter(answers.values())
        best, count = counts.most_common(1)[0]
        return best if count >= threshold else None

    def _prune_no_answer(
        self,
        env: Any,
        bid_set: set,
        last_valid: Dict[int, str],
        terminal_answers: Dict[int, Optional[str]],
    ) -> None:
        """Prune branches that have never produced a valid answer.

        These contribute nothing to voting and their removal reduces
        per-round latency for subsequent continue_many rounds.
        Keeps at least 3 unpruned branches.
        """
        state = env.observe()
        active_bids: List[int] = []
        no_answer_bids: List[int] = []
        for b in state["branches"]:
            bid = int(b["branch_id"])
            if bid in bid_set and not b["pruned"]:
                active_bids.append(bid)
                if bid not in last_valid and bid not in terminal_answers:
                    no_answer_bids.append(bid)

        # Keep at least 3 active branches
        max_to_prune = max(0, len(active_bids) - 3)
        for bid in no_answer_bids[:max_to_prune]:
            env.prune(bid)
