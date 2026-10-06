from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Wide parallel Qwen3-4B majority-vote with full branch completion.

    Diagnosis from rounds 0-2 and baseline:
      - The baseline (sequential ASC, Qwen3-4B, ~12 branches avg) achieves
        ~78.2% accuracy, nearly meeting the 79% target, but fails the
        batch latency constraint because sequential spawns accumulate latency
        (avg 5,377s, p90 >21,000s vs 3,337s limit).
      - Rounds 0-2 used parallel spawning (spawn_many/continue_many) which
        fixes latency, BUT checked for consensus after every checkpoint
        round.  This stopped branches mid-reasoning (~12 of ~37 checkpoints),
        dramatically lowering per-branch accuracy and overall accuracy to
        55.7% (round 2).

    This policy fixes both failure modes:
      1. Parallel execution via spawn_many + continue_many collapses latency
         from O(width * depth) to O(depth).  Even the hardest problems
         (p99 from round 2) finish in ~1,043s, well under 3,337s.
      2. NO intermediate consensus checking: every branch runs to full
         stage_terminal completion (can_continue = false) before voting.
         This preserves the deep reasoning that gives the baseline its
         high per-branch accuracy (~60%).
      3. Wide vote: ~36 parallel branches boost majority-vote accuracy to
         ~83-85% from individual per-branch ~60%, exceeding the 79% target.
      4. Compute stays well under budget: 36 branches x ~132K WC each
         = 4.75M << 9.73M limit.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _select_model(self, env) -> str:
        """Pick the available model closest to 4B parameters.

        Qwen3-4B has the best observed per-branch accuracy on AIME among
        tested models.  If several models tie in distance to 4B, prefer
        the smaller one (cheaper per token).
        """
        models = list(env.models)
        if len(models) == 1:
            return models[0]

        best_name = models[0]
        best_dist = float("inf")
        best_size = float("inf")

        for name in models:
            try:
                token = name.split("-")[-1].upper().rstrip("B")
                size = float(token)
            except (ValueError, IndexError):
                size = float("inf")
            dist = abs(size - 4.0)
            if dist < best_dist or (dist == best_dist and size < best_size):
                best_dist = dist
                best_size = size
                best_name = name

        return best_name

    def _target_width(self, env, model: str) -> int:
        """Compute target parallel width from the compute budget.

        Uses 55% of the weighted-compute limit as a per-episode ceiling.
        The actual mean is lower because easy problems have shorter branches.
        Clamps width to [8, 48].
        """
        req = env.requirements
        if req is None:
            return 32

        compute_limit = float(req["weighted_compute"]["value"])

        try:
            token = model.split("-")[-1].upper().rstrip("B")
            model_size = float(token)
        except (ValueError, IndexError):
            model_size = 4.0

        weight = model_size / 0.6
        # Conservative per-branch raw-token estimate (baseline avg ~20K)
        est_wc_per_branch = 22000.0 * weight
        budget = compute_limit * 0.55
        affordable = int(budget / max(est_wc_per_branch, 1.0))
        return max(8, min(affordable, 48))

    # ------------------------------------------------------------------
    # main controller
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)
        target = self._target_width(env, model)

        # Check available pool to size the cohort accurately
        state = env.observe()
        remaining = state.get("remaining_by_model", {}).get(model, 0)
        if remaining > 0:
            width = min(target, remaining)
        else:
            width = target

        # Spawn a parallel cohort; halve width on pool/budget failure
        opened = None
        while width >= 2:
            try:
                opened = env.spawn_many(model, width)
                break
            except Exception:
                width = width // 2

        # Absolute fallback: single sequential branch
        if opened is None:
            branch = env.spawn(model)
            while branch["can_continue"]:
                branch = env.continue_branch(branch["branch_id"])
            return env.finish(valid_answer(branch.get("latest_answer")))

        # ----------------------------------------------------------
        # Run ALL branches to full completion (no early stopping).
        # This is the critical difference from rounds 0-2: we do NOT
        # check consensus or prune during reasoning.  Every branch
        # runs until can_continue is false, preserving the full
        # reasoning chain that gives high per-branch accuracy.
        # ----------------------------------------------------------
        for _ in range(2000):
            state = env.observe()
            continuable = [
                b
                for b in state["branches"]
                if b["can_continue"] and not b["pruned"]
            ]
            if not continuable:
                break
            try:
                env.continue_many(
                    [b["branch_id"] for b in continuable]
                )
            except Exception:
                break  # budget or pool exhausted

        # Final majority vote over all completed branches
        state = env.observe()
        votes = []
        for b in state["branches"]:
            if not b["pruned"]:
                a = valid_answer(b.get("latest_answer"))
                if a is not None:
                    votes.append(a)
        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
