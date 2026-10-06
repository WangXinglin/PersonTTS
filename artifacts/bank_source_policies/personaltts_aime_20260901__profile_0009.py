from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Maximum-depth single-branch policy with self-consistent answer selection.

    Designed for profile_0009's tight weighted-compute budget (~14.2K mean).

    Key improvements over rounds 0 and 1:

    Round 0 used Qwen3-4B but stopped at the *first* valid answer (often
    at checkpoint 1), wasting ~42% of the compute budget.  Round 1 switched
    to the heavier Qwen3-8B (weight 13.33 vs 6.67), which could afford only
    2 checkpoints and achieved only 1.9% accuracy.

    This policy:
    1. Uses Qwen3-4B (best proven accuracy-per-compute-unit).
    2. Runs all affordable checkpoints (4 with current budget) without
       early-stopping, so the model reasons as deeply as possible.
    3. Uses the last valid answer produced by the branch. Baseline traces
       show later checkpoints refine answers and correct earlier mistakes.
    4. Falls back to plurality over the full answer_history when the final
       checkpoint reverts to null.
    """

    NAME = "candidate"

    # Confirmed weight factors: weighted_compute / raw_tokens.
    MODEL_WEIGHTS: Dict[str, float] = {
        "Qwen3-0.6B": 1.0,
        "Qwen3-1.7B": 2.8333,
        "Qwen3-4B": 6.6667,
        "Qwen3-8B": 13.3333,
        "Qwen3-14B": 23.3333,
        "Qwen3-32B": 53.3333,
    }

    # Observed ~530 raw tokens per checkpoint across models.
    TOKENS_PER_CHECKPOINT: int = 530

    # Model preference: Qwen3-4B first (proven best accuracy per compute
    # unit), then lighter models that allow more depth, then heavier ones.
    MODEL_PREFERENCE: List[str] = [
        "Qwen3-4B",
        "Qwen3-1.7B",
        "Qwen3-8B",
        "Qwen3-0.6B",
        "Qwen3-14B",
        "Qwen3-32B",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ── helpers ───────────────────────────────────────────────────────

    def _weight_for(self, model: str) -> float:
        """Return the estimated size-weight factor for *model*."""
        if model in self.MODEL_WEIGHTS:
            return self.MODEL_WEIGHTS[model]
        parts = model.rsplit("-", 1)
        if len(parts) == 2 and parts[1].upper().endswith("B"):
            try:
                size = float(parts[1][:-1])
                return size * 1.6667
            except (ValueError, IndexError):
                pass
        return 10.0

    def _compute_limit(self, env: Any) -> float:
        """Read the weighted-compute ceiling from the live requirements."""
        if env.requirements is not None:
            wc = env.requirements.get("weighted_compute")
            if isinstance(wc, dict):
                return float(wc.get("value", 14157.0))
        return 14157.0

    def _choose_model(self, env: Any, budget: float) -> str:
        """Pick the model with the best expected depth within *budget*.

        Prefer Qwen3-4B (proven), requiring >= 3 affordable checkpoints
        so the model has enough tokens to produce answers.  Fall back to
        any model that can afford >= 2 checkpoints.
        """
        available = set(env.models)
        # First pass: >= 3 checkpoints (enough for Qwen3-4B to produce answers)
        for name in self.MODEL_PREFERENCE:
            if name in available:
                cost = self.TOKENS_PER_CHECKPOINT * self._weight_for(name)
                if budget / cost >= 3.0:
                    return name
        # Second pass: >= 2 checkpoints
        for name in self.MODEL_PREFERENCE:
            if name in available:
                cost = self.TOKENS_PER_CHECKPOINT * self._weight_for(name)
                if budget / cost >= 2.0:
                    return name
        # Last resort: any available model
        for name in self.MODEL_PREFERENCE:
            if name in available:
                return name
        return list(env.models)[0]

    def _max_checkpoints(self, budget: float, model: str) -> int:
        """Maximum checkpoint count that fits the mean compute budget."""
        cost_per = self.TOKENS_PER_CHECKPOINT * self._weight_for(model)
        return max(1, int(budget / cost_per))

    @staticmethod
    def _plurality(answers: List[str]) -> Optional[str]:
        """Deterministic plurality: break ties by first appearance."""
        if not answers:
            return None
        counts = Counter(answers)
        top = max(counts.values())
        return next(a for a in answers if counts[a] == top)

    # ── main entry point ─────────────────────────────────────────────

    def solve(self, env: Any) -> Optional[str]:
        budget = self._compute_limit(env)
        model = self._choose_model(env, budget)
        cap = self._max_checkpoints(budget, model)

        # Phase 1: spawn (reveals checkpoint 0)
        branch = env.spawn(model)
        used = 1

        # Phase 2: continue to the cap (or until branch cannot continue).
        # Do NOT early-stop: later checkpoints refine the answer.
        while used < cap and branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])
            used += 1

        # Phase 3: select the best answer from the full history.
        # Primary: last valid answer (most refined).
        # Fallback: plurality over all valid historical answers.
        history = branch.get("answer_history", [])
        valid_history = [valid_answer(a) for a in history]
        valid_history = [v for v in valid_history if v is not None]

        # The latest_answer covers the current checkpoint; use it if valid.
        last = valid_answer(branch.get("latest_answer"))
        if last is not None:
            answer = last
        elif valid_history:
            answer = self._plurality(valid_history)
        else:
            answer = None

        return env.finish(answer)
