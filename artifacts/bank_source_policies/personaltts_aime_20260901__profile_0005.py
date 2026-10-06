from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Budget-aware policy: deep primary branch with answer-history recovery
    and a cheap-model fallback for null-answer episodes.

    Profile 0005 requires:
      - accuracy  (mean) >= 26.13 %
      - latency   (batch)  <= 3,245 s
      - weighted_compute (mean) <= 23,898

    Round 1 achieved 20.55 % accuracy with a single Qwen3-4B branch
    (5 continues, ~20,866 WC) but fell short on accuracy.  Latency and
    WC had large margins (3,200 s and 3,032 WC respectively).

    Improvements over round 1
    -------------------------
    1. **Answer-history recovery** -- when latest_answer is null, scan
       the branch's answer_history for the most-common valid intermediate
       answer.  Zero extra compute cost; rescues episodes where the model
       found an answer at an earlier checkpoint then lost it.
    2. **Cheap-model fallback** -- when the entire primary history is
       null, spawn the cheapest available model.  With Qwen3-0.6B at
       weight 1.0 and 4 continues the cost is ~2,650 WC per fallback
       episode.  Even if every episode triggers the fallback the total
       20,866 + 2,650 = 23,516 < 23,898 budget.  The 0.6B model adds
       ~4-5 pp accuracy on the ~70 % of episodes with null primary.
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_size(name: str) -> float:
        """Parse approximate parameter count (billions) from model name.

        E.g. "Qwen3-4B" -> 4.0, "Qwen3-0.6B" -> 0.6.
        Returns inf when no size token is found.
        """
        upper = name.upper()
        best = float("inf")
        for i, ch in enumerate(upper):
            if ch != "B" or i == 0:
                continue
            j = i - 1
            while j >= 0 and (upper[j].isdigit() or upper[j] == "."):
                j -= 1
            num_str = upper[j + 1: i]
            if num_str:
                try:
                    val = float(num_str)
                    if val > 0:
                        best = min(best, val)
                except ValueError:
                    pass
        return best

    @staticmethod
    def _best_history_answer(branch: Dict[str, Any]) -> Optional[str]:
        """Most-common valid answer from answer_history, tie-broken by
        last appearance (most recent checkpoint preferred)."""
        history = branch.get("answer_history")
        if not history:
            return None
        valid_list: List[str] = []
        for ans in history:
            v = valid_answer(ans)
            if v is not None:
                valid_list.append(v)
        if not valid_list:
            return None
        counts: Counter = Counter(valid_list)
        top = max(counts.values())
        best: Optional[str] = None
        for ans in valid_list:
            if counts[ans] == top:
                best = ans  # keep overwriting -> last appearance wins
        return best

    def _run_branch(
        self, env: Any, model: str, max_continues: int
    ) -> Optional[str]:
        """Spawn one branch, advance it, return the best available answer."""
        branch = env.spawn(model)
        for _ in range(max_continues):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(branch["branch_id"])
        answer = valid_answer(branch.get("latest_answer"))
        if answer is None:
            answer = self._best_history_answer(branch)
        return answer

    def _plan(
        self, models: List[str], budget: float
    ) -> Tuple[str, int, Optional[str], int]:
        """Choose (primary_model, primary_continues,
                   fallback_model | None, fallback_continues).

        Selects the largest model whose single-branch full-depth cost
        fits within the budget while reserving room for a cheap fallback.
        """
        TOKENS = 530  # conservative tokens-per-step estimate

        # Parse sizes, sort ascending
        parsed: List[Tuple[str, float]] = []
        for m in models:
            s = self._model_size(m)
            if 0 < s < float("inf"):
                parsed.append((m, s))
        parsed.sort(key=lambda x: x[1])

        if not parsed:
            # Unparseable model names: single branch, guess weight 1
            return models[0], max(int(budget / TOKENS) - 1, 2), None, 0

        # Try each model as primary, largest first
        for model, size in reversed(parsed):
            weight = size / 0.6

            # Fallback: cheapest model strictly smaller than primary
            fallback: Optional[Tuple[str, float]] = None
            for fm, fs in parsed:
                if fm != model and fs < size:
                    fallback = (fm, fs)
                    break

            # Reserve budget for fallback at 5 steps (worst-case)
            if fallback is not None:
                fb_weight = fallback[1] / 0.6
                fb_reserve = 5 * TOKENS * fb_weight
            else:
                fb_reserve = 0.0

            primary_raw = (budget - fb_reserve) / weight
            primary_steps = int(primary_raw / TOKENS)
            primary_cont = primary_steps - 1  # spawn counts as one step

            if primary_cont < 3:
                continue  # insufficient depth, try smaller model

            primary_cont = min(primary_cont, 30)
            primary_cost = (primary_cont + 1) * TOKENS * weight

            # Compute actual fallback depth from remaining budget
            fb_model: Optional[str] = None
            fb_cont = 0
            if fallback is not None:
                fb_weight = fallback[1] / 0.6
                remaining = budget - primary_cost
                fb_steps = int(remaining / (TOKENS * fb_weight))
                if fb_steps >= 2:
                    fb_model = fallback[0]
                    fb_cont = min(fb_steps - 1, 10)

            return model, primary_cont, fb_model, fb_cont

        # No model achieves >= 4 steps; fall back to smallest model
        model, size = parsed[0]
        weight = size / 0.6
        steps = int(budget / (TOKENS * weight))
        cont = max(min(steps - 1, 30), 2)
        return model, cont, None, 0

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env: Any) -> Optional[str]:
        models = list(env.models)
        budget = float(env.requirements["weighted_compute"]["value"])

        primary, max_cont, fallback, fb_cont = self._plan(models, budget)

        # Phase 1: primary branch (largest affordable model, full depth)
        answer = self._run_branch(env, primary, max_cont)

        # Phase 2: cheap fallback when primary yielded no answer at all
        if answer is None and fallback is not None:
            answer = self._run_branch(env, fallback, fb_cont)

        return env.finish(answer)
