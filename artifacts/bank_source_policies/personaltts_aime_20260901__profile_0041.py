from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from policies.base import (
    ControllerPolicy,
    deterministic_plurality,
    valid_answer,
)


class CandidatePolicy(ControllerPolicy):
    """Parallel majority-vote policy tuned for tight compute/latency budgets.

    Strategy:
    - Spawn a small number of branches in parallel using spawn_many.
      Parallel spawning means latency = max(branch_latencies) per phase,
      not sum, which dramatically cuts tail latency.
    - Continue all branches in parallel (continue_many) until terminal.
    - Early-stop as soon as an irrevocable majority forms among terminal
      branches (saves the remaining running branch's compute).
    - Take deterministic plurality for the final answer.

    This targets profiles where the accuracy threshold is moderate (allowing
    fewer branches) but latency and compute ceilings are tight (requiring
    parallel execution and limited width).
    """

    NAME = "candidate"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _pick_model(self, env: Any) -> str:
        """Choose the cheapest available model (smallest parameter count)."""
        models = list(env.models)
        if not models:
            raise ValueError("no models available")
        if "Qwen3-4B" in models:
            return "Qwen3-4B"
        best = models[0]
        best_size = self._model_param_size(best)
        for m in models[1:]:
            s = self._model_param_size(m)
            if s < best_size or (s == best_size and m < best):
                best = m
                best_size = s
        return best

    @staticmethod
    def _model_param_size(name: str) -> float:
        """Extract a numeric parameter-count proxy from a model name."""
        size = float("inf")
        i = 0
        n = len(name)
        while i < n:
            if name[i].isdigit() or (name[i] == "." and i + 1 < n and name[i + 1].isdigit()):
                j = i
                while j < n and (name[j].isdigit() or name[j] == "."):
                    j += 1
                try:
                    val = float(name[i:j])
                except ValueError:
                    i = j
                    continue
                if j < n and name[j] in ("B", "b"):
                    size = min(size, val)
                i = j
            else:
                i += 1
        return size

    @staticmethod
    def _irrevocable_majority(
        terminal_answers: Dict[int, Optional[str]], total_width: int
    ) -> Optional[str]:
        """Return the majority answer if it can no longer be overturned."""
        valid = [a for a in terminal_answers.values() if a is not None]
        if not valid:
            return None
        counts = Counter(valid)
        best_answer, best_count = counts.most_common(1)[0]
        remaining_active = total_width - len(terminal_answers)
        if best_count > remaining_active:
            return best_answer
        return None

    # ------------------------------------------------------------------
    # main solve
    # ------------------------------------------------------------------

    def solve(self, env: Any) -> Optional[str]:
        model = self._pick_model(env)

        # 3 parallel branches: good majority-vote boost while keeping
        # compute well under budget for this profile.
        width = 3

        # --- Spawn all branches in one parallel phase ---
        try:
            opened: List[Dict[str, Any]] = env.spawn_many(model, width)
        except Exception:
            # Fall back to a single branch if pool is too small
            try:
                opened = [env.spawn(model)]
                width = 1
            except Exception:
                return env.finish(None)

        branch_ids: List[int] = [b["branch_id"] for b in opened]
        terminal_answers: Dict[int, Optional[str]] = {}

        # Record any branches already terminal at checkpoint 0
        for b in opened:
            if not b["can_continue"]:
                terminal_answers[b["branch_id"]] = valid_answer(
                    b.get("latest_answer")
                )

        # --- Continue in parallel rounds until all terminal or majority ---
        max_rounds = 500
        for _ in range(max_rounds):
            # Check for irrevocable majority among terminal branches
            winner = self._irrevocable_majority(terminal_answers, width)
            if winner is not None:
                return env.finish(winner)

            # Advance all branches that can still continue
            state = env.observe()
            by_id: Dict[int, Dict[str, Any]] = {
                b["branch_id"]: b for b in state["branches"]
            }
            active_ids = [
                bid for bid in branch_ids if by_id[bid]["can_continue"]
            ]
            if not active_ids:
                break

            try:
                updated = env.continue_many(active_ids)
            except Exception:
                # Budget exhausted - stop and vote with what we have
                break

            # Record newly terminal branches
            for b in updated:
                bid = b["branch_id"]
                if not b["can_continue"] and bid not in terminal_answers:
                    terminal_answers[bid] = valid_answer(
                        b.get("latest_answer")
                    )

        # --- Final answer ---
        winner = self._irrevocable_majority(terminal_answers, width)
        if winner is not None:
            return env.finish(winner)

        # Fallback: deterministic plurality of all available answers
        state = env.observe()
        votes: List[str] = []
        for b in state["branches"]:
            a = valid_answer(b.get("latest_answer"))
            if a is not None:
                votes.append(a)
        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)
