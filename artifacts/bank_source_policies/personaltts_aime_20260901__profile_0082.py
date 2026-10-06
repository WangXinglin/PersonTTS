from __future__ import annotations

from typing import Any, Dict, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Parallel-spawn low-latency policy for latency-constrained profiles.

    The baseline runs sequential spawn + continue_branch, giving
    O(width * depth) latency that far exceeds the 382 s batch cap.

    This policy uses spawn_many / continue_many so every cohort action
    charges only the max-of-cohort latency, collapsing total latency to
    O(depth).  A width-5 parallel cohort with 4 continue rounds (5 total
    latency phases) keeps the per-problem latency well under the cap while
    preserving ample accuracy via plurality voting (requirement is only
    ~14 %).
    """

    NAME = "candidate"

    def _select_model(self, env: Any) -> str:
        """Return the smallest available model for minimum per-step latency."""
        models = env.models
        # Prefer known small models (smallest first)
        for name in ("Qwen3-4B", "Qwen3-1.8B", "Qwen3-0.6B"):
            if name in models:
                return name
        return sorted(models)[0]

    def solve(self, env: Any) -> Optional[str]:
        model = self._select_model(env)
        width = 5

        # --- parallel spawn: 1 latency phase instead of 'width' sequential phases ---
        try:
            env.spawn_many(model, width)
        except Exception:
            # If parallel spawn fails (e.g. pool too small), fall back to minimal single-branch
            return self._solve_single(env, model)

        # --- parallel continues: each round is 1 latency phase (max of cohort) ---
        max_continue_rounds = 4
        for _ in range(max_continue_rounds):
            state = env.observe()
            continuable = [
                branch["branch_id"]
                for branch in state["branches"]
                if branch["can_continue"] and not branch["pruned"]
            ]
            if not continuable:
                break
            try:
                env.continue_many(continuable)
            except Exception:
                break

        # Gather answers and pick plurality
        state = env.observe()
        answers = []
        for branch in state["branches"]:
            if not branch["pruned"]:
                ans = valid_answer(branch.get("latest_answer"))
                if ans is not None:
                    answers.append(ans)
        chosen = deterministic_plurality(answers) if answers else None
        return env.finish(chosen)

    # ---- fallback for edge cases where spawn_many is unavailable ----

    def _solve_single(self, env: Any, model: str) -> Optional[str]:
        """Single-branch sequential fallback (minimal latency footprint)."""
        branch = env.spawn(model)
        for _ in range(4):
            if not branch["can_continue"]:
                break
            branch = env.continue_branch(branch["branch_id"])
        ans = valid_answer(branch.get("latest_answer"))
        return env.finish(ans)
