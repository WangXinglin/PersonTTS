from __future__ import annotations

from typing import Optional

from policies.base import ControllerPolicy, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Multi-branch first-valid-answer policy with adaptive model selection.

    Improvements over the incumbent (single Qwen3-0.6B spawn):

    1. **Model selection**: pick the largest available model up to 2.5 B
       parameters.  A medium model (e.g. Qwen3-1.7B) may produce
       checkpoint-0 answers with significantly higher accuracy than
       Qwen3-0.6B while keeping single-phase latency under 6.18 s.
       Models >= 3 B are avoided because baseline evidence shows
       Qwen3-4B returns null at checkpoint 0.

    2. **Parallel multi-branch spawn**: spawn_many with width 5.  This
       virtually eliminates the ~15 % null rate at checkpoint 0 for
       Qwen3-0.6B (P(all 5 null) < 0.01 %).  Each branch is an
       independent sample; using the *first valid answer* (not
       plurality voting) avoids the failure mode of round 2 where
       correlated wrong-answer majorities destroyed the signal.

    Latency: a single spawn_many phase costs the max-of-N latency,
    roughly the same as a single spawn (~4.8-5.2 s), well within the
    6.18 s batch budget.

    Weighted compute: 5 branches of Qwen3-0.6B cost ~2 600 per
    episode (60 problems -> ~156 000 mean), far below the 20.6 M
    budget.  Even a 2 B model with 5 branches costs < 50 000 per
    episode.
    """

    NAME = "candidate"

    # Maximum model size (in billions) to use with the checkpoint-0-only
    # strategy.  Baseline traces confirm Qwen3-4B always returns null at
    # checkpoint 0, so we exclude models >= 3 B.
    MAX_MODEL_SIZE_B = 2.5

    # Number of parallel branches.  Eliminates most null-answer episodes
    # while keeping latency to a single parallel phase.
    BRANCH_WIDTH = 5

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_size(name: str) -> float:
        """Parse the parameter-count number preceding a 'B' suffix.

        Handles names such as ``Qwen3-4B``, ``Qwen3-0.6B``, ``Llama-70B``.
        Returns 999.0 when no parseable size is found so unknown models
        are ranked last.
        """
        upper = name.upper()
        best = 999.0
        i = 0
        while i < len(upper):
            if upper[i] == "B" and i > 0:
                j = i - 1
                while j >= 0 and (upper[j].isdigit() or upper[j] == "."):
                    j -= 1
                num_str = upper[j + 1 : i]
                if num_str and num_str != ".":
                    try:
                        val = float(num_str)
                        if val < best:
                            best = val
                    except ValueError:
                        pass
            i += 1
        return best

    def select_model(self, models: list[str]) -> str:
        """Pick the largest model <= MAX_MODEL_SIZE_B.

        Falls back to the smallest available model if nothing qualifies.
        Ties broken alphabetically for determinism.
        """
        best_name: Optional[str] = None
        best_size = -1.0
        fallback_name: Optional[str] = None
        fallback_size = 1e18

        for name in models:
            size = self.extract_size(name)
            if size <= self.MAX_MODEL_SIZE_B:
                if size > best_size or (
                    size == best_size
                    and (best_name is None or name < best_name)
                ):
                    best_size = size
                    best_name = name
            if size < fallback_size or (
                size == fallback_size
                and (fallback_name is None or name < fallback_name)
            ):
                fallback_size = size
                fallback_name = name

        if best_name is not None:
            return best_name
        if fallback_name is not None:
            return fallback_name
        return models[0]

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, env) -> Optional[str]:
        model = self.select_model(list(env.models))

        # Determine affordable width from the model pool.
        state = env.observe()
        pool = int(state["remaining_by_model"].get(model, 0))
        width = min(self.BRANCH_WIDTH, pool)

        if width < 1:
            return env.finish()

        # Spawn branches in one parallel phase.
        branches = None
        if width >= 2:
            try:
                branches = env.spawn_many(model, width)
            except Exception:
                branches = None

        if branches is None:
            try:
                branches = [env.spawn(model)]
            except Exception:
                return env.finish()

        # Pick the first valid (non-null) answer.
        # Avoids plurality voting which destroys signal when per-branch
        # accuracy is low (round 2 lesson: 100-branch vote -> 0 %).
        for branch in branches:
            answer = valid_answer(branch.get("latest_answer"))
            if answer is not None:
                return env.finish(answer)

        # All branches returned null -- submit without an answer.
        return env.finish()
