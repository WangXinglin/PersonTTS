from __future__ import annotations

from typing import Any, Dict, List, Optional

from policies.base import ControllerPolicy, deterministic_plurality, valid_answer


class CandidatePolicy(ControllerPolicy):
    """Latency-optimized parallel Qwen3-4B with plurality voting for profile 0058.

    Profile 0058: accuracy>=10.58%, latency batch<=57.66s, compute<=8.775M.

    Incumbent (round 2, utility 0.252) uses 7 branches with 7 max continues
    (8 sequential phases). Accuracy passes all batches (39.9% vs 10.58%
    threshold), but latency barely fails: p50=55.78s, p90=57.26s, p99=58.20s.
    Only 252/1000 batches satisfy batch<=57.66s because ~10% of episodes
    exceed the threshold, causing >3 of 60 problems per batch to violate.

    Fix: reduce max continues from 7 to 6 (7 phases instead of 8). This
    saves ~7s on high-latency episodes, bringing estimated batch to ~50s.
    Accuracy remains well above 10.58% (estimated ~35-38%) since most
    branches already have reliable answers by checkpoint 5. Compute stays
    trivially within budget (~170K vs 8.775M).
    """

    NAME = "candidate"

    def solve(self, env) -> Optional[str]:
        model = self._select_model(env)
        width = 7
        max_continues = 6  # Reduced from 7 to cut ~7s latency on high-latency episodes

        # Spawn parallel branches (reveals checkpoint 0) - 1 parallel phase
        branches = env.spawn_many(model, width)
        branch_ids = set(int(b["branch_id"]) for b in branches)

        # Continue all active branches in parallel - up to 6 parallel phases
        for _ in range(max_continues):
            state = env.observe()
            continuable = [
                int(b["branch_id"])
                for b in state["branches"]
                if int(b["branch_id"]) in branch_ids
                and not b["pruned"]
                and b["can_continue"]
            ]
            if not continuable:
                break
            env.continue_many(continuable)

        # Collect final answers and plurality vote
        state = env.observe()
        answers = self._collect_answers(state["branches"], branch_ids)
        answer = deterministic_plurality(answers) if answers else None
        return env.finish(answer)

    def _collect_answers(
        self, branches: List[Dict[str, Any]], branch_ids: set
    ) -> List[str]:
        """Gather valid answers from tracked branches."""
        answers: List[str] = []
        for b in branches:
            if int(b["branch_id"]) in branch_ids and not b["pruned"]:
                ans = valid_answer(b.get("latest_answer"))
                if ans is not None:
                    answers.append(ans)
        return answers

    def _select_model(self, env) -> str:
        """Pick Qwen3-4B for proven latency/accuracy characteristics."""
        models = list(env.models)
        if "Qwen3-4B" in models:
            return "Qwen3-4B"
        return min(models, key=self._model_size)

    @staticmethod
    def _model_size(name: str) -> float:
        """Extract approximate parameter count from model name."""
        parts = name.replace("-", " ").replace("_", " ").split()
        for part in parts:
            upper = part.upper()
            if upper.endswith("B"):
                try:
                    return float(upper[:-1])
                except ValueError:
                    continue
        return float("inf")
