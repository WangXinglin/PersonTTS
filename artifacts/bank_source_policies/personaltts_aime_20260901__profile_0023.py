from __future__ import annotations

from typing import Any, List, Optional

from policies.base import ControllerPolicy, valid_answer, deterministic_plurality


class CandidatePolicy(ControllerPolicy):
    """Adaptive parallel-probe policy tuned for profile_0023 constraints.

    Profile requires accuracy >= 0.44, latency batch <= 3992 s,
    weighted-compute mean <= 274 708.

    Strategy
    --------
    Use Qwen3-4B (per-branch cost ~132 k wc, ~464 s latency).
    Spawn 2 parallel branches and advance checkpoint-by-checkpoint.
    When both branches agree on the same non-null answer for two
    consecutive rounds, finish early (saves compute on easy problems
    and boosts accuracy because agreement implies near-certain
    correctness).  When they never agree, run to terminal and take
    deterministic plurality of the two terminal answers.

    Expected cost  ~140-200 k wc mean (well under 274 k).
    Expected batch latency  ~1 000 s (well under 3 992 s).
    """

    NAME = "candidate"
    AGREE_ROUNDS = 2
    WARMUP_CHECKPOINTS = 1

    def solve(self, env) -> Optional[str]:
        model = self.select_model(env.models)

        # --- Try spawning 2 parallel branches; fall back to 1 ------
        width = 2
        try:
            branches = env.spawn_many(model, width)
        except Exception:
            branches = [env.spawn(model)]
            width = 1

        if width == 1:
            return self.solve_single(env, branches[0])

        return self.solve_pair(env, branches)

    # ---------- two-branch parallel probe --------------------------
    def solve_pair(self, env, branches: List[Any]) -> Optional[str]:
        bid_to_idx = {branches[i]["branch_id"]: i for i in range(len(branches))}
        agree_streak = 0
        round_num = 0

        while True:
            a0 = valid_answer(branches[0].get("latest_answer"))
            a1 = valid_answer(branches[1].get("latest_answer"))

            if (
                round_num >= self.WARMUP_CHECKPOINTS
                and a0 is not None
                and a1 is not None
                and a0 == a1
            ):
                agree_streak += 1
                if agree_streak >= self.AGREE_ROUNDS:
                    return env.finish(a0)
            else:
                agree_streak = 0

            continuable_ids = [
                branches[i]["branch_id"]
                for i in range(len(branches))
                if branches[i]["can_continue"]
            ]
            if not continuable_ids:
                break

            try:
                updated = env.continue_many(continuable_ids)
            except Exception:
                break

            for u in updated:
                idx = bid_to_idx[u["branch_id"]]
                branches[idx] = u

            round_num += 1

        # --- Terminal: deterministic plurality of latest answers ----
        votes: List[str] = []
        for b in branches:
            ans = valid_answer(b.get("latest_answer"))
            if ans is not None:
                votes.append(ans)

        answer = deterministic_plurality(votes) if votes else None
        return env.finish(answer)

    # ---------- single-branch fallback -----------------------------
    @staticmethod
    def solve_single(env, branch: Any) -> Optional[str]:
        while branch["can_continue"]:
            branch = env.continue_branch(branch["branch_id"])
        answer = valid_answer(branch.get("latest_answer"))
        return env.finish(answer)

    # ---------- model selection ------------------------------------
    @staticmethod
    def select_model(models: Any) -> str:
        """Pick Qwen3-4B if available; otherwise largest model <= 8 B."""
        if "Qwen3-4B" in models:
            return "Qwen3-4B"

        def param_size(name: str) -> float:
            for segment in name.replace("_", "-").split("-"):
                if segment.upper().endswith("B") and len(segment) > 1:
                    try:
                        return float(segment[:-1])
                    except ValueError:
                        pass
            return 0.0

        affordable = [m for m in models if 0 < param_size(m) <= 8.0]
        if affordable:
            return max(affordable, key=param_size)
        return min(
            models,
            key=lambda m: param_size(m) if param_size(m) > 0 else float("inf"),
        )
