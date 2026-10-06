from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import (
    ControllerPolicy,
    ResourceProfile,
    asc_posterior_confidence,
    deterministic_plurality,
    top_two_counts,
    unique_mode,
    valid_answer,
)


@dataclass(frozen=True)
class CandidateResult:
    branch_id: int
    answer: Optional[str]
    max_stage_reached: int
    truncated_before_refine: bool


class CompleteCandidateRunner:
    """Run one branch under fixed depth/refine limits and return one final vote."""

    def __init__(self, env, profile: ResourceProfile):
        self.env = env
        self.profile = profile

    def _run_stage(self, branch: Dict[str, Any]) -> Dict[str, Any]:
        while (
            branch["can_continue"]
            and branch["checkpoint_index"] + 1 < self.profile.checkpoints_per_stage
        ):
            branch = self.env.continue_branch(branch["branch_id"])
        return branch

    def run(self) -> CandidateResult:
        branch = self._run_stage(self.env.spawn(self.profile.model_id))
        truncated = False
        for _ in range(self.profile.max_refine_rounds):
            if not branch["can_refine"]:
                truncated = bool(branch["can_continue"])
                break
            branch = self.env.refine(branch["branch_id"])
            branch = self._run_stage(branch)
        return CandidateResult(
            branch_id=int(branch["branch_id"]),
            answer=valid_answer(branch.get("latest_answer")),
            max_stage_reached=int(branch["stage_index"]),
            truncated_before_refine=truncated,
        )


class _ConfiguredBaseline(ControllerPolicy):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.resources = ResourceProfile.from_config(self.config)

    def _validate_environment(self, env) -> None:
        if self.resources.model_id not in env.models:
            raise ValueError(f"configured model is unavailable: {self.resources.model_id}")

    def _remaining(self, env) -> int:
        return int(env.observe()["remaining_by_model"].get(self.resources.model_id, 0))

    def _can_run_candidates(self, env, count: int) -> bool:
        checker = getattr(env, "can_afford_complete_candidates", None)
        if checker is None:
            return self._remaining(env) >= count
        return bool(
            checker(
                self.resources.model_id,
                count=count,
                checkpoints_per_stage=self.resources.checkpoints_per_stage,
                max_refine_rounds=self.resources.max_refine_rounds,
            )
        )

    def _metrics(self, env, stop_reason: str, **extra: Any) -> Dict[str, Any]:
        state = env.observe()
        branches = state["branches"]
        actions = [event.get("action") for event in env.trace]
        result: Dict[str, Any] = {
            "algorithm": self.NAME,
            "configured_model": self.resources.model_id,
            "configured_max_width": self.resources.max_width,
            "configured_checkpoints_per_stage": self.resources.checkpoints_per_stage,
            "configured_max_refine_rounds": self.resources.max_refine_rounds,
            "realized_width": actions.count("SPAWN"),
            "final_active_width": sum(not branch["pruned"] for branch in branches),
            "max_stage_reached": max(
                (int(branch["stage_index"]) for branch in branches), default=0
            ),
            "spawn_count": actions.count("SPAWN"),
            "continue_action_count": actions.count("CONTINUE"),
            "prune_count": actions.count("PRUNE"),
            "refine_count": actions.count("REFINE"),
            "stop_reason": stop_reason,
            "cost": env.cost,
        }
        result.update(extra)
        return result

    def _finish(self, env, answer: Optional[str], stop_reason: str, **extra: Any) -> Optional[str]:
        final_answer = env.finish(answer)
        self.last_result = self._metrics(
            env, stop_reason, final_answer=final_answer, **extra
        )
        return final_answer


class ASCConfigured4D(_ConfiguredBaseline):
    """AutoTTS ASC over complete, single-model candidates with fixed depth/stages."""

    NAME = "asc_configured_4d"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        asc = self.config.get("asc", {})
        self.confidence_threshold = float(asc.get("confidence_threshold", 0.95))
        self.min_samples = int(asc.get("min_samples", 2))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("ASC confidence_threshold must be in [0, 1]")
        if self.min_samples < 1:
            raise ValueError("ASC min_samples must be >= 1")

    def solve(self, env) -> Optional[str]:
        self._validate_environment(env)
        runner = CompleteCandidateRunner(env, self.resources)
        votes: list[str] = []
        spawned = 0
        invalid_answers = 0
        truncated_candidates = 0
        top1 = top2 = 0
        confidence = 0.0

        while spawned < self.resources.max_width:
            if not self._can_run_candidates(env, 1):
                reason = "pool_exhausted" if self._remaining(env) < 1 else "budget_exhausted"
                break
            result = runner.run()
            spawned += 1
            truncated_candidates += int(result.truncated_before_refine)
            if result.answer is None:
                invalid_answers += 1
                continue
            votes.append(result.answer)
            winner, top1, top2, tied = top_two_counts(votes)
            if tied:
                confidence = 0.0
                continue
            confidence = asc_posterior_confidence(top1, top2)
            if len(votes) < self.min_samples:
                continue
            if confidence >= self.confidence_threshold:
                return self._finish(
                    env,
                    winner,
                    "asc_confidence",
                    valid_vote_count=len(votes),
                    invalid_answer_count=invalid_answers,
                    samples_at_stop=spawned,
                    top1_count=top1,
                    top2_count=top2,
                    asc_posterior_confidence=confidence,
                    truncated_candidates=truncated_candidates,
                )
        else:
            reason = "max_width"

        final_answer = deterministic_plurality(votes)
        if not votes and reason not in {"budget_exhausted", "pool_exhausted"}:
            reason = "no_valid_answer"
        return self._finish(
            env,
            final_answer,
            reason,
            valid_vote_count=len(votes),
            invalid_answer_count=invalid_answers,
            samples_at_stop=spawned,
            top1_count=top1,
            top2_count=top2,
            asc_posterior_confidence=confidence,
            truncated_candidates=truncated_candidates,
        )


class ESCConfigured4D(_ConfiguredBaseline):
    """AutoTTS ESC over non-overlapping windows of complete candidates."""

    NAME = "esc_configured_4d"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        esc = self.config.get("esc", {})
        self.window_size = int(esc.get("window_size", 5))
        if self.window_size < 2:
            raise ValueError("ESC window_size must be >= 2")
        if self.resources.max_width < self.window_size:
            raise ValueError("ESC max_width must be >= window_size")
        if self.resources.max_width % self.window_size:
            raise ValueError("ESC max_width must be divisible by window_size")

    def solve(self, env) -> Optional[str]:
        self._validate_environment(env)
        runner = CompleteCandidateRunner(env, self.resources)
        all_votes: list[str] = []
        spawned = completed_windows = invalid_answers = truncated_candidates = 0
        last_window_answers: list[Optional[str]] = []
        unanimous_window_index: Optional[int] = None

        while spawned < self.resources.max_width:
            if not self._can_run_candidates(env, self.window_size):
                reason = (
                    "pool_exhausted"
                    if self._remaining(env) < self.window_size
                    else "budget_exhausted"
                )
                break
            window = [runner.run() for _ in range(self.window_size)]
            spawned += self.window_size
            completed_windows += 1
            last_window_answers = [item.answer for item in window]
            invalid_answers += sum(answer is None for answer in last_window_answers)
            truncated_candidates += sum(item.truncated_before_refine for item in window)
            valid_window = [answer for answer in last_window_answers if answer is not None]
            all_votes.extend(valid_window)
            if (
                len(valid_window) == self.window_size
                and len(set(valid_window)) == 1
            ):
                unanimous_window_index = completed_windows - 1
                return self._finish(
                    env,
                    valid_window[0],
                    "esc_unanimous_window",
                    valid_vote_count=len(all_votes),
                    invalid_answer_count=invalid_answers,
                    completed_windows=completed_windows,
                    window_size=self.window_size,
                    last_window_answers=last_window_answers,
                    unanimous_window_index=unanimous_window_index,
                    truncated_candidates=truncated_candidates,
                )
        else:
            reason = "max_width"

        final_answer = deterministic_plurality(all_votes)
        if not all_votes and reason not in {"budget_exhausted", "pool_exhausted"}:
            reason = "no_valid_answer"
        return self._finish(
            env,
            final_answer,
            reason,
            valid_vote_count=len(all_votes),
            invalid_answer_count=invalid_answers,
            completed_windows=completed_windows,
            window_size=self.window_size,
            last_window_answers=last_window_answers,
            unanimous_window_index=unanimous_window_index,
            truncated_candidates=truncated_candidates,
        )


class ParallelProbeSR(_ConfiguredBaseline):
    """Stage-aware AutoTTS Parallel-Probe with synchronous checkpoint rounds."""

    NAME = "parallel_probe_sr"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        settings = self.config.get("parallel_probe", {})
        self.warmup_checkpoints = int(settings.get("warmup_checkpoints", 10))
        self.stable_window = int(settings.get("stable_window", 14))
        self.prune_patience = int(settings.get("prune_patience", 8))
        if self.warmup_checkpoints < 0:
            raise ValueError("warmup_checkpoints must be >= 0")
        if self.stable_window < 1:
            raise ValueError("stable_window must be >= 1")
        if self.prune_patience < 1:
            raise ValueError("prune_patience must be >= 1")

    def solve(self, env) -> Optional[str]:
        self._validate_environment(env)
        spawner = getattr(env, "spawn_many", None)
        if spawner is None:
            raise RuntimeError("ParallelProbeSR requires ReplayEnvironment.spawn_many")
        try:
            opened = spawner(self.resources.model_id, self.resources.max_width)
        except Exception as exc:
            if exc.__class__.__name__ != "InvalidAction":
                raise
            return self._finish(env, None, "insufficient_spawn_budget")

        branch_ids = [int(branch["branch_id"]) for branch in opened]
        pruned: set[int] = set()
        last_valid: Dict[int, Optional[str]] = {
            int(branch["branch_id"]): valid_answer(branch.get("latest_answer"))
            for branch in opened
        }
        current: Dict[int, Optional[str]] = dict(last_valid)
        off_track = {branch_id: 0 for branch_id in branch_ids}
        snapshots: list[Dict[str, Any]] = []
        checkpoint_round_count = 0
        stop_reason = "stage_or_depth_limit"

        def survivors() -> list[Dict[str, Any]]:
            return [
                branch
                for branch in env.observe()["branches"]
                if int(branch["branch_id"]) in branch_ids and not branch["pruned"]
            ]

        for stage_index in range(self.resources.max_stages):
            winner_history: list[str] = []
            for branch in survivors():
                branch_id = int(branch["branch_id"])
                off_track[branch_id] = 0
                if stage_index > 0:
                    current[branch_id] = None

            for checkpoint_round in range(self.resources.checkpoints_per_stage):
                active = survivors()
                if not active:
                    break
                if not (stage_index == 0 and checkpoint_round == 0):
                    advancing = [branch for branch in active if branch["can_continue"]]
                    if not advancing:
                        break
                    try:
                        updated = env.continue_many(
                            [int(branch["branch_id"]) for branch in advancing]
                        )
                    except Exception as exc:
                        if exc.__class__.__name__ != "InvalidAction":
                            raise
                        stop_reason = "budget_exhausted"
                        active = []
                        break
                    for branch in updated:
                        branch_id = int(branch["branch_id"])
                        answer = valid_answer(branch.get("latest_answer"))
                        current[branch_id] = answer
                        if answer is not None:
                            last_valid[branch_id] = answer

                active = survivors()
                checkpoint_round_count += 1
                answers = [
                    current[int(branch["branch_id"])]
                    for branch in active
                    if current[int(branch["branch_id"])] is not None
                ]
                winner = unique_mode(answers)
                eligible = checkpoint_round >= self.warmup_checkpoints

                if not eligible:
                    winner_history.clear()
                elif winner is None:
                    winner_history.clear()
                    for branch in active:
                        off_track[int(branch["branch_id"])] = 0
                else:
                    if winner_history and winner_history[-1] == winner:
                        winner_history.append(winner)
                    else:
                        winner_history = [winner]
                    for branch in active:
                        branch_id = int(branch["branch_id"])
                        answer = current[branch_id]
                        off_track[branch_id] = (
                            off_track[branch_id] + 1
                            if answer is not None and answer != winner
                            else 0
                        )

                snapshots.append(
                    {
                        "stage_index": stage_index,
                        "checkpoint_index": checkpoint_round,
                        "winner": winner,
                        "winner_streak": len(winner_history),
                        "active_width": len(active),
                    }
                )

                if eligible and winner is not None and len(winner_history) >= self.stable_window:
                    return self._finish_parallel(
                        env,
                        winner,
                        "parallel_probe_stable_winner",
                        snapshots,
                        off_track,
                        pruned,
                        checkpoint_round_count,
                        current,
                    )

                if eligible and winner is not None:
                    for branch in active:
                        branch_id = int(branch["branch_id"])
                        if off_track[branch_id] >= self.prune_patience:
                            env.prune(branch_id)
                            pruned.add(branch_id)

            if stop_reason == "budget_exhausted" or stage_index >= self.resources.max_refine_rounds:
                break
            active = survivors()
            refinable = [branch for branch in active if branch["can_refine"]]
            for branch in active:
                if not branch["can_refine"]:
                    branch_id = int(branch["branch_id"])
                    env.prune(branch_id)
                    pruned.add(branch_id)
            if not refinable:
                break
            try:
                env.refine_many([int(branch["branch_id"]) for branch in refinable])
            except Exception as exc:
                if exc.__class__.__name__ != "InvalidAction":
                    raise
                stop_reason = "budget_exhausted"
                break

        active = survivors()
        votes = [
            current[int(branch["branch_id"])]
            for branch in active
            if current[int(branch["branch_id"])] is not None
        ]
        if not votes:
            votes = [
                last_valid[int(branch["branch_id"])]
                for branch in active
                if last_valid[int(branch["branch_id"])] is not None
            ]
        answer = deterministic_plurality(votes)
        if answer is None:
            stop_reason = "no_valid_answer"
        return self._finish_parallel(
            env,
            answer,
            stop_reason,
            snapshots,
            off_track,
            pruned,
            checkpoint_round_count,
            current,
        )

    def _finish_parallel(
        self,
        env,
        answer: Optional[str],
        stop_reason: str,
        snapshots: list[Dict[str, Any]],
        off_track: Dict[int, int],
        pruned: set[int],
        checkpoint_round_count: int,
        current: Dict[int, Optional[str]],
    ) -> Optional[str]:
        survivor_ids = [
            int(branch["branch_id"])
            for branch in env.observe()["branches"]
            if not branch["pruned"]
        ]
        return self._finish(
            env,
            answer,
            stop_reason,
            checkpoint_round_count=checkpoint_round_count,
            branch_checkpoint_step_count=sum(
                event.get("action") in {"SPAWN", "CONTINUE"} for event in env.trace
            ),
            valid_vote_count=sum(current.get(branch_id) is not None for branch_id in survivor_ids),
            invalid_answer_count=sum(current.get(branch_id) is None for branch_id in survivor_ids),
            winner_by_stage_checkpoint=snapshots,
            active_width_by_stage_checkpoint=[item["active_width"] for item in snapshots],
            branch_off_track_streak={str(key): value for key, value in off_track.items()},
            pruned_branch_ids=sorted(pruned),
            survivor_branch_ids=survivor_ids,
        )


class FixedBudgetGridPolicy(_ConfiguredBaseline):
    """Calibration-only policy that exhausts one explicit four-dimensional budget."""

    NAME = "fixed_budget_grid_calibration"

    def solve(self, env) -> Optional[str]:
        self._validate_environment(env)
        spawner = getattr(env, "spawn_many", None)
        continuer = getattr(env, "continue_many", None)
        refiner = getattr(env, "refine_many", None)
        if spawner is None or continuer is None or refiner is None:
            raise RuntimeError(
                "FixedBudgetGridPolicy requires replay cohort actions"
            )

        opened = spawner(self.resources.model_id, self.resources.max_width)
        branch_ids = [int(branch["branch_id"]) for branch in opened]
        active_ids = list(branch_ids)
        completed_refine_cohorts = 0

        for stage_index in range(self.resources.max_stages):
            # SPAWN already reveals checkpoint zero of the initial stage.
            start_checkpoint = 1 if stage_index == 0 else 0
            for _ in range(start_checkpoint, self.resources.checkpoints_per_stage):
                by_id = {
                    int(branch["branch_id"]): branch
                    for branch in env.observe()["branches"]
                }
                advancing = [
                    branch_id
                    for branch_id in active_ids
                    if by_id[branch_id]["can_continue"]
                ]
                if not advancing:
                    break
                continuer(advancing)

            if stage_index >= self.resources.max_refine_rounds:
                break
            by_id = {
                int(branch["branch_id"]): branch
                for branch in env.observe()["branches"]
            }
            refinable = [
                branch_id
                for branch_id in active_ids
                if by_id[branch_id]["can_refine"]
            ]
            if not refinable:
                break
            refiner(refinable)
            active_ids = refinable
            completed_refine_cohorts += 1

        by_id = {
            int(branch["branch_id"]): branch
            for branch in env.observe()["branches"]
        }
        votes = [
            answer
            for branch_id in branch_ids
            if (answer := valid_answer(by_id[branch_id].get("latest_answer")))
            is not None
        ]
        return self._finish(
            env,
            deterministic_plurality(votes),
            "fixed_budget_complete",
            valid_vote_count=len(votes),
            invalid_answer_count=len(branch_ids) - len(votes),
            completed_refine_cohorts=completed_refine_cohorts,
            final_voting_width=len(branch_ids),
        )


BASELINE_REGISTRY = {
    ASCConfigured4D.NAME: ASCConfigured4D,
    ESCConfigured4D.NAME: ESCConfigured4D,
    ParallelProbeSR.NAME: ParallelProbeSR,
}
