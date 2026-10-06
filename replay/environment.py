from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .cost_model import CostLedger, ParameterWeightedCost
from .schema import Branch, ReplayProblem


class InvalidAction(RuntimeError):
    pass


@dataclass
class RuntimeBranch:
    runtime_id: int
    model: str
    source: Branch
    stage_index: int = 0
    checkpoint_index: int = -1
    latest_answer: Optional[str] = None
    answer_history: List[str] = field(default_factory=list)
    pruned: bool = False

    @property
    def stage(self):
        return self.source.stages[self.stage_index]

    @property
    def stage_terminal(self) -> bool:
        return bool(
            self.stage.terminal_reason is not None
            and self.checkpoint_index == len(self.stage.checkpoints) - 1
        )

    @property
    def can_continue(self) -> bool:
        return not self.pruned and self.checkpoint_index + 1 < len(self.stage.checkpoints)

    @property
    def can_refine(self) -> bool:
        return (
            not self.pruned
            and self.stage_terminal
            and self.stage_index + 1 < len(self.source.stages)
        )


def _stable_seed(*parts: Any) -> int:
    text = "\u241f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _default_normalizer(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.upper() in {"ERR", "PARSE_ERROR"} else text


class ReplayEnvironment:
    """Causal replay environment with atomic batch actions for standard baselines."""

    def __init__(
        self,
        problem: ReplayProblem,
        model_weights: Dict[str, float],
        *,
        seed: int,
        beta: Optional[float] = None,
        requirements: Optional[Dict[str, Any]] = None,
        max_weighted_compute: Optional[float] = None,
        answer_normalizer: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> None:
        self._problem = problem
        self._cost_model = ParameterWeightedCost(model_weights)
        self._pools: Dict[str, List[Branch]] = {}
        for model, branches in problem.models.items():
            pool = list(branches)
            random.Random(_stable_seed(seed, problem.dataset, problem.index, model)).shuffle(pool)
            self._pools[model] = pool
        self._pool_offsets = {model: 0 for model in self._pools}
        self._branches: Dict[int, RuntimeBranch] = {}
        self._next_runtime_id = 0
        self._ledger = CostLedger()
        self._trace: List[Dict[str, Any]] = []
        self._finished = False
        self._answer_normalizer = answer_normalizer or _default_normalizer
        self.seed = int(seed)
        if beta is not None:
            self.beta = max(0.0, min(1.0, float(beta)))
        self._requirements = deepcopy(requirements) if requirements is not None else None
        self.max_weighted_compute = max_weighted_compute

    @property
    def models(self) -> List[str]:
        return list(self._pools)

    @property
    def cost(self) -> Dict[str, float | int]:
        return self._ledger.to_dict()

    @property
    def trace(self) -> List[Dict[str, Any]]:
        return list(self._trace)

    @property
    def requirements(self) -> Optional[Dict[str, Any]]:
        """Return an isolated public copy of the current user's hard requirements."""
        return deepcopy(self._requirements)

    def _normalize(self, value: Any) -> Optional[str]:
        normalized = self._answer_normalizer(value)
        return None if normalized is None else str(normalized)

    def _fits(self, additional_weighted_compute: float) -> bool:
        return self.max_weighted_compute is None or (
            self._ledger.weighted_compute + additional_weighted_compute
            <= self.max_weighted_compute
        )

    def _charge(
        self,
        model: str,
        *,
        think: int = 0,
        probe: int = 0,
        feedback: int = 0,
        latency_s: float = 0.0,
    ) -> None:
        raw = think + probe + feedback
        weighted = self._cost_model.weighted(model, raw)
        if not self._fits(weighted):
            raise InvalidAction("Action exceeds environment weighted-compute budget")
        self._ledger.think_tokens += think
        self._ledger.probe_tokens += probe
        self._ledger.feedback_tokens += feedback
        self._ledger.weighted_compute += weighted
        self._ledger.latency_s += latency_s

    def _charge_parallel_latency(self, latencies: List[float]) -> None:
        if latencies:
            self._ledger.latency_s += max(latencies)

    def _require_running(self) -> None:
        if self._finished:
            raise InvalidAction("Episode is already finished")

    def _public_branch(self, branch: RuntimeBranch) -> Dict[str, Any]:
        return {
            "branch_id": branch.runtime_id,
            "model": branch.model,
            "stage_index": branch.stage_index,
            "stage_kind": branch.stage.kind,
            "checkpoint_index": branch.checkpoint_index,
            "latest_answer": branch.latest_answer,
            "answer_history": list(branch.answer_history),
            "stage_terminal": branch.stage_terminal,
            "can_continue": branch.can_continue,
            "can_refine": branch.can_refine,
            "pruned": branch.pruned,
        }

    def observe(self) -> Dict[str, Any]:
        state = {
            "requirements": self.requirements,
            "models": self.models,
            "remaining_by_model": {
                model: len(pool) - self._pool_offsets[model] for model, pool in self._pools.items()
            },
            "branches": [self._public_branch(self._branches[key]) for key in sorted(self._branches)],
            "cost": self.cost,
        }
        if hasattr(self, "beta"):
            state["beta"] = self.beta
        return state

    def _advance(self, branch: RuntimeBranch, *, charge_latency: bool = True) -> Optional[str]:
        if not branch.can_continue:
            raise InvalidAction(f"Branch {branch.runtime_id} cannot continue")
        checkpoint = branch.stage.checkpoints[branch.checkpoint_index + 1]
        self._charge(
            branch.model,
            think=checkpoint.think_tokens,
            probe=checkpoint.probe_tokens,
            latency_s=(
                checkpoint.think_elapsed_s + checkpoint.probe_elapsed_s
                if charge_latency
                else 0.0
            ),
        )
        branch.checkpoint_index += 1
        branch.latest_answer = self._normalize(checkpoint.answer)
        branch.answer_history.append(branch.latest_answer or "")
        return branch.latest_answer

    def _append_action(self, action: str, branch: RuntimeBranch, **extra: Any) -> None:
        event = {
            "step": len(self._trace),
            "action": action,
            "model": branch.model,
            "branch_id": branch.runtime_id,
            "observation": self._public_branch(branch),
            "cost": self.cost,
        }
        event.update(extra)
        self._trace.append(event)

    def _spawn(self, model: str, *, charge_latency: bool) -> Dict[str, Any]:
        self._require_running()
        if model not in self._pools:
            raise InvalidAction(f"Unknown model: {model}")
        offset = self._pool_offsets[model]
        if offset >= len(self._pools[model]):
            raise InvalidAction(f"No remaining branches for model {model}")
        source = self._pools[model][offset]
        self._pool_offsets[model] += 1
        branch = RuntimeBranch(runtime_id=self._next_runtime_id, model=model, source=source)
        self._next_runtime_id += 1
        self._branches[branch.runtime_id] = branch
        answer = self._advance(branch, charge_latency=charge_latency)
        self._append_action("SPAWN", branch, answer=answer)
        return self._public_branch(branch)

    def spawn(self, model: str) -> Dict[str, Any]:
        return self._spawn(model, charge_latency=True)

    def spawn_many(self, model: str, count: int) -> List[Dict[str, Any]]:
        """Atomically validate budget/pool capacity before opening a branch cohort."""
        self._require_running()
        count = int(count)
        if count < 1:
            raise InvalidAction("spawn_many count must be positive")
        if model not in self._pools:
            raise InvalidAction(f"Unknown model: {model}")
        offset = self._pool_offsets[model]
        sources = self._pools[model][offset : offset + count]
        if len(sources) != count:
            raise InvalidAction(f"Not enough remaining branches for {model}: need {count}")
        weighted = sum(
            self._cost_model.weighted(
                model,
                source.stages[0].checkpoints[0].raw_tokens,
            )
            for source in sources
        )
        if not self._fits(weighted):
            raise InvalidAction("Atomic spawn cohort exceeds weighted-compute budget")
        first_checkpoints = [source.stages[0].checkpoints[0] for source in sources]
        self._charge_parallel_latency(
            [checkpoint.think_elapsed_s for checkpoint in first_checkpoints]
        )
        self._charge_parallel_latency(
            [checkpoint.probe_elapsed_s for checkpoint in first_checkpoints]
        )
        return [self._spawn(model, charge_latency=False) for _ in range(count)]

    def continue_branch(self, branch_id: int) -> Dict[str, Any]:
        self._require_running()
        branch = self._require_branch(branch_id)
        answer = self._advance(branch, charge_latency=True)
        self._append_action("CONTINUE", branch, answer=answer)
        return self._public_branch(branch)

    def continue_many(self, branch_ids: List[int]) -> List[Dict[str, Any]]:
        """Advance a synchronous cohort only if the complete round fits the budget."""
        self._require_running()
        ids = [int(value) for value in branch_ids]
        if len(ids) != len(set(ids)):
            raise InvalidAction("continue_many branch_ids must be unique")
        branches = [self._require_branch(branch_id) for branch_id in ids]
        if any(not branch.can_continue for branch in branches):
            raise InvalidAction("Every continue_many branch must be able to continue")
        weighted = sum(
            self._cost_model.weighted(
                branch.model,
                branch.stage.checkpoints[branch.checkpoint_index + 1].raw_tokens,
            )
            for branch in branches
        )
        if not self._fits(weighted):
            raise InvalidAction("Atomic checkpoint round exceeds weighted-compute budget")
        checkpoints = [
            branch.stage.checkpoints[branch.checkpoint_index + 1]
            for branch in branches
        ]
        self._charge_parallel_latency(
            [checkpoint.think_elapsed_s for checkpoint in checkpoints]
        )
        self._charge_parallel_latency(
            [checkpoint.probe_elapsed_s for checkpoint in checkpoints]
        )
        results = []
        for branch in branches:
            answer = self._advance(branch, charge_latency=False)
            self._append_action("CONTINUE", branch, answer=answer)
            results.append(self._public_branch(branch))
        return results

    def _refine(self, branch_id: int, *, charge_latency: bool) -> Dict[str, Any]:
        self._require_running()
        branch = self._require_branch(branch_id)
        if not branch.can_refine:
            raise InvalidAction(f"Branch {branch_id} cannot refine before terminal stage")
        feedback_tokens = branch.stage.feedback_tokens
        self._charge(
            branch.model,
            feedback=feedback_tokens,
            latency_s=branch.stage.feedback_elapsed_s if charge_latency else 0.0,
        )
        branch.stage_index += 1
        branch.checkpoint_index = -1
        self._append_action("REFINE", branch, feedback_tokens=feedback_tokens)
        return self._public_branch(branch)

    def refine(self, branch_id: int) -> Dict[str, Any]:
        return self._refine(branch_id, charge_latency=True)

    def refine_many(self, branch_ids: List[int]) -> List[Dict[str, Any]]:
        """Enter the next stage for all survivors as one atomic budget operation."""
        self._require_running()
        ids = [int(value) for value in branch_ids]
        if len(ids) != len(set(ids)):
            raise InvalidAction("refine_many branch_ids must be unique")
        branches = [self._require_branch(branch_id) for branch_id in ids]
        if any(not branch.can_refine for branch in branches):
            raise InvalidAction("Every refine_many branch must be able to refine")
        weighted = sum(
            self._cost_model.weighted(branch.model, branch.stage.feedback_tokens)
            for branch in branches
        )
        if not self._fits(weighted):
            raise InvalidAction("Atomic refine cohort exceeds weighted-compute budget")
        self._charge_parallel_latency(
            [branch.stage.feedback_elapsed_s for branch in branches]
        )
        return [
            self._refine(branch.runtime_id, charge_latency=False)
            for branch in branches
        ]

    def _complete_candidate_weighted_cost(
        self,
        model: str,
        source: Branch,
        checkpoints_per_stage: int,
        max_refine_rounds: int,
    ) -> float:
        raw_tokens = 0
        stage_index = 0
        for refine_index in range(max_refine_rounds + 1):
            stage = source.stages[stage_index]
            observed = stage.checkpoints[:checkpoints_per_stage]
            raw_tokens += sum(checkpoint.raw_tokens for checkpoint in observed)
            if refine_index >= max_refine_rounds:
                break
            if len(observed) < len(stage.checkpoints) or stage_index + 1 >= len(source.stages):
                break
            raw_tokens += stage.feedback_tokens
            stage_index += 1
        return self._cost_model.weighted(model, raw_tokens)

    def can_afford_complete_candidates(
        self,
        model: str,
        *,
        count: int,
        checkpoints_per_stage: int,
        max_refine_rounds: int,
    ) -> bool:
        """Return only feasibility; future token deltas and answers remain private."""
        if model not in self._pools or count < 1 or checkpoints_per_stage < 1:
            return False
        offset = self._pool_offsets[model]
        sources = self._pools[model][offset : offset + count]
        if len(sources) != count:
            return False
        weighted = sum(
            self._complete_candidate_weighted_cost(
                model,
                source,
                checkpoints_per_stage,
                max_refine_rounds,
            )
            for source in sources
        )
        return self._fits(weighted)

    def prune(self, branch_id: int) -> Dict[str, Any]:
        self._require_running()
        branch = self._require_branch(branch_id)
        if branch.pruned:
            raise InvalidAction(f"Branch {branch_id} is already pruned")
        branch.pruned = True
        self._append_action("PRUNE", branch)
        return self._public_branch(branch)

    def finish(self, answer: Optional[str] = None) -> Optional[str]:
        self._require_running()
        final_answer = self._normalize(answer) if answer is not None else self.majority_answer()
        self._trace.append(
            {
                "step": len(self._trace),
                "action": "ANSWER",
                "answer": final_answer,
                "cost": self.cost,
            }
        )
        self._finished = True
        return final_answer

    def majority_answer(self) -> Optional[str]:
        answers = [
            branch.latest_answer
            for branch in self._branches.values()
            if not branch.pruned and branch.latest_answer is not None
        ]
        if not answers:
            return None
        counts = Counter(answers)
        top = max(counts.values())
        return next(answer for answer in answers if counts[answer] == top)

    def _require_branch(self, branch_id: int) -> RuntimeBranch:
        try:
            return self._branches[int(branch_id)]
        except (KeyError, ValueError) as exc:
            raise InvalidAction(f"Unknown branch: {branch_id}") from exc
