from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from math import comb
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
from copy import deepcopy


INVALID_ANSWERS = {"", "ERR", "PARSE_ERROR"}


@dataclass(frozen=True)
class ResourceProfile:
    """Fixed per-episode resources for a standard four-dimensional baseline."""

    model_id: str
    max_width: int
    checkpoints_per_stage: int
    max_refine_rounds: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be one non-empty model name")
        if self.max_width < 1:
            raise ValueError("max_width must be >= 1")
        if self.checkpoints_per_stage < 1:
            raise ValueError("checkpoints_per_stage must be >= 1")
        if self.max_refine_rounds < 0:
            raise ValueError("max_refine_rounds must be >= 0")

    @property
    def max_stages(self) -> int:
        return 1 + self.max_refine_rounds

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResourceProfile":
        resources = config.get("resources")
        if not isinstance(resources, dict):
            raise ValueError("standard baselines require a resources mapping")
        model_id = resources.get("model_id")
        if isinstance(model_id, (list, tuple, set)):
            raise ValueError("standard baselines accept exactly one model_id per episode")
        required = {
            "model_id",
            "max_width",
            "checkpoints_per_stage",
            "max_refine_rounds",
        }
        missing = sorted(required - set(resources))
        if missing:
            raise ValueError(f"missing resource fields: {missing}")
        return cls(
            model_id=str(model_id),
            max_width=int(resources["max_width"]),
            checkpoints_per_stage=int(resources["checkpoints_per_stage"]),
            max_refine_rounds=int(resources["max_refine_rounds"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_width": self.max_width,
            "checkpoints_per_stage": self.checkpoints_per_stage,
            "max_refine_rounds": self.max_refine_rounds,
        }


class ControllerPolicy(ABC):
    NAME = "controller"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        raw_beta = self.config.get("budget_beta", self.config.get("beta"))
        # Only legacy beta-grid policies receive these compatibility fields.
        if raw_beta is not None:
            self.budget_beta = max(0.0, min(1.0, float(raw_beta)))
            self.beta = self.budget_beta
        requirements = self.config.get("requirements")
        self.requirements = deepcopy(requirements) if isinstance(requirements, dict) else None
        self.last_result: Dict[str, Any] = {}

    @abstractmethod
    def solve(self, env) -> Optional[str]:
        """Use only ReplayEnvironment's public methods and return an observed answer."""
        raise NotImplementedError

    def description(self) -> str:
        budget_beta = getattr(self, "budget_beta", None)
        if budget_beta is None:
            return self.NAME
        return f"{self.NAME} | budget_beta={budget_beta:.3f}"


def valid_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in INVALID_ANSWERS:
        return None
    return text


def usable_answers(branches: Iterable[Dict[str, Any]]) -> list[str]:
    answers: list[str] = []
    for branch in branches:
        if branch.get("pruned"):
            continue
        answer = valid_answer(branch.get("latest_answer"))
        if answer is not None:
            answers.append(answer)
    return answers


def unique_mode(answers: Sequence[str]) -> Optional[str]:
    """Return the unique plurality answer, or None when the top count is tied."""
    if not answers:
        return None
    counts = Counter(answers)
    top = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else None


def deterministic_plurality(answers: Sequence[str]) -> Optional[str]:
    """Use first appearance as the deterministic final tie-breaker."""
    if not answers:
        return None
    counts = Counter(answers)
    top = max(counts.values())
    return next(answer for answer in answers if counts[answer] == top)


def top_two_counts(answers: Sequence[str]) -> Tuple[Optional[str], int, int, bool]:
    if not answers:
        return None, 0, 0, False
    counts = Counter(answers)
    top1 = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == top1]
    top2 = max((count for count in counts.values() if count < top1), default=0)
    return winners[0], top1, top2 if len(winners) == 1 else top1, len(winners) > 1


def vote_stats(answers: Iterable[str]) -> Tuple[Optional[str], int, int, int, float]:
    values = list(answers)
    winner, top1, top2, tied = top_two_counts(values)
    if not values:
        return None, 0, 0, 0, 0.0
    return (None if tied else winner), top1, top2, len(values), top1 / len(values)


def asc_posterior_confidence(top1_count: int, top2_count: int) -> float:
    """Compute 1-BetaCDF(0.5, top1+1, top2+1) exactly for integer counts."""
    if top1_count < 0 or top2_count < 0:
        raise ValueError("vote counts must be non-negative")
    a = top1_count + 1
    b = top2_count + 1
    n = a + b - 1
    return sum(comb(n, index) for index in range(a)) / (2**n)
