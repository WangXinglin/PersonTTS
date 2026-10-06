from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class CostLedger:
    think_tokens: int = 0
    probe_tokens: int = 0
    feedback_tokens: int = 0
    weighted_compute: float = 0.0
    latency_s: float = 0.0

    @property
    def raw_tokens(self) -> int:
        return self.think_tokens + self.probe_tokens + self.feedback_tokens

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "think_tokens": self.think_tokens,
            "probe_tokens": self.probe_tokens,
            "feedback_tokens": self.feedback_tokens,
            "raw_tokens": self.raw_tokens,
            "weighted_compute": self.weighted_compute,
            "latency_s": self.latency_s,
        }


class ParameterWeightedCost:
    def __init__(self, model_weights: Dict[str, float]):
        if not model_weights or any(weight <= 0 for weight in model_weights.values()):
            raise ValueError("All model weights must be positive")
        self.model_weights = dict(model_weights)

    def weight(self, model: str) -> float:
        try:
            return self.model_weights[model]
        except KeyError as exc:
            raise ValueError(f"Unknown model: {model}") from exc

    def weighted(self, model: str, raw_tokens: int) -> float:
        return self.weight(model) * raw_tokens
