from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


REPLAY_SCHEMA_VERSION = "person_tts_compact_replay_v2_latency"
LEGACY_REPLAY_SCHEMA_VERSIONS = {"person_tts_compact_replay_v1"}


@dataclass(frozen=True)
class Checkpoint:
    answer: str
    think_tokens: int
    probe_tokens: int
    think_elapsed_s: float = 0.0
    probe_elapsed_s: float = 0.0

    @property
    def raw_tokens(self) -> int:
        return self.think_tokens + self.probe_tokens


@dataclass(frozen=True)
class Stage:
    stage_index: int
    kind: str
    refine_round: int
    terminal_reason: Optional[str]
    feedback_tokens: int
    checkpoints: List[Checkpoint]
    feedback_elapsed_s: float = 0.0


@dataclass(frozen=True)
class Branch:
    branch_id: int
    stages: List[Stage]


@dataclass(frozen=True)
class ReplayProblem:
    dataset: str
    index: int
    question_id: str
    question_hash: str
    gold_answer: str
    models: Dict[str, List[Branch]]
    source_files: Dict[str, Dict[str, Any]]
    latency_available: bool = True
    schema_version: str = REPLAY_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ReplayProblem":
        schema_version = value.get("schema_version")
        if schema_version not in {REPLAY_SCHEMA_VERSION, *LEGACY_REPLAY_SCHEMA_VERSIONS}:
            raise ValueError(
                f"Unsupported replay schema {schema_version!r}; expected one of "
                f"{sorted({REPLAY_SCHEMA_VERSION, *LEGACY_REPLAY_SCHEMA_VERSIONS})!r}"
            )
        latency_available = bool(
            value.get("latency_available", schema_version == REPLAY_SCHEMA_VERSION)
        )
        models: Dict[str, List[Branch]] = {}
        for model, raw_branches in value["models"].items():
            branches: List[Branch] = []
            for raw_branch in raw_branches:
                stages = [
                    Stage(
                        stage_index=int(raw_stage["stage_index"]),
                        kind=str(raw_stage["kind"]),
                        refine_round=int(raw_stage["refine_round"]),
                        terminal_reason=raw_stage.get("terminal_reason"),
                        feedback_tokens=int(raw_stage.get("feedback_tokens", 0)),
                        checkpoints=[
                            Checkpoint(
                                answer=str(checkpoint["answer"]),
                                think_tokens=int(checkpoint["think_tokens"]),
                                probe_tokens=int(checkpoint["probe_tokens"]),
                                think_elapsed_s=float(checkpoint.get("think_elapsed_s", 0.0)),
                                probe_elapsed_s=float(checkpoint.get("probe_elapsed_s", 0.0)),
                            )
                            for checkpoint in raw_stage["checkpoints"]
                        ],
                        feedback_elapsed_s=float(raw_stage.get("feedback_elapsed_s", 0.0)),
                    )
                    for raw_stage in raw_branch["stages"]
                ]
                branches.append(Branch(branch_id=int(raw_branch["branch_id"]), stages=stages))
            models[model] = branches
        return cls(
            dataset=str(value["dataset"]),
            index=int(value["index"]),
            question_id=str(value["question_id"]),
            question_hash=str(value["question_hash"]),
            gold_answer=str(value["gold_answer"]),
            models=models,
            source_files=dict(value.get("source_files", {})),
            latency_available=latency_available,
            schema_version=schema_version,
        )
