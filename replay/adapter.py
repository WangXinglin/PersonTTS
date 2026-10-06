from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .schema import Branch, Checkpoint, REPLAY_SCHEMA_VERSION, ReplayProblem, Stage


SOURCE_SCHEMA_VERSION = "personalized_tts_traj_v5_full_probe"


class ReplayDataError(ValueError):
    pass


def _elapsed_s(event: Dict[str, Any], path: Path, context: str) -> float:
    value = float(event.get("branch_elapsed_s", -1.0))
    if not math.isfinite(value) or value < 0.0:
        raise ReplayDataError(
            f"Missing or invalid branch_elapsed_s in {path}, {context}: {value!r}"
        )
    return value


def _source_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _question_identity(data: Dict[str, Any], path: Path) -> Tuple[str, str, str]:
    question = str(data.get("question") or "")
    if not question:
        raise ReplayDataError(f"Missing question in {path}")
    question_id = str(data.get("question_id", data.get("index", path.stem)))
    gold = str(data.get("answer", ""))
    if not gold:
        raise ReplayDataError(f"Missing gold answer in {path}")
    return question_id, hashlib.sha256(question.encode("utf-8")).hexdigest(), gold


def _compact_stage(raw: Dict[str, Any], path: Path, branch_id: int) -> Stage:
    answers = list(raw.get("answer_trace") or [])
    probe_tokens = [int(value) for value in (raw.get("probe_token_trace") or [])]
    think_events = list(raw.get("think_events") or [])
    probe_events = list(raw.get("probe_events") or [])
    if not (len(answers) == len(probe_tokens) == len(think_events) == len(probe_events)):
        raise ReplayDataError(
            f"Checkpoint alignment mismatch in {path}, branch={branch_id}, "
            f"stage={raw.get('stage_index')}: answers={len(answers)}, "
            f"probe_tokens={len(probe_tokens)}, think_events={len(think_events)}, "
            f"probe_events={len(probe_events)}"
        )
    checkpoints = []
    for checkpoint_index, (answer, probe_count, think_event, probe_event) in enumerate(
        zip(answers, probe_tokens, think_events, probe_events)
    ):
        context = (
            f"branch={branch_id}, stage={raw.get('stage_index')}, "
            f"checkpoint={checkpoint_index}"
        )
        checkpoints.append(
            Checkpoint(
                answer=str(answer or "ERR"),
                think_tokens=int(think_event.get("tokens", 0)),
                probe_tokens=probe_count,
                think_elapsed_s=_elapsed_s(think_event, path, f"{context}, think"),
                probe_elapsed_s=_elapsed_s(probe_event, path, f"{context}, probe"),
            )
        )
    if not checkpoints:
        raise ReplayDataError(
            f"No checkpoints in {path}, branch={branch_id}, stage={raw.get('stage_index')}"
        )
    feedback_event = raw.get("feedback_event") or {}
    feedback_tokens = int(feedback_event.get("tokens", 0))
    feedback_elapsed_s = (
        _elapsed_s(
            feedback_event,
            path,
            f"branch={branch_id}, stage={raw.get('stage_index')}, feedback",
        )
        if feedback_tokens > 0
        else 0.0
    )
    return Stage(
        stage_index=int(raw.get("stage_index", 0)),
        kind=str(raw.get("kind", "")),
        refine_round=int(raw.get("refine_round", 0)),
        terminal_reason=raw.get("terminal_reason"),
        feedback_tokens=feedback_tokens,
        checkpoints=checkpoints,
        feedback_elapsed_s=feedback_elapsed_s,
    )


def _compact_model(path: Path) -> Tuple[Tuple[str, str, str], List[Branch], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    trajectory = data.get("trajectory") or {}
    if trajectory.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ReplayDataError(
            f"Unexpected source schema in {path}: {trajectory.get('schema_version')!r}"
        )
    branches: List[Branch] = []
    for raw_branch in trajectory.get("branches") or []:
        branch_id = int(raw_branch.get("branch_id", len(branches)))
        stages = [_compact_stage(stage, path, branch_id) for stage in raw_branch.get("stages") or []]
        if not stages:
            raise ReplayDataError(f"No stages in {path}, branch={branch_id}")
        branches.append(Branch(branch_id=branch_id, stages=stages))
    expected = int(trajectory.get("total_branches", len(branches)))
    if not branches or len(branches) != expected:
        raise ReplayDataError(f"Branch count mismatch in {path}: {len(branches)} != {expected}")
    compact_totals = {
        "think_tokens": sum(
            checkpoint.think_tokens
            for branch in branches
            for stage in branch.stages
            for checkpoint in stage.checkpoints
        ),
        "probe_tokens": sum(
            checkpoint.probe_tokens
            for branch in branches
            for stage in branch.stages
            for checkpoint in stage.checkpoints
        ),
        "feedback_tokens": sum(
            stage.feedback_tokens for branch in branches for stage in branch.stages
        ),
    }
    source_totals = {
        "think_tokens": int(trajectory.get("total_think_tokens", -1)),
        "probe_tokens": int(trajectory.get("total_probe_tokens", -1)),
        "feedback_tokens": int(trajectory.get("total_feedback_tokens", -1)),
    }
    if compact_totals != source_totals:
        raise ReplayDataError(
            f"Token conservation mismatch in {path}: compact={compact_totals}, "
            f"source={source_totals}"
        )
    identity = _question_identity(data, path)
    fingerprint = _source_fingerprint(path)
    fingerprint["token_totals"] = compact_totals
    del data
    gc.collect()
    return identity, branches, fingerprint


def build_problem(source_root: Path, dataset: str, index: int, models: Iterable[str]) -> ReplayProblem:
    compact_models: Dict[str, List[Branch]] = {}
    source_files: Dict[str, Dict[str, Any]] = {}
    expected_identity: Tuple[str, str, str] | None = None
    for model in models:
        source = source_root / dataset / model / f"{index}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        identity, branches, fingerprint = _compact_model(source)
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ReplayDataError(
                f"Question/gold mismatch across models for {dataset}/{index}: "
                f"{identity!r} != {expected_identity!r}"
            )
        compact_models[model] = branches
        source_files[model] = fingerprint
    assert expected_identity is not None
    question_id, question_hash, gold = expected_identity
    return ReplayProblem(
        dataset=dataset,
        index=index,
        question_id=question_id,
        question_hash=question_hash,
        gold_answer=gold,
        models=compact_models,
        source_files=source_files,
        latency_available=True,
        schema_version=REPLAY_SCHEMA_VERSION,
    )


def write_problem_atomic(problem: ReplayProblem, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(problem.to_dict(), handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)


def load_problem(path: Path) -> ReplayProblem:
    with path.open("r", encoding="utf-8") as handle:
        return ReplayProblem.from_dict(json.load(handle))
