from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping


PUBLIC_TRACE_FILENAME = "public_traces.jsonl"
TRACE_SUMMARY_FILENAME = "trace_summary.json"
MAX_QUERY_CASES = 50
MAX_QUERY_TRACE_STEPS = 40


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def public_trace_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    trace = list(record.get("public_trace") or [])
    result = {
        "record_key": str(record["record_key"]),
        "dataset": str(record["dataset"]),
        "problem_id": f"P{int(record['index']):04d}",
        "seed": int(record["seed"]),
        "correct": bool(record["correct"]),
        "cost": {
            key: record["cost"][key]
            for key in (
                "think_tokens",
                "probe_tokens",
                "feedback_tokens",
                "raw_tokens",
                "weighted_compute",
                "latency_s",
            )
            if key in record["cost"]
        },
        "trace_steps_total": int(record.get("trace_steps", len(trace))),
        "trace_steps_retained": len(trace),
        "trace": trace,
    }
    if "profile_id" in record:
        result["profile_id"] = str(record["profile_id"])
        result["profile_hash"] = str(record["profile_hash"])
    else:
        result["beta"] = float(record["beta"])
    return result


@dataclass
class _TraceAccumulator:
    episodes: int = 0
    correct: int = 0
    raw_tokens: float = 0.0
    weighted_compute: float = 0.0
    latency_s: float = 0.0
    trace_steps_total: float = 0.0
    trace_steps_retained: float = 0.0
    with_refine: int = 0
    with_prune: int = 0
    with_model_routing: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    model_spawns: Counter[str] = field(default_factory=Counter)

    def add(self, item: Mapping[str, Any]) -> tuple[float, float, float]:
        trace = list(item.get("trace") or [])
        actions = [str(event.get("action")) for event in trace if event.get("action")]
        models = {
            str(event["model"])
            for event in trace
            if event.get("action") == "SPAWN" and event.get("model")
        }
        self.episodes += 1
        self.correct += int(bool(item.get("correct")))
        self.action_counts.update(actions)
        for event in trace:
            if event.get("action") == "SPAWN" and event.get("model"):
                self.model_spawns[str(event["model"])] += 1
        self.with_refine += int("REFINE" in actions)
        self.with_prune += int("PRUNE" in actions)
        self.with_model_routing += int(len(models) > 1)
        cost = item.get("cost") or {}
        raw = float(cost.get("raw_tokens", 0.0))
        weighted = float(cost.get("weighted_compute", 0.0))
        latency = float(cost.get("latency_s", 0.0))
        self.raw_tokens += raw
        self.weighted_compute += weighted
        self.latency_s += latency
        self.trace_steps_total += float(item.get("trace_steps_total", len(trace)))
        self.trace_steps_retained += float(item.get("trace_steps_retained", len(trace)))
        return raw, weighted, latency


class _TraceSummaryStore:
    def __init__(self) -> None:
        # An empty SQLite path creates a private, automatically deleted disk-backed DB.
        self.connection = sqlite3.connect("")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE metrics (group_key TEXT, raw REAL, weighted REAL, latency REAL)"
        )
        self.accumulators: Dict[str, _TraceAccumulator] = {}

    def close(self) -> None:
        self.connection.close()

    def add(self, group_key: str, item: Mapping[str, Any]) -> None:
        accumulator = self.accumulators.setdefault(group_key, _TraceAccumulator())
        raw, weighted, latency = accumulator.add(item)
        self.connection.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?)",
            (group_key, raw, weighted, latency),
        )

    def _percentile(self, group_key: str, column: str, fraction: float) -> float:
        if column not in {"raw", "weighted", "latency"}:
            raise ValueError(f"unsupported percentile column: {column}")
        count = self.accumulators[group_key].episodes
        if not count:
            return 0.0
        index = round(fraction * (count - 1))
        row = self.connection.execute(
            f"SELECT {column} FROM metrics WHERE group_key = ? "
            f"ORDER BY {column} LIMIT 1 OFFSET ?",
            (group_key, index),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing trace percentile row for {group_key}")
        return float(row[0])

    def _sum(self, group_key: str, column: str) -> float:
        if column not in {"raw", "weighted", "latency"}:
            raise ValueError(f"unsupported sum column: {column}")
        return sum(
            float(row[0])
            for row in self.connection.execute(
                f"SELECT {column} FROM metrics WHERE group_key = ? ORDER BY rowid",
                (group_key,),
            )
        )

    def summarize(self, group_key: str) -> Dict[str, Any]:
        item = self.accumulators.get(group_key, _TraceAccumulator())
        divisor = item.episodes or 1

        def percentiles(column: str) -> Dict[str, float]:
            return {
                "p50": self._percentile(group_key, column, 0.50),
                "p90": self._percentile(group_key, column, 0.90),
                "p99": self._percentile(group_key, column, 0.99),
            } if item.episodes else {"p50": 0.0, "p90": 0.0, "p99": 0.0}

        return {
            "episodes": item.episodes,
            "correct": item.correct,
            "accuracy": item.correct / divisor,
            "avg_raw_tokens": self._sum(group_key, "raw") / divisor,
            "avg_weighted_compute": self._sum(group_key, "weighted") / divisor,
            "avg_latency_s": self._sum(group_key, "latency") / divisor,
            "raw_token_percentiles": percentiles("raw"),
            "weighted_compute_percentiles": percentiles("weighted"),
            "latency_s_percentiles": percentiles("latency"),
            "avg_trace_steps_total": item.trace_steps_total / divisor,
            "avg_trace_steps_retained": item.trace_steps_retained / divisor,
            "episodes_with_refine": item.with_refine,
            "episodes_with_prune": item.with_prune,
            "episodes_with_multiple_spawn_models": item.with_model_routing,
            "action_counts": dict(sorted(item.action_counts.items())),
            "spawn_counts_by_model": dict(sorted(item.model_spawns.items())),
        }


def summarize_trace_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    store = _TraceSummaryStore()
    mode: str | None = None
    profile_keys: set[str] = set()
    dataset_profile_keys: set[tuple[str, str]] = set()
    beta_keys: set[float] = set()
    dataset_beta_keys: set[tuple[str, float]] = set()
    try:
        for record in records:
            personalized = "profile_id" in record
            current_mode = "personalized" if personalized else "legacy"
            if mode is None:
                mode = current_mode
            elif mode != current_mode:
                raise ValueError("trace JSONL mixes personalized and legacy records")
            store.add("all", record)
            dataset = str(record["dataset"])
            if personalized:
                profile_id = str(record["profile_id"])
                profile_keys.add(profile_id)
                dataset_profile_keys.add((dataset, profile_id))
                store.add(f"profile\u241f{profile_id}", record)
                store.add(f"dataset_profile\u241f{dataset}\u241f{profile_id}", record)
            else:
                beta = float(record["beta"])
                beta_keys.add(beta)
                dataset_beta_keys.add((dataset, beta))
                store.add(f"beta\u241f{beta!r}", record)
                store.add(f"dataset_beta\u241f{dataset}\u241f{beta!r}", record)
        store.connection.commit()
        for column in ("raw", "weighted", "latency"):
            store.connection.execute(
                f"CREATE INDEX metrics_{column} ON metrics (group_key, {column})"
            )
        if mode == "personalized":
            return {
                "schema_version": "person_tts_public_trace_v2_profiles",
                "mode": "personalized_hard_requirements",
                "all": store.summarize("all"),
                "by_profile": {
                    profile_id: store.summarize(f"profile\u241f{profile_id}")
                    for profile_id in sorted(profile_keys)
                },
                "by_dataset_profile": {
                    f"{dataset}|profile={profile_id}": store.summarize(
                        f"dataset_profile\u241f{dataset}\u241f{profile_id}"
                    )
                    for dataset, profile_id in sorted(dataset_profile_keys)
                },
            }
        return {
            "schema_version": "person_tts_public_trace_v1",
            "all": store.summarize("all"),
            "by_beta": {
                f"beta={beta:.1f}": store.summarize(f"beta\u241f{beta!r}")
                for beta in sorted(beta_keys)
            },
            "by_dataset_beta": {
                f"{dataset}|beta={beta:.1f}": store.summarize(
                    f"dataset_beta\u241f{dataset}\u241f{beta!r}"
                )
                for dataset, beta in sorted(dataset_beta_keys)
            },
        }
    finally:
        store.close()


def write_trace_artifacts(
    episode_records: Iterable[Mapping[str, Any]],
    *,
    trace_path: Path,
    summary_path: Path,
) -> Dict[str, Any]:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = trace_path.with_suffix(trace_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for episode_record in episode_records:
            record = public_trace_record(episode_record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, trace_path)
    summary = write_trace_summary(trace_path, summary_path)
    return summary


def write_trace_summary(trace_path: Path, summary_path: Path) -> Dict[str, Any]:
    summary = summarize_trace_records(_iter_jsonl(trace_path))
    _atomic_json(summary_path, summary)
    return summary


def _workspace_path(raw: str) -> Path:
    root = Path.cwd().resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("trace source must stay inside the discovery workspace") from exc
    if path.suffix != ".jsonl":
        raise ValueError("trace source must be a JSONL file")
    return path


def _case_matches(record: Mapping[str, Any], args: argparse.Namespace) -> bool:
    if args.dataset is not None and record.get("dataset") != args.dataset:
        return False
    if args.beta is not None and float(record.get("beta", -1)) != args.beta:
        return False
    if args.profile is not None and record.get("profile_id") != args.profile:
        return False
    if args.correct != "any" and bool(record.get("correct")) != (args.correct == "true"):
        return False
    trace = list(record.get("trace") or [])
    if args.action is not None and not any(
        event.get("action") == args.action for event in trace
    ):
        return False
    if args.model is not None and not any(
        event.get("model") == args.model for event in trace
    ):
        return False
    return True


def _spread_positions(total: int, limit: int) -> List[int]:
    if total <= 0 or limit <= 0:
        return []
    if limit >= total:
        return list(range(total))
    if limit == 1:
        return [0]
    return sorted(
        {
            round(index * (total - 1) / (limit - 1))
            for index in range(limit)
        }
    )


def _bounded_case(record: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(record)
    trace = list(record.get("trace") or [])
    if len(trace) <= MAX_QUERY_TRACE_STEPS:
        return result
    priority = {
        index
        for index, event in enumerate(trace)
        if event.get("action") != "CONTINUE"
    }
    priority.update((0, len(trace) - 1))
    ordered_priority = sorted(priority)
    if len(ordered_priority) >= MAX_QUERY_TRACE_STEPS:
        selected = [
            ordered_priority[index]
            for index in _spread_positions(
                len(ordered_priority), MAX_QUERY_TRACE_STEPS
            )
        ]
    else:
        selected_set = set(ordered_priority)
        remaining = [
            index for index in range(len(trace)) if index not in selected_set
        ]
        slots = MAX_QUERY_TRACE_STEPS - len(selected_set)
        selected_set.update(
            remaining[index]
            for index in _spread_positions(len(remaining), slots)
        )
        selected = sorted(selected_set)
    result["trace"] = [trace[index] for index in selected]
    result["trace_steps_retained"] = len(result["trace"])
    return result


def _query_cases(
    source: Path, args: argparse.Namespace
) -> Dict[str, Any]:
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    if args.limit < 1 or args.limit > MAX_QUERY_CASES:
        raise ValueError(f"--limit must be in [1, {MAX_QUERY_CASES}]")
    if args.sort.startswith("weighted"):
        cost_key = "weighted_compute"
    elif args.sort.startswith("latency"):
        cost_key = "latency_s"
    else:
        cost_key = "raw_tokens"

    source_records = matching_records = 0
    page: List[Mapping[str, Any]] = []
    connection: sqlite3.Connection | None = None
    if args.sort != "record":
        connection = sqlite3.connect("")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE cases (sequence INTEGER, sort_value REAL, payload TEXT)"
        )
    try:
        for record in _iter_jsonl(source):
            source_records += 1
            if not _case_matches(record, args):
                continue
            sequence = matching_records
            matching_records += 1
            if connection is None:
                if args.offset <= sequence < args.offset + args.limit:
                    page.append(record)
            else:
                connection.execute(
                    "INSERT INTO cases VALUES (?, ?, ?)",
                    (
                        sequence,
                        float((record.get("cost") or {}).get(cost_key, 0.0)),
                        json.dumps(record, ensure_ascii=False),
                    ),
                )
        if connection is not None:
            connection.commit()
            direction = "DESC" if args.sort.endswith("desc") else "ASC"
            rows = connection.execute(
                f"SELECT payload FROM cases ORDER BY sort_value {direction}, sequence ASC "
                "LIMIT ? OFFSET ?",
                (args.limit, args.offset),
            )
            page = [json.loads(row[0]) for row in rows]
    finally:
        if connection is not None:
            connection.close()
    return {
        "source_records": source_records,
        "matching_records": matching_records,
        "offset": args.offset,
        "returned_records": len(page),
        "case_trace_step_limit": MAX_QUERY_TRACE_STEPS,
        "cases": [_bounded_case(record) for record in page],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze full sanitized search traces without reading private replay data"
    )
    parser.add_argument("--source", default="SEARCH_TRACES.jsonl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summary")
    cases = subparsers.add_parser("cases")
    cases.add_argument("--dataset")
    cases.add_argument("--beta", type=float)
    cases.add_argument("--profile")
    cases.add_argument("--correct", choices=("any", "true", "false"), default="any")
    cases.add_argument("--action")
    cases.add_argument("--model")
    cases.add_argument(
        "--sort",
        choices=(
            "weighted-desc",
            "weighted-asc",
            "latency-desc",
            "latency-asc",
            "raw-desc",
            "raw-asc",
            "record",
        ),
        default="record",
    )
    cases.add_argument("--offset", type=int, default=0)
    cases.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    source = _workspace_path(args.source)
    if args.command == "summary":
        result: Any = summarize_trace_records(_iter_jsonl(source))
    else:
        result = _query_cases(source, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
