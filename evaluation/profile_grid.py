from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from array import array
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from config_utils import load_config, model_weights, resolve_from_root
from evaluation.answering import is_correct, normalize_prediction
from policies.seeds import FixedBudgetGridPolicy
from replay.adapter import load_problem
from replay.environment import ReplayEnvironment


MANIFEST_SCHEMA = "person_tts_profile_grid_manifest_v1"
SHARD_SCHEMA = "person_tts_profile_grid_shard_v1"
RESULT_SCHEMA = "person_tts_profile_grid_results_v1"
ANCHOR_SCHEMA = "person_tts_profile_anchors_v1"
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _csv_values(raw: str, converter) -> tuple:
    values = tuple(converter(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("grid arguments must contain at least one value")
    if len(values) != len(set(values)):
        raise ValueError(f"grid arguments must be unique: {raw}")
    return values


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty distribution")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1 - ratio) + ordered[upper] * ratio


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "quantiles": {
            f"q{round(fraction * 100):02d}": _percentile(values, fraction)
            for fraction in QUANTILES
        },
    }


@dataclass(frozen=True, order=True)
class GridPoint:
    model: str
    width: int
    depth: int
    refine_rounds: int

    @property
    def point_id(self) -> str:
        safe_model = self.model.replace("/", "_")
        return (
            f"fixed_grid__{safe_model}__w{self.width}"
            f"__d{self.depth}__r{self.refine_rounds}"
        )

    def policy_config(self) -> Dict[str, Any]:
        return {
            "resources": {
                "model_id": self.model,
                "max_width": self.width,
                "checkpoints_per_stage": self.depth,
                "max_refine_rounds": self.refine_rounds,
            }
        }


@dataclass(frozen=True)
class GridShardTask:
    cache_path: str
    task_key: str


_WORKER_CONFIG: Dict[str, Any] = {}
_WORKER_POINTS: tuple[GridPoint, ...] = ()
_WORKER_SEED_COUNT = 0
_WORKER_WEIGHTS: Dict[str, float] = {}
_WORKER_PROJECT_ROOT = Path(".")
_WORKER_MANIFEST_SHA256 = ""


def _initialize_worker(
    config: Dict[str, Any],
    points: tuple[GridPoint, ...],
    seed_count: int,
    manifest_sha256: str,
) -> None:
    global _WORKER_CONFIG, _WORKER_POINTS, _WORKER_SEED_COUNT
    global _WORKER_WEIGHTS, _WORKER_PROJECT_ROOT, _WORKER_MANIFEST_SHA256
    _WORKER_CONFIG = config
    _WORKER_POINTS = points
    _WORKER_SEED_COUNT = seed_count
    _WORKER_WEIGHTS = model_weights(config)
    _WORKER_PROJECT_ROOT = Path(config["_root"]).parent.resolve()
    _WORKER_MANIFEST_SHA256 = manifest_sha256


def _run_shard(task: GridShardTask) -> Dict[str, Any]:
    started = time.monotonic()
    problem = load_problem(Path(task.cache_path))
    required_models = {point.model for point in _WORKER_POINTS}
    missing_models = sorted(required_models - set(problem.models))
    if missing_models:
        raise ValueError(
            f"models {missing_models!r} are absent from "
            f"{problem.dataset}:{problem.index}"
        )

    @lru_cache(maxsize=None)
    def normalize_answer(value: str) -> str | None:
        normalized, is_null = normalize_prediction(
            value, problem.dataset, _WORKER_PROJECT_ROOT
        )
        return None if is_null else normalized

    @lru_cache(maxsize=None)
    def grade_prediction(value: str | None) -> bool:
        return is_correct(
            value, problem.gold_answer, problem.dataset, _WORKER_PROJECT_ROOT
        )

    results: Dict[str, Any] = {}
    for point in _WORKER_POINTS:
        correct: list[int] = []
        latency_s: list[float] = []
        weighted_compute: list[float] = []
        raw_tokens: list[int] = []
        realized_refine_cohorts: list[int] = []
        refine_action_count: list[int] = []
        for seed in range(_WORKER_SEED_COUNT):
            env = ReplayEnvironment(
                problem,
                _WORKER_WEIGHTS,
                seed=seed,
                beta=0.5,
                answer_normalizer=normalize_answer,
            )
            policy = FixedBudgetGridPolicy(config=point.policy_config())
            prediction = policy.solve(env)
            correct.append(int(grade_prediction(prediction)))
            cost = env.cost
            latency_s.append(float(cost["latency_s"]))
            weighted_compute.append(float(cost["weighted_compute"]))
            raw_tokens.append(int(cost["raw_tokens"]))
            realized_refine_cohorts.append(
                int(policy.last_result["completed_refine_cohorts"])
            )
            refine_action_count.append(int(policy.last_result["refine_count"]))
        results[point.point_id] = {
            "correct": correct,
            "latency_s": latency_s,
            "weighted_compute": weighted_compute,
            "raw_tokens": raw_tokens,
            "realized_refine_cohorts": realized_refine_cohorts,
            "refine_action_count": refine_action_count,
        }

    return {
        "schema_version": SHARD_SCHEMA,
        "manifest_sha256": _WORKER_MANIFEST_SHA256,
        "task_key": task.task_key,
        "dataset": problem.dataset,
        "index": problem.index,
        "question_hash": problem.question_hash,
        "models": sorted(required_models),
        "seed_count": _WORKER_SEED_COUNT,
        "grid_point_count": len(_WORKER_POINTS),
        "elapsed_s": time.monotonic() - started,
        "points": results,
    }


def _cache_paths(
    config: Mapping[str, Any],
    split: str,
    datasets: Sequence[str] | None,
    indices: Sequence[int] | None,
) -> list[Path]:
    selected_datasets = list(datasets or config[f"{split}_datasets"])
    allowed = set(config[f"{split}_datasets"])
    unknown = sorted(set(selected_datasets) - allowed)
    if unknown:
        raise ValueError(f"datasets are outside the {split} split: {unknown}")
    wanted_indices = set(indices) if indices is not None else None
    root = resolve_from_root(dict(config), str(config["private_cache_root"]))
    paths: list[Path] = []
    for dataset in selected_datasets:
        candidates = sorted(
            (root / dataset).glob("*.json"), key=lambda path: int(path.stem)
        )
        paths.extend(
            path
            for path in candidates
            if wanted_indices is None or int(path.stem) in wanted_indices
        )
    if not paths:
        raise ValueError("the selected profile-grid scope contains no replay files")
    return paths


def _task_key(path: Path) -> str:
    return f"{path.parent.name}:{int(path.stem)}"


def _shard_path(output_root: Path, task_key: str) -> Path:
    dataset, index = task_key.split(":", 1)
    return output_root / "shards" / dataset / f"{int(index):03d}.json"


def _validated_existing_shard(
    path: Path,
    task: GridShardTask,
    manifest_sha256: str,
    expected_point_count: int,
    seed_count: int,
    expected_models: Sequence[str],
) -> bool:
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SHARD_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "task_key": task.task_key,
        "models": sorted(expected_models),
        "seed_count": seed_count,
        "grid_point_count": expected_point_count,
    }
    actual = {key: value.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"existing grid shard is incompatible with this run: {path}; "
            f"expected={expected}, actual={actual}"
        )
    if len(value.get("points", {})) != expected_point_count:
        raise ValueError(f"existing grid shard is incomplete: {path}")
    return True


def _acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_unix_s": time.time(),
    }
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("hostname") == socket.gethostname():
            try:
                os.kill(int(existing["pid"]), 0)
            except (OSError, ValueError, KeyError):
                path.unlink()
                return _acquire_lock(path)
        raise RuntimeError(f"profile-grid run is already locked: {existing}")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _build_points(
    models: Sequence[str],
    widths: Sequence[int],
    depths: Sequence[int],
    refine_rounds: Sequence[int],
) -> tuple[GridPoint, ...]:
    points = tuple(
        GridPoint(model, width, depth, refine)
        for model, width, depth, refine in product(
            models, widths, depths, refine_rounds
        )
    )
    for point in points:
        FixedBudgetGridPolicy(config=point.policy_config())
    return points


def _aggregate(
    *,
    output_root: Path,
    tasks: Sequence[GridShardTask],
    points: Sequence[GridPoint],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    seed_count: int,
) -> Dict[str, Any]:
    point_by_id = {point.point_id: point for point in points}
    problem_count = len({task.cache_path for task in tasks})
    correct_by_seed = {
        point.point_id: [0] * seed_count for point in points
    }
    compute_by_seed = {
        point.point_id: [0.0] * seed_count for point in points
    }
    raw_by_seed = {point.point_id: [0] * seed_count for point in points}
    refine_cohorts_by_seed = {
        point.point_id: [0] * seed_count for point in points
    }
    refine_actions_by_seed = {
        point.point_id: [0] * seed_count for point in points
    }
    latency_by_seed = {
        point.point_id: [array("d") for _ in range(seed_count)]
        for point in points
    }
    shard_count_by_point: Dict[str, int] = defaultdict(int)
    shard_elapsed_s: list[float] = []

    for task in sorted(tasks, key=lambda item: item.task_key):
        shard = json.loads(
            _shard_path(output_root, task.task_key).read_text(encoding="utf-8")
        )
        if shard.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"shard manifest mismatch: {task.task_key}")
        shard_elapsed_s.append(float(shard["elapsed_s"]))
        for point_id, record in shard["points"].items():
            if point_id not in point_by_id:
                raise ValueError(f"unknown point in shard {task.task_key}: {point_id}")
            columns = (
                record["correct"],
                record["latency_s"],
                record["weighted_compute"],
                record["raw_tokens"],
                record["realized_refine_cohorts"],
                record["refine_action_count"],
            )
            if any(len(column) != seed_count for column in columns):
                raise ValueError(
                    f"wrong seed cardinality in {task.task_key}/{point_id}"
                )
            shard_count_by_point[point_id] += 1
            for seed in range(seed_count):
                correct_by_seed[point_id][seed] += int(record["correct"][seed])
                latency_by_seed[point_id][seed].append(
                    float(record["latency_s"][seed])
                )
                compute_by_seed[point_id][seed] += float(
                    record["weighted_compute"][seed]
                )
                raw_by_seed[point_id][seed] += int(record["raw_tokens"][seed])
                refine_cohorts_by_seed[point_id][seed] += int(
                    record["realized_refine_cohorts"][seed]
                )
                refine_actions_by_seed[point_id][seed] += int(
                    record["refine_action_count"][seed]
                )

    grid_points: list[Dict[str, Any]] = []
    anchors: list[Dict[str, Any]] = []
    for point in points:
        point_id = point.point_id
        if shard_count_by_point[point_id] != problem_count:
            raise ValueError(
                f"point {point_id} has {shard_count_by_point[point_id]} "
                f"problem shards; expected {problem_count}"
            )
        episode_count = problem_count * seed_count
        accuracy_by_seed = [
            value / problem_count for value in correct_by_seed[point_id]
        ]
        compute_mean_by_seed = [
            value / problem_count for value in compute_by_seed[point_id]
        ]
        raw_mean_by_seed = [
            value / problem_count for value in raw_by_seed[point_id]
        ]
        refine_cohorts_mean_by_seed = [
            value / problem_count for value in refine_cohorts_by_seed[point_id]
        ]
        refine_actions_mean_by_seed = [
            value / problem_count for value in refine_actions_by_seed[point_id]
        ]
        latency_batch_by_seed = [
            _percentile(values, 0.95) for values in latency_by_seed[point_id]
        ]
        all_latencies = array("d")
        for values in latency_by_seed[point_id]:
            all_latencies.extend(values)
        aggregate_metrics = {
            "accuracy": sum(correct_by_seed[point_id]) / episode_count,
            "latency_s": _percentile(all_latencies, 0.95),
            "weighted_compute": sum(compute_by_seed[point_id]) / episode_count,
            "raw_tokens": sum(raw_by_seed[point_id]) / episode_count,
            "realized_refine_cohorts": sum(
                refine_cohorts_by_seed[point_id]
            )
            / episode_count,
            "refine_action_count": sum(refine_actions_by_seed[point_id])
            / episode_count,
        }
        seed_metrics = [
            {
                "seed": seed,
                "accuracy_mean": accuracy_by_seed[seed],
                "latency_batch_s": latency_batch_by_seed[seed],
                "weighted_compute_mean": compute_mean_by_seed[seed],
                "raw_tokens_mean": raw_mean_by_seed[seed],
                "realized_refine_cohorts_mean": refine_cohorts_mean_by_seed[seed],
                "refine_action_count_mean": refine_actions_mean_by_seed[seed],
            }
            for seed in range(seed_count)
        ]
        grid_points.append(
            {
                "id": point_id,
                "configuration": asdict(point),
                "episode_count": episode_count,
                "aggregate": aggregate_metrics,
                "seed_batch_distributions": {
                    "accuracy_mean": _distribution(accuracy_by_seed),
                    "latency_batch_s": _distribution(latency_batch_by_seed),
                    "weighted_compute_mean": _distribution(compute_mean_by_seed),
                    "raw_tokens_mean": _distribution(raw_mean_by_seed),
                    "realized_refine_cohorts_mean": _distribution(
                        refine_cohorts_mean_by_seed
                    ),
                    "refine_action_count_mean": _distribution(
                        refine_actions_mean_by_seed
                    ),
                },
                "seed_metrics": seed_metrics,
            }
        )
        anchors.append(
            {
                "id": point_id,
                "source": "fixed_budget_grid_training_aggregate",
                "accuracy": aggregate_metrics["accuracy"],
                "latency_s": aggregate_metrics["latency_s"],
                "weighted_compute": aggregate_metrics["weighted_compute"],
                "metadata": {
                    **asdict(point),
                    "policy": FixedBudgetGridPolicy.NAME,
                    "problems": problem_count,
                    "shuffle_seeds": seed_count,
                    "episodes": episode_count,
                    "aggregation": {
                        "accuracy": "mean_over_problem_x_seed",
                        "latency_s": "batch_over_problem_x_seed",
                        "weighted_compute": "mean_over_problem_x_seed",
                    },
                },
            }
        )

    semantic_result = {
        "schema_version": RESULT_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "grid_point_count": len(points),
        "problem_count": problem_count,
        "seed_count": seed_count,
        "episodes_per_grid_point": problem_count * seed_count,
        "total_episode_evaluations": len(points) * problem_count * seed_count,
        "grid_points": grid_points,
    }
    result = {
        **semantic_result,
        "runtime_diagnostics": {
            "worker_shard_elapsed_s": _distribution(shard_elapsed_s),
        },
        "results_sha256": _canonical_sha256(semantic_result),
    }
    _atomic_json(output_root / "grid_points.json", result)

    anchor_set = {
        "schema_version": ANCHOR_SCHEMA,
        "scope": {
            "split": manifest["split"],
            "datasets": manifest["datasets"],
            "problem_count": problem_count,
            "shuffle_seeds": seed_count,
        },
        "aggregation": {
            "accuracy": "mean_over_problem_x_seed",
            "latency_s": "batch_over_problem_x_seed",
            "weighted_compute": "mean_over_problem_x_seed",
        },
        "execution_semantics": manifest["execution_semantics"],
        "anchors": anchors,
    }
    anchor_set["anchors_sha256"] = _canonical_sha256(anchors)
    _atomic_json(output_root / "anchors.json", anchor_set)
    return result


def run_grid(
    *,
    config: Dict[str, Any],
    output_root: Path,
    split: str,
    datasets: Sequence[str] | None,
    indices: Sequence[int] | None,
    models: Sequence[str],
    widths: Sequence[int],
    depths: Sequence[int],
    refine_rounds: Sequence[int],
    seed_count: int,
    workers: int,
    aggregate_only: bool = False,
) -> Dict[str, Any]:
    if split != "discovery":
        raise ValueError("profile calibration is restricted to the discovery split")
    if seed_count < 1 or workers < 1:
        raise ValueError("seed_count and workers must be positive")
    configured_models = {item["name"] for item in config["models"]}
    unknown_models = sorted(set(models) - configured_models)
    if unknown_models:
        raise ValueError(f"unknown configured models: {unknown_models}")
    if any(value < 1 for value in widths) or max(widths) > 128:
        raise ValueError("profile-grid widths must be in [1, 128]")
    if any(value < 1 for value in depths) or max(depths) > 66:
        raise ValueError("profile-grid depths must be in [1, 66]")
    if any(value < 0 for value in refine_rounds) or max(refine_rounds) > 5:
        raise ValueError("profile-grid refine rounds must be in [0, 5]")

    paths = _cache_paths(config, split, datasets, indices)
    points = _build_points(models, widths, depths, refine_rounds)
    cache_files = [
        {
            "dataset": path.parent.name,
            "index": int(path.stem),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "split": split,
        "datasets": list(datasets or config[f"{split}_datasets"]),
        "cache_files": cache_files,
        "models": list(models),
        "widths": list(widths),
        "depths": list(depths),
        "refine_rounds": list(refine_rounds),
        "seed_count": seed_count,
        "grid_point_count": len(points),
        "policy": FixedBudgetGridPolicy.NAME,
        "execution_semantics": {
            "model_binding": "one_model_per_episode_no_switching",
            "width": "spawn_exact_width_as_one_parallel_cohort",
            "depth": "maximum_checkpoints_per_stage",
            "self_refine": "only_stage_terminal_branches_enter_next_stage",
            "stopping": "no_early_stop_no_pruning",
            "answer": "deterministic_plurality_of_each_branch_latest_observed_answer",
            "parallel_latency": "sum_of_synchronous_phase_maxima",
        },
        "aggregation": {
            "accuracy": "mean_over_problem_x_seed",
            "latency_s": "batch_over_problem_x_seed",
            "weighted_compute": "mean_over_problem_x_seed",
            "seed": "repeated_measurement_not_a_search_dimension",
        },
    }
    manifest_sha256 = _canonical_sha256(manifest)
    manifest["manifest_sha256"] = manifest_sha256
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"profile-grid resume parameters differ from {manifest_path}"
            )
    else:
        _atomic_json(manifest_path, manifest)

    tasks = [
        GridShardTask(str(path), _task_key(path))
        for path in paths
    ]
    pending = [
        task
        for task in tasks
        if not _validated_existing_shard(
            _shard_path(output_root, task.task_key),
            task,
            manifest_sha256,
            len(points),
            seed_count,
            models,
        )
    ]
    if aggregate_only and pending:
        raise ValueError(
            f"cannot aggregate: {len(pending)} of {len(tasks)} shards are missing"
        )

    lock_path = output_root / "run.lock"
    if pending:
        _acquire_lock(lock_path)
        try:
            completed = len(tasks) - len(pending)
            _atomic_json(
                output_root / "heartbeat.json",
                {
                    "status": "running",
                    "completed_shards": completed,
                    "total_shards": len(tasks),
                    "updated_unix_s": time.time(),
                },
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(config, points, seed_count, manifest_sha256),
            ) as executor:
                futures = {executor.submit(_run_shard, task): task for task in pending}
                for future in as_completed(futures):
                    task = futures[future]
                    shard = future.result()
                    _atomic_json(_shard_path(output_root, task.task_key), shard)
                    completed += 1
                    _atomic_json(
                        output_root / "heartbeat.json",
                        {
                            "status": "running",
                            "completed_shards": completed,
                            "total_shards": len(tasks),
                            "last_completed_task": task.task_key,
                            "last_shard_elapsed_s": shard["elapsed_s"],
                            "updated_unix_s": time.time(),
                        },
                    )
        finally:
            lock_path.unlink(missing_ok=True)

    result = _aggregate(
        output_root=output_root,
        tasks=tasks,
        points=points,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        seed_count=seed_count,
    )
    _atomic_json(
        output_root / "heartbeat.json",
        {
            "status": "complete",
            "completed_shards": len(tasks),
            "total_shards": len(tasks),
            "grid_point_count": len(points),
            "total_episode_evaluations": result["total_episode_evaluations"],
            "updated_unix_s": time.time(),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate user profiles over a fixed four-dimensional resource grid"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config/experiment_aime.json"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("discovery", "heldout"), default="discovery")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--indices", default=None)
    parser.add_argument(
        "--models",
        default="Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-14B,Qwen3-32B",
    )
    parser.add_argument("--widths", default="1,4,8,16,64")
    parser.add_argument("--depths", default="1,2,4,8,16,32,66")
    parser.add_argument("--refine-rounds", default="0,1,2,3,4,5")
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_grid(
        config=config,
        output_root=Path(args.output_root).resolve(),
        split=args.split,
        datasets=(
            _csv_values(args.datasets, str) if args.datasets is not None else None
        ),
        indices=(
            _csv_values(args.indices, int) if args.indices is not None else None
        ),
        models=_csv_values(args.models, str),
        widths=_csv_values(args.widths, int),
        depths=_csv_values(args.depths, int),
        refine_rounds=_csv_values(args.refine_rounds, int),
        seed_count=int(
            args.seed_count
            if args.seed_count is not None
            else config["shuffle_seeds"]
        ),
        workers=int(
            args.workers
            if args.workers is not None
            else config.get("evaluation_workers", 4)
        ),
        aggregate_only=args.aggregate_only,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(Path(args.output_root).resolve()),
                "grid_point_count": result["grid_point_count"],
                "problem_count": result["problem_count"],
                "seed_count": result["seed_count"],
                "total_episode_evaluations": result["total_episode_evaluations"],
                "results_sha256": result["results_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
