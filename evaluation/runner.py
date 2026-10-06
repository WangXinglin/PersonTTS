from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sqlite3
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type

from config_utils import load_config, model_weights, resolve_from_root, validate_evaluation_workers
from evaluation.answering import is_correct, normalize_prediction
from evaluation.trace_analysis import (
    PUBLIC_TRACE_FILENAME,
    TRACE_SUMMARY_FILENAME,
    public_trace_record,
    write_trace_summary,
)
from evaluation.trace_sanitizer import anonymize_full_trace
from evaluation.requirements import (
    RequirementProfile,
    evaluate_seed_batch_utility,
    load_requirement_profiles,
)
from replay.adapter import load_problem
from replay.environment import ReplayEnvironment


def _load_policy_class(path: Path):
    spec = importlib.util.spec_from_file_location("autoresearch_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy_class = getattr(module, "CandidatePolicy", None)
    if policy_class is None:
        raise RuntimeError(f"CandidatePolicy is missing from {path}")
    return policy_class


def _load_importable_policy(module_name: str, qualname: str) -> Type[Any]:
    value: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def _cache_paths(config: Dict[str, Any], split: str) -> List[Path]:
    if split not in {"discovery", "heldout"}:
        raise ValueError("split must be discovery or heldout")
    cache_root = resolve_from_root(config, config["private_cache_root"])
    paths: List[Path] = []
    for dataset in config[f"{split}_datasets"]:
        paths.extend(sorted((cache_root / dataset).glob("*.json"), key=lambda path: int(path.stem)))
    if not paths:
        raise RuntimeError(f"No compact replay files found for {split} under {cache_root}")
    return paths


def _record_key(dataset: str, index: int, beta: float, seed: int) -> str:
    return f"{dataset}:{index}:b{beta:.4f}:s{seed}"


def _profile_record_key(
    dataset: str,
    index: int,
    profile_id: str,
    profile_hash: str,
    seed: int,
) -> str:
    return f"{dataset}:{index}:p{profile_id}:h{profile_hash}:s{seed}"


class _CheckpointIndex:
    def __init__(self) -> None:
        self.connection = sqlite3.connect("")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE records (record_key TEXT PRIMARY KEY, episode INTEGER, trace INTEGER)"
        )

    def close(self) -> None:
        self.connection.close()

    def load(self, path: Path, *, kind: str) -> None:
        if kind not in {"episode", "trace"}:
            raise ValueError(f"unsupported checkpoint kind: {kind}")
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                key = str(json.loads(line)["record_key"])
                self.connection.execute(
                    f"INSERT INTO records (record_key, episode, trace) VALUES (?, ?, ?) "
                    f"ON CONFLICT(record_key) DO UPDATE SET {kind} = 1",
                    (key, int(kind == "episode"), int(kind == "trace")),
                )
        self.connection.commit()

    def contains(self, key: str, *, kind: str) -> bool:
        if kind not in {"episode", "trace"}:
            raise ValueError(f"unsupported checkpoint kind: {kind}")
        row = self.connection.execute(
            f"SELECT {kind} FROM records WHERE record_key = ?", (key,)
        ).fetchone()
        return bool(row and row[0])

    def mark(self, key: str, *, kind: str) -> None:
        if kind not in {"episode", "trace"}:
            raise ValueError(f"unsupported checkpoint kind: {kind}")
        self.connection.execute(
            f"INSERT INTO records (record_key, episode, trace) VALUES (?, ?, ?) "
            f"ON CONFLICT(record_key) DO UPDATE SET {kind} = 1",
            (key, int(kind == "episode"), int(kind == "trace")),
        )

    def missing_trace_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM records WHERE episode = 1 AND trace != 1"
            ).fetchone()[0]
        )


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_or_validate_manifest(path: Path, value: Dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(
                f"evaluation resume parameters differ from {path}: "
                f"recorded={existing}, requested={value}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    connection = sqlite3.connect("")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE episodes (dataset TEXT, beta REAL, correct INTEGER, "
        "raw_tokens REAL, weighted_compute REAL)"
    )
    try:
        for record in records:
            connection.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
                (
                    str(record["dataset"]),
                    float(record["beta"]),
                    int(record["correct"]),
                    float(record["cost"]["raw_tokens"]),
                    float(record["cost"]["weighted_compute"]),
                ),
            )
        connection.commit()

        def summarize(where: str, parameters: tuple[Any, ...]) -> Dict[str, Any]:
            count, correct = connection.execute(
                f"SELECT COUNT(*), SUM(correct) FROM episodes {where}", parameters
            ).fetchone()
            if not count:
                raise ValueError("cannot summarize an empty evaluation group")

            def ordered_sum(column: str) -> float:
                if column not in {"raw_tokens", "weighted_compute"}:
                    raise ValueError(f"unsupported summary column: {column}")
                return sum(
                    float(row[0])
                    for row in connection.execute(
                        f"SELECT {column} FROM episodes {where} ORDER BY rowid",
                        parameters,
                    )
                )

            return {
                "episodes": int(count),
                "accuracy": int(correct) / int(count),
                "avg_raw_tokens": ordered_sum("raw_tokens") / int(count),
                "avg_weighted_compute": ordered_sum("weighted_compute") / int(count),
            }

        betas = [
            float(row[0])
            for row in connection.execute("SELECT DISTINCT beta FROM episodes ORDER BY beta")
        ]
        groups = [
            (str(dataset), float(beta))
            for dataset, beta in connection.execute(
                "SELECT DISTINCT dataset, beta FROM episodes ORDER BY dataset, beta"
            )
        ]
        return {
            "overall": {
                f"beta={beta:.1f}": summarize("WHERE beta = ?", (beta,))
                for beta in betas
            },
            "by_dataset": {
                f"{dataset}|beta={beta:.1f}": summarize(
                    "WHERE dataset = ? AND beta = ?", (dataset, beta)
                )
                for dataset, beta in groups
            },
        }
    finally:
        connection.close()


@dataclass(frozen=True)
class EpisodeTask:
    cache_path: str
    record_key: str
    beta: Optional[float]
    seed: int
    profile_id: Optional[str] = None
    profile_hash: Optional[str] = None
    requirements: Optional[Dict[str, Any]] = None


def _evaluate_episode(
    task: EpisodeTask,
    *,
    problem: Any,
    policy_class: Type[Any],
    base_policy_config: Dict[str, Any],
    weights: Dict[str, float],
    project_root: Path,
    max_weighted_compute: Optional[float],
) -> Dict[str, Any]:
    def normalize_answer(value: Any) -> Optional[str]:
        normalized, is_null = normalize_prediction(value, problem.dataset, project_root)
        return None if is_null else normalized

    personalized = task.requirements is not None
    env = ReplayEnvironment(
        problem,
        weights,
        seed=task.seed,
        beta=task.beta,
        requirements=task.requirements,
        max_weighted_compute=max_weighted_compute,
        answer_normalizer=normalize_answer,
    )
    instance_config = dict(base_policy_config)
    if personalized:
        instance_config.pop("beta", None)
        instance_config.pop("budget_beta", None)
    else:
        if task.beta is None:
            raise RuntimeError("legacy episode task is missing beta")
        instance_config["beta"] = task.beta
    if task.requirements is not None:
        instance_config["requirements"] = task.requirements
    policy = policy_class(config=instance_config)
    prediction = policy.solve(env)
    if not env.trace or env.trace[-1].get("action") != "ANSWER":
        prediction = env.finish(prediction)
    correct = is_correct(prediction, problem.gold_answer, problem.dataset, project_root)
    # Every evaluated episode contributes a sanitized executed trace. The full
    # private trajectory, question text, gold answer, and future checkpoints are
    # never copied into this record.
    public_trace = anonymize_full_trace(env.trace)
    record = {
        "record_key": task.record_key,
        "dataset": problem.dataset,
        "index": problem.index,
        "seed": task.seed,
        "correct": correct,
        "prediction": prediction,
        "cost": env.cost,
        "policy_result": getattr(policy, "last_result", None),
        "trace_steps": len(env.trace),
        "public_trace": public_trace,
    }
    if task.profile_id is not None:
        record["profile_id"] = task.profile_id
        record["profile_hash"] = task.profile_hash
    else:
        if task.beta is None:
            raise RuntimeError("legacy episode task is missing beta")
        record["beta"] = task.beta
    return record


_WORKER_POLICY_CLASS: Optional[Type[Any]] = None
_WORKER_POLICY_CONFIG: Dict[str, Any] = {}
_WORKER_WEIGHTS: Dict[str, float] = {}
_WORKER_PROJECT_ROOT = Path(".")
_WORKER_MAX_WEIGHTED_COMPUTE: Optional[float] = None


@lru_cache(maxsize=2)
def _load_worker_problem(cache_path: str):
    return load_problem(Path(cache_path))


def _initialize_episode_worker(
    config: Dict[str, Any],
    policy_path: Optional[str],
    policy_module: Optional[str],
    policy_qualname: Optional[str],
    policy_config: Dict[str, Any],
    max_weighted_compute: Optional[float],
) -> None:
    global _WORKER_POLICY_CLASS
    global _WORKER_POLICY_CONFIG
    global _WORKER_WEIGHTS
    global _WORKER_PROJECT_ROOT
    global _WORKER_MAX_WEIGHTED_COMPUTE

    if policy_path is not None:
        _WORKER_POLICY_CLASS = _load_policy_class(Path(policy_path))
    else:
        assert policy_module is not None and policy_qualname is not None
        _WORKER_POLICY_CLASS = _load_importable_policy(policy_module, policy_qualname)
    _WORKER_POLICY_CONFIG = policy_config
    _WORKER_WEIGHTS = model_weights(config)
    _WORKER_PROJECT_ROOT = Path(config["_root"]).parent.resolve()
    _WORKER_MAX_WEIGHTED_COMPUTE = max_weighted_compute
    _load_worker_problem.cache_clear()


def _run_worker_episode(task: EpisodeTask) -> Dict[str, Any]:
    if _WORKER_POLICY_CLASS is None:
        raise RuntimeError("episode worker was not initialized")
    return _evaluate_episode(
        task,
        problem=_load_worker_problem(task.cache_path),
        policy_class=_WORKER_POLICY_CLASS,
        base_policy_config=_WORKER_POLICY_CONFIG,
        weights=_WORKER_WEIGHTS,
        project_root=_WORKER_PROJECT_ROOT,
        max_weighted_compute=_WORKER_MAX_WEIGHTED_COMPUTE,
    )


def _build_tasks(
    *,
    paths: List[Path],
    betas: List[float],
    seeds: Tuple[int, ...],
    checkpoints: _CheckpointIndex,
    requirement_profiles: Optional[Sequence[RequirementProfile]] = None,
) -> Iterator[EpisodeTask]:
    for cache_path in paths:
        problem = load_problem(cache_path)
        if requirement_profiles is not None:
            for profile in requirement_profiles:
                requirements = profile.policy_payload()
                for seed in seeds:
                    key = _profile_record_key(
                        problem.dataset,
                        problem.index,
                        profile.profile_id,
                        profile.fingerprint,
                        seed,
                    )
                    if not checkpoints.contains(key, kind="episode"):
                        yield EpisodeTask(
                            cache_path=str(cache_path),
                            record_key=key,
                            beta=None,
                            seed=seed,
                            profile_id=profile.profile_id,
                            profile_hash=profile.fingerprint,
                            requirements=requirements,
                        )
        else:
            for beta in betas:
                for seed in seeds:
                    key = _record_key(problem.dataset, problem.index, beta, seed)
                    if checkpoints.contains(key, kind="episode"):
                        continue
                    yield EpisodeTask(
                        cache_path=str(cache_path),
                        record_key=key,
                        beta=beta,
                        seed=seed,
                    )


def _bounded_batches(items: Iterable[EpisodeTask], size: int) -> Iterator[List[EpisodeTask]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def _policy_worker_reference(
    policy_path: Optional[Path], policy_class: Optional[Type[Any]]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if policy_path is not None:
        return str(policy_path.resolve()), None, None
    assert policy_class is not None
    module_name = policy_class.__module__
    qualname = policy_class.__qualname__
    if "<locals>" in qualname:
        raise ValueError("parallel evaluation requires an importable policy class; use workers=1")
    return None, module_name, qualname


def evaluate(
    *,
    config: Dict[str, Any],
    output_dir: Path,
    split: str,
    policy_path: Optional[Path] = None,
    policy_class: Optional[Type[Any]] = None,
    policy_config: Optional[Dict[str, Any]] = None,
    seed_count: Optional[int] = None,
    seeds: Optional[Sequence[int]] = None,
    beta_grid: Optional[List[float]] = None,
    max_weighted_compute: Optional[float] = None,
    workers: Optional[int] = None,
    requirement_profiles: Optional[Sequence[RequirementProfile]] = None,
) -> Dict[str, Any]:
    if split == "heldout" and "discovery" in str(output_dir).lower():
        raise ValueError("Held-out output must not be written into discovery history")
    if (policy_path is None) == (policy_class is None):
        raise ValueError("provide exactly one of policy_path or policy_class")
    effective_workers = validate_evaluation_workers(
        workers if workers is not None else config.get("evaluation_workers", 1)
    )

    paths = _cache_paths(config, split)
    if seed_count is not None and seeds is not None:
        raise ValueError("seed_count and explicit seeds are mutually exclusive")
    if seeds is None:
        evaluated_seeds = tuple(
            range(int(seed_count if seed_count is not None else config["shuffle_seeds"]))
        )
    else:
        evaluated_seeds = tuple(int(seed) for seed in seeds)
        if not evaluated_seeds or any(seed < 0 for seed in evaluated_seeds):
            raise ValueError("explicit seeds must be non-empty and non-negative")
        if len(evaluated_seeds) != len(set(evaluated_seeds)):
            raise ValueError("explicit seeds must not contain duplicates")
    profiles = list(requirement_profiles) if requirement_profiles is not None else None
    if profiles is not None and not profiles:
        raise ValueError("requirement_profiles must not be empty")
    if profiles is not None and beta_grid is not None:
        raise ValueError("personalized profile evaluation cannot also specify beta_grid")
    betas = (
        []
        if profiles is not None
        else list(beta_grid if beta_grid is not None else config["beta_grid"])
    )
    weights = model_weights(config)
    project_root = Path(config["_root"]).parent.resolve()
    problem_keys: List[str] = []
    missing_latency: List[str] = []
    for path in paths:
        problem = load_problem(path)
        key = f"{problem.dataset}:{problem.index}"
        problem_keys.append(key)
        if profiles is not None and not problem.latency_available:
            missing_latency.append(key)
    expected_count = config.get("expected_problem_count_by_split", {}).get(split)
    if expected_count is not None and len(problem_keys) != int(expected_count):
        raise ValueError(
            f"{split} evaluation requires exactly {int(expected_count)} cached problems; "
            f"found {len(problem_keys)}"
        )
    if profiles is not None:
        if missing_latency:
            raise ValueError(
                "personalized evaluation requires latency-aware cache v2; "
                f"outdated problems={missing_latency[:10]}"
            )
    evaluation_manifest = {
        "schema_version": "person_tts_evaluation_manifest_v1",
        "mode": "personalized_hard_requirements" if profiles is not None else "legacy_beta",
        "split": split,
        "problem_keys": problem_keys,
        "seeds": list(evaluated_seeds),
        "beta_grid": None if profiles is not None else betas,
        "profiles": (
            [
                {"id": profile.profile_id, "sha256": profile.fingerprint}
                for profile in profiles
            ]
            if profiles is not None
            else None
        ),
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    # Migrate the inert compatibility field from personalized checkpoints so
    # resume identity depends only on actual personalized experiment inputs.
    if profiles is not None and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "personalized_legacy_beta" in existing_manifest:
            existing_manifest.pop("personalized_legacy_beta")
            temporary = manifest_path.with_suffix(
                manifest_path.suffix + f".tmp.{os.getpid()}"
            )
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(existing_manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
    _write_or_validate_manifest(manifest_path, evaluation_manifest)
    staging_dir: Optional[Path] = None
    try:
        if effective_workers > 1:
            staging_dir = Path(tempfile.mkdtemp(prefix="eval_staging_"))
            local_cache = staging_dir / "cache"
            eval_paths: List[Path] = []
            for orig in paths:
                dst = local_cache / orig.parent.name / orig.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(orig, dst)
                eval_paths.append(dst)
            local_output = staging_dir / "output"
            local_output.mkdir()
        else:
            eval_paths = paths

        final_raw_path = output_dir / "episodes.jsonl"
        final_trace_path = output_dir / PUBLIC_TRACE_FILENAME
        raw_path = (staging_dir / "output" / "episodes.jsonl") if staging_dir else final_raw_path
        trace_path = (staging_dir / "output" / PUBLIC_TRACE_FILENAME) if staging_dir else final_trace_path

        checkpoints = _CheckpointIndex()
        try:
            checkpoints.load(final_raw_path, kind="episode")
            checkpoints.load(final_trace_path, kind="trace")
            missing_traces = checkpoints.missing_trace_count()
            if missing_traces:
                raise RuntimeError(
                    "Existing episode checkpoints predate full trace capture; use a new "
                    f"output directory to regenerate {missing_traces} missing traces"
                )
            tasks = _build_tasks(
                paths=eval_paths,
                betas=betas,
                seeds=evaluated_seeds,
                checkpoints=checkpoints,
                requirement_profiles=profiles,
            )
            instance_config = dict(config.get("candidate_policy_config", {}))
            instance_config.update(policy_config or {})

            def persist(record: Dict[str, Any]) -> None:
                key = record["record_key"]
                if not checkpoints.contains(key, kind="trace"):
                    _append_jsonl(trace_path, public_trace_record(record))
                    checkpoints.mark(key, kind="trace")
                episode_record = dict(record)
                episode_record.pop("public_trace", None)
                _append_jsonl(raw_path, episode_record)
                checkpoints.mark(key, kind="episode")

            if effective_workers == 1:
                loaded_policy_class = policy_class or _load_policy_class(policy_path.resolve())  # type: ignore[union-attr]
                current_cache_path: Optional[str] = None
                current_problem: Any = None
                for task in tasks:
                    if task.cache_path != current_cache_path:
                        current_problem = load_problem(Path(task.cache_path))
                        current_cache_path = task.cache_path
                    persist(
                        _evaluate_episode(
                            task,
                            problem=current_problem,
                            policy_class=loaded_policy_class,
                            base_policy_config=instance_config,
                            weights=weights,
                            project_root=project_root,
                            max_weighted_compute=max_weighted_compute,
                        )
                    )
            else:
                policy_file, policy_module, policy_qualname = _policy_worker_reference(
                    policy_path, policy_class
                )
                with ProcessPoolExecutor(
                    max_workers=effective_workers,
                    initializer=_initialize_episode_worker,
                    initargs=(
                        config,
                        policy_file,
                        policy_module,
                        policy_qualname,
                        instance_config,
                        max_weighted_compute,
                    ),
                ) as executor:
                    submission_window = max(1, effective_workers * 4)
                    for task_batch in _bounded_batches(tasks, submission_window):
                        for record in executor.map(
                            _run_worker_episode, task_batch, chunksize=1
                        ):
                            persist(record)
        finally:
            checkpoints.close()

        if staging_dir is not None:
            for local_p, final_p in [(raw_path, final_raw_path), (trace_path, final_trace_path)]:
                if local_p.is_file():
                    final_p.parent.mkdir(parents=True, exist_ok=True)
                    with local_p.open("rb") as src, final_p.open("ab") as dst:
                        shutil.copyfileobj(src, dst)

        for p in (final_raw_path, final_trace_path):
            if p.is_file():
                with p.open("rb") as fh:
                    os.fsync(fh.fileno())

        if profiles is None:
            summary = aggregate(_iter_jsonl(final_raw_path))
            utility_detail = None
        else:
            utility_detail = evaluate_seed_batch_utility(
                _iter_jsonl(final_raw_path),
                profiles,
                expected_problem_keys=problem_keys,
                expected_seeds=evaluated_seeds,
            )
            summary = {
                key: value for key, value in utility_detail.items() if key != "batches"
            }
            summary["mode"] = "personalized_hard_requirements"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if utility_detail is not None:
            (output_dir / "batch_utility.json").write_text(
                json.dumps(utility_detail, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        trace_summary = write_trace_summary(
            final_trace_path,
            output_dir / TRACE_SUMMARY_FILENAME,
        )
        public_feedback = {
            "summary": summary,
            "trace_access": {
                "mode": "full_anonymized_executed_traces",
                "records": trace_summary["all"]["episodes"],
                "workspace_trace_file": "SEARCH_TRACES.jsonl",
                "workspace_summary_file": "TRACE_SUMMARY.json",
                "workspace_analyzer": "TRACE_ANALYZER.py",
            },
        }
        (output_dir / "public_feedback.json").write_text(
            json.dumps(public_feedback, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one controller over compact replay data")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / "config/experiment_aime.json")
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["discovery", "heldout"], default="discovery")
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Explicit comma-separated replay seeds")
    parser.add_argument("--betas", default=None)
    parser.add_argument(
        "--profiles",
        default=None,
        help="Requirement profile JSON; mutually exclusive with --betas",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Episode worker processes (default: config evaluation_workers or 1)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.profiles and args.betas:
        parser.error("--profiles and --betas are mutually exclusive")
    if args.seed_count is not None and args.seeds:
        parser.error("--seed-count and --seeds are mutually exclusive")
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else None
    betas = [float(value) for value in args.betas.split(",")] if args.betas else None
    profiles = load_requirement_profiles(Path(args.profiles).resolve()) if args.profiles else None
    summary = evaluate(
        config=config,
        policy_path=Path(args.policy).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        split=args.split,
        seed_count=args.seed_count,
        seeds=seeds,
        beta_grid=betas,
        workers=args.workers,
        requirement_profiles=profiles,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
