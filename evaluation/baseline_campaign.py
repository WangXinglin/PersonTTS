from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config_utils import load_config
from evaluation.cache_inventory import verify_cache_inventory
from discovery.profile_campaign import (
    CampaignLayout,
    Claim,
    ClaimHeartbeat,
    QUEUE_STATES,
    _atomic_json,
    _bucket,
    _count_json_files,
    _mkdir_lock,
    _now,
    _read_json,
    _safe_component,
    _sha256,
    _touch_active,
    claim_task,
    complete_claim,
    fail_claim,
    prepare_retry_backoff,
    reap_stale_claims,
    retry_failed_tasks,
    wait_for_retry_backoff,
)
from policies.seeds import BASELINE_REGISTRY


BASELINE_SPECS_SCHEMA = "person_tts_baseline_specs_v1"
BASELINE_CAMPAIGN_SCHEMA = "person_tts_baseline_campaign_v1"
BASELINE_TASK_SCHEMA = "person_tts_baseline_seed_task_v1"
BASELINE_RESULT_SCHEMA = "person_tts_baseline_seed_result_v1"
BASELINE_INDEX_SCHEMA = "person_tts_baseline_campaign_index_v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_baseline_specs(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != BASELINE_SPECS_SCHEMA:
        raise ValueError(f"baseline specs must use schema {BASELINE_SPECS_SCHEMA!r}")
    values = raw.get("baselines")
    if not isinstance(values, list) or not values:
        raise ValueError("baseline specs must contain a non-empty baselines list")
    specs: List[Dict[str, Any]] = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each baseline spec must be an object")
        baseline_id = _safe_component(str(value.get("id", "")), name="baseline_id")
        if baseline_id in seen:
            raise ValueError(f"duplicate baseline id: {baseline_id}")
        seen.add(baseline_id)
        algorithm = str(value.get("algorithm", ""))
        if algorithm not in BASELINE_REGISTRY:
            raise ValueError(f"unknown baseline algorithm: {algorithm!r}")
        policy_config = value.get("policy_config")
        if not isinstance(policy_config, dict):
            raise ValueError(f"baseline {baseline_id!r} is missing policy_config")
        BASELINE_REGISTRY[algorithm](config=policy_config)
        normalized = {
            "id": baseline_id,
            "algorithm": algorithm,
            "policy_config": policy_config,
        }
        normalized["spec_sha256"] = _canonical_sha256(normalized)
        specs.append(normalized)
    return specs


def baseline_campaign_settings(
    *,
    config_path: Path,
    specs_path: Path,
    campaign_id: str,
    seed_count: int,
    bucket_count: int,
    max_attempts: int,
    evaluation_workers: int,
    split: str = "discovery",
) -> Dict[str, Any]:
    config = load_config(config_path)
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    if bucket_count < 1 or bucket_count > 4096:
        raise ValueError("bucket_count must be in [1, 4096]")
    if max_attempts < 1 or evaluation_workers < 1:
        raise ValueError("max_attempts and evaluation_workers must be positive")
    if split not in {"discovery", "heldout"}:
        raise ValueError("split must be discovery or heldout")
    return {
        "campaign_id": _safe_component(campaign_id, name="campaign_id"),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "specs_path": str(specs_path.resolve()),
        "specs_sha256": _sha256(specs_path.resolve()),
        "split": split,
        "seed_count": seed_count,
        "bucket_count": bucket_count,
        "max_attempts": max_attempts,
        "evaluation_workers": evaluation_workers,
        # Legacy baseline_runner still transports beta, but this fixed value is
        # independent of personalized requirement-profile discovery.
        "legacy_beta": 0.5,
        "expected_problem_count": int(
            config.get("expected_problem_count_by_split", {}).get(split, 0)
        ),
    }


def _task_id(baseline_id: str, seed: int) -> str:
    return f"{baseline_id}--seed-{seed:04d}"


def initialize_baseline_campaign(
    *,
    campaign_root: Path,
    settings: Dict[str, Any],
    cache_inventory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    layout = CampaignLayout(campaign_root.resolve())
    specs = load_baseline_specs(Path(settings["specs_path"]))
    layout.root.mkdir(parents=True, exist_ok=True)
    for state in QUEUE_STATES:
        (layout.root / "queue" / state).mkdir(parents=True, exist_ok=True)
    for state in ("pending", "running"):
        layout.active(state).mkdir(parents=True, exist_ok=True)

    with _mkdir_lock(layout.root / "locks" / "initialize.lock"):
        manifest = {
            "schema_version": BASELINE_CAMPAIGN_SCHEMA,
            "created_at": _now(),
            "baselines": specs,
            **settings,
        }
        if cache_inventory is not None:
            cache_sha256 = str(cache_inventory["semantic_sha256"])
            manifest["cache_inventory_sha256"] = cache_sha256
            if layout.cache_inventory.is_file():
                recorded_inventory = _read_json(layout.cache_inventory)
                if recorded_inventory.get("semantic_sha256") != cache_sha256:
                    raise ValueError("replay cache inventory differs from immutable campaign")
            else:
                _atomic_json(layout.cache_inventory, cache_inventory)
        if layout.manifest.is_file():
            existing = _read_json(layout.manifest)
            if "compatibility_beta" in existing and "legacy_beta" not in existing:
                existing["legacy_beta"] = existing.pop("compatibility_beta")
                _atomic_json(layout.manifest, existing)
            comparable = {key: value for key, value in existing.items() if key != "created_at"}
            requested = {key: value for key, value in manifest.items() if key != "created_at"}
            if comparable != requested:
                raise ValueError("baseline campaign settings differ from immutable manifest")
        else:
            _atomic_json(layout.manifest, manifest)

        added: List[str] = []
        existing_ids: List[str] = []
        for spec in specs:
            for seed in range(int(settings["seed_count"])):
                task_id = _task_id(spec["id"], seed)
                bucket = _bucket(task_id, int(settings["bucket_count"]))
                registry_path = layout.registry(bucket, task_id)
                task = {
                    "schema_version": BASELINE_TASK_SCHEMA,
                    "profile_id": task_id,
                    "task_id": task_id,
                    "baseline_id": spec["id"],
                    "algorithm": spec["algorithm"],
                    "spec_sha256": spec["spec_sha256"],
                    "seed": seed,
                    "bucket": bucket,
                    "attempts": 0,
                    "created_at": _now(),
                }
                if registry_path.is_file():
                    if _read_json(registry_path) != task:
                        recorded = _read_json(registry_path)
                        stable_keys = (
                            "schema_version",
                            "profile_id",
                            "task_id",
                            "baseline_id",
                            "algorithm",
                            "spec_sha256",
                            "seed",
                            "bucket",
                            "attempts",
                        )
                        if any(recorded.get(key) != task.get(key) for key in stable_keys):
                            raise ValueError(f"registered task changed: {task_id}")
                    existing_ids.append(task_id)
                    continue
                with _mkdir_lock(layout.bucket_lock("pending", bucket)):
                    _atomic_json(registry_path, task)
                    _atomic_json(layout.queue("pending", bucket) / f"{task_id}.json", task)
                    _touch_active(layout, "pending", bucket)
                added.append(task_id)
    return {
        "campaign_id": settings["campaign_id"],
        "cache_inventory_sha256": manifest.get("cache_inventory_sha256"),
        "baselines": len(specs),
        "seeds": int(settings["seed_count"]),
        "tasks": len(specs) * int(settings["seed_count"]),
        "added": len(added),
        "already_present": len(existing_ids),
    }


def _verify_baseline_inputs(
    layout: CampaignLayout, manifest: Dict[str, Any]
) -> Dict[str, Any]:
    required = {"config_path", "config_sha256", "specs_path", "specs_sha256"}
    if not required.issubset(manifest):
        return {"legacy_unfrozen": True}
    config_path = Path(str(manifest["config_path"])).resolve()
    specs_path = Path(str(manifest["specs_path"])).resolve()
    if _sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("experiment config changed after baseline campaign initialization")
    if _sha256(specs_path) != manifest["specs_sha256"]:
        raise ValueError("baseline specs changed after campaign initialization")
    report: Dict[str, Any] = {
        "config_sha256": manifest["config_sha256"],
        "specs_sha256": manifest["specs_sha256"],
    }
    inventory_sha256 = manifest.get("cache_inventory_sha256")
    if inventory_sha256 is not None:
        if not layout.cache_inventory.is_file():
            raise RuntimeError("baseline campaign cache inventory is missing")
        inventory = _read_json(layout.cache_inventory)
        if inventory.get("semantic_sha256") != inventory_sha256:
            raise ValueError("baseline cache inventory fingerprint mismatch")
        report["cache"] = verify_cache_inventory(load_config(config_path), inventory)
        report["cache_inventory_sha256"] = inventory_sha256
    return report


def _spec_by_id(manifest: Dict[str, Any], baseline_id: str) -> Dict[str, Any]:
    for spec in manifest["baselines"]:
        if spec["id"] == baseline_id:
            return spec
    raise ValueError(f"task references unknown baseline: {baseline_id}")


def _episode_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def execute_baseline_claim(
    *,
    claim: Claim,
    manifest: Dict[str, Any],
    heartbeat_interval_seconds: float,
    shard_timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    _verify_baseline_inputs(claim.layout, manifest)
    task_id = str(claim.task["task_id"])
    baseline_id = str(claim.task["baseline_id"])
    seed = int(claim.task["seed"])
    spec = _spec_by_id(manifest, baseline_id)
    if claim.task["spec_sha256"] != spec["spec_sha256"]:
        raise ValueError(f"spec fingerprint mismatch for {task_id}")
    output_dir = (
        claim.layout.root
        / "results"
        / "shards"
        / baseline_id
        / f"seed_{seed:04d}"
    )
    progress = {
        "task_id": task_id,
        "baseline_id": baseline_id,
        "seed": seed,
        "status": "running",
        "worker_id": claim.worker_id,
        "attempt": claim.task["attempts"],
        "started_at": _now(),
        "output_dir": str(output_dir),
    }
    log_path = (
        claim.layout.root
        / "logs"
        / f"{task_id}.attempt_{claim.task['attempts']}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    progress["log_path"] = str(log_path)
    _atomic_json(claim.layout.progress(task_id), progress)
    project_root = Path(manifest["config_path"]).resolve().parent.parent
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_baseline_shard.py"),
        "--config",
        str(Path(manifest["config_path"]).resolve()),
        "--specs",
        str(Path(manifest["specs_path"]).resolve()),
        "--baseline-id",
        baseline_id,
        "--seed",
        str(seed),
        "--split",
        str(manifest["split"]),
        "--output-dir",
        str(output_dir),
        "--beta",
        str(manifest.get("legacy_beta", manifest.get("compatibility_beta", 0.5))),
        "--workers",
        str(manifest["evaluation_workers"]),
    ]
    heartbeat = ClaimHeartbeat(claim, heartbeat_interval_seconds)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        heartbeat.start()
        try:
            while process.poll() is None:
                if heartbeat.lost_event.wait(1.0):
                    process.terminate()
                    raise RuntimeError("claim heartbeat failed during baseline evaluation")
                if (
                    shard_timeout_seconds is not None
                    and time.monotonic() - started > shard_timeout_seconds
                ):
                    process.terminate()
                    raise TimeoutError(
                        f"baseline shard exceeded {shard_timeout_seconds}s; log={log_path}"
                    )
            if process.returncode != 0:
                raise RuntimeError(
                    f"baseline subprocess exited with code {process.returncode}; log={log_path}"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
                if process.poll() is None:
                    process.kill()
            heartbeat.stop()

    _verify_baseline_inputs(claim.layout, manifest)
    summary = _read_json(output_dir / "summary.json")

    episodes_path = output_dir / "episodes.jsonl"
    expected = int(manifest["expected_problem_count"])
    count = _episode_count(episodes_path)
    if expected and count != expected:
        raise RuntimeError(f"expected {expected} episodes for {task_id}, found {count}")
    result = {
        "schema_version": BASELINE_RESULT_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "task_id": task_id,
        "baseline_id": baseline_id,
        "algorithm": spec["algorithm"],
        "spec_sha256": spec["spec_sha256"],
        "seed": seed,
        "episode_count": count,
        "output_dir": str(output_dir),
        "episodes_path": str(episodes_path),
        "episodes_sha256": _sha256(episodes_path),
        "summary_path": str(output_dir / "summary.json"),
        "summary": summary,
        "completed_at": _now(),
    }
    _atomic_json(claim.layout.result(task_id), result)
    _atomic_json(
        claim.layout.progress(task_id),
        {
            **progress,
            "status": "complete",
            "completed_at": _now(),
            "result_path": str(claim.layout.result(task_id)),
        },
    )
    return result


def run_baseline_worker(
    *,
    campaign_root: Path,
    worker_id: Optional[str] = None,
    once: bool = False,
    idle_exit_seconds: float = 120.0,
    heartbeat_interval_seconds: float = 30.0,
    lease_timeout_seconds: float = 300.0,
    shard_timeout_seconds: Optional[float] = None,
    max_bucket_probes: int = 32,
    max_task_probes: int = 16,
    reap_bucket_probes: int = 4,
    reap_task_probes: int = 8,
    retry_backoff_base_seconds: float = 60.0,
    retry_backoff_max_seconds: float = 900.0,
) -> Dict[str, Any]:
    layout = CampaignLayout(campaign_root.resolve())
    manifest = _read_json(layout.manifest)
    if manifest.get("schema_version") != BASELINE_CAMPAIGN_SCHEMA:
        raise ValueError("unsupported baseline campaign manifest")
    _verify_baseline_inputs(layout, manifest)
    identity = _safe_component(
        worker_id or f"{socket.gethostname()}.{os.getpid()}", name="worker_id"
    )
    if (
        retry_backoff_base_seconds < 0
        or retry_backoff_max_seconds < retry_backoff_base_seconds
    ):
        raise ValueError("retry backoff requires 0 <= base <= max")
    completed = failures = claim_sequence = 0
    idle_started: Optional[float] = None
    while True:
        reap_stale_claims(
            layout=layout,
            lease_timeout_seconds=lease_timeout_seconds,
            max_attempts=int(manifest["max_attempts"]),
            max_bucket_probes=reap_bucket_probes,
            max_task_probes=reap_task_probes,
            reaper_id=identity,
            reap_sequence=claim_sequence,
        )
        claim, _ = claim_task(
            layout=layout,
            worker_id=identity,
            claim_sequence=claim_sequence,
            max_bucket_probes=max_bucket_probes,
            max_task_probes=max_task_probes,
        )
        claim_sequence += 1
        if claim is None:
            if once:
                break
            if idle_started is None:
                idle_started = time.monotonic()
            if time.monotonic() - idle_started >= idle_exit_seconds:
                break
            time.sleep(min(5.0, max(0.2, idle_exit_seconds / 10.0)))
            continue
        idle_started = None
        try:
            result = execute_baseline_claim(
                claim=claim,
                manifest=manifest,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                shard_timeout_seconds=shard_timeout_seconds,
            )
            complete_claim(
                claim,
                {
                    "result_path": str(layout.result(str(claim.task["task_id"]))),
                    "episodes_sha256": result["episodes_sha256"],
                },
            )
            completed += 1
        except Exception as exc:
            failures += 1
            error = {"type": type(exc).__name__, "message": str(exc), "timestamp": _now()}
            will_retry = int(claim.task["attempts"]) < int(manifest["max_attempts"])
            delay = not_before = None
            if will_retry:
                delay, not_before = prepare_retry_backoff(
                    claim,
                    error,
                    base_seconds=retry_backoff_base_seconds,
                    max_seconds=retry_backoff_max_seconds,
                )
            _atomic_json(
                layout.progress(str(claim.task["task_id"])),
                {
                    "task_id": claim.task["task_id"],
                    "baseline_id": claim.task["baseline_id"],
                    "seed": claim.task["seed"],
                    "status": "retry_backoff" if will_retry else "failed",
                    "worker_id": identity,
                    "attempt": claim.task["attempts"],
                    "error": error,
                    "retry_backoff_seconds": delay,
                    "retry_not_before_epoch": not_before,
                },
            )
            if will_retry:
                wait_for_retry_backoff(
                    claim,
                    delay_seconds=float(delay),
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                )
            fail_claim(claim, error, max_attempts=int(manifest["max_attempts"]))
        if once:
            break
    return {
        "worker_id": identity,
        "completed": completed,
        "failures": failures,
        "claim_cycles": claim_sequence,
    }


def baseline_campaign_status(campaign_root: Path, *, details_limit: int = 20) -> Dict[str, Any]:
    if details_limit < 0:
        raise ValueError("details_limit must be non-negative")
    layout = CampaignLayout(campaign_root.resolve())
    manifest = _read_json(layout.manifest)
    counts = {
        state: _count_json_files(layout.root / "queue" / state)
        for state in QUEUE_STATES
    }
    total = _count_json_files(layout.root / "registry")
    details = []
    running_root = layout.root / "queue" / "running"
    if details_limit and running_root.is_dir():
        for bucket in sorted(path for path in running_root.iterdir() if path.is_dir()):
            for path in sorted(bucket.glob("*.json")):
                task = _read_json(path)
                progress_path = layout.progress(str(task["task_id"]))
                progress = _read_json(progress_path) if progress_path.is_file() else {}
                details.append(
                    {
                        "task_id": task["task_id"],
                        "baseline_id": task["baseline_id"],
                        "seed": task["seed"],
                        "worker_id": task.get("worker_id"),
                        "attempt": task.get("attempts"),
                        "status": progress.get("status", "running"),
                    }
                )
                if len(details) >= details_limit:
                    break
            if len(details) >= details_limit:
                break
    return {
        "schema_version": BASELINE_CAMPAIGN_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "baselines": len(manifest["baselines"]),
        "seed_count": int(manifest["seed_count"]),
        "tasks_total": total,
        "queue": counts,
        "accounted": sum(counts.values()),
        "consistent": total == sum(counts.values()),
        "running_details": details,
        "running_details_truncated": counts["running"] > len(details),
    }


def retry_failed_baseline_tasks(
    campaign_root: Path,
    *,
    task_ids: Optional[Sequence[str]] = None,
    reset_attempts: bool = False,
) -> Dict[str, Any]:
    return retry_failed_tasks(
        campaign_root,
        profile_ids=task_ids,
        reset_attempts=reset_attempts,
    )


def collect_baseline_index(
    campaign_root: Path, *, allow_partial: bool = False
) -> Dict[str, Any]:
    layout = CampaignLayout(campaign_root.resolve())
    manifest = _read_json(layout.manifest)
    _verify_baseline_inputs(layout, manifest)
    status = baseline_campaign_status(campaign_root, details_limit=0)
    unfinished = {
        state: int(status["queue"][state])
        for state in ("pending", "running", "failed")
        if int(status["queue"][state])
    }
    if unfinished and not allow_partial:
        raise RuntimeError(
            f"refusing to collect an incomplete baseline campaign: {unfinished}"
        )
    entries = []
    summaries = layout.root / "results" / "summaries"
    if summaries.is_dir():
        for path in sorted(summaries.glob("*.json")):
            result = _read_json(path)
            episodes_path = Path(result["episodes_path"])
            if not episodes_path.is_file() or _sha256(episodes_path) != result["episodes_sha256"]:
                raise RuntimeError(f"missing or mismatched baseline shard: {result['task_id']}")
            entries.append(result)
    expected = len(manifest["baselines"]) * int(manifest["seed_count"])
    index = {
        "schema_version": BASELINE_INDEX_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "generated_at": _now(),
        "config_sha256": manifest["config_sha256"],
        "cache_inventory_sha256": manifest.get("cache_inventory_sha256"),
        "split": manifest["split"],
        "seed_count": manifest["seed_count"],
        "baselines": manifest["baselines"],
        "entries": entries,
        "entry_count": len(entries),
        "complete": not unfinished and len(entries) == expected,
        "queue_at_collection": status["queue"],
    }
    _atomic_json(layout.root / "results" / "index.json", index)
    return index
