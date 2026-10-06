from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from config_utils import load_config, validate_evaluation_workers
from discovery.claude_proposer import preflight_claude_sdk
from discovery.validator import validate_candidate
from evaluation.answering import preflight_project_grader
from evaluation.cache_inventory import verify_cache_inventory
from evaluation.requirements import (
    PROFILE_SCHEMA_VERSION,
    RequirementProfile,
    load_requirement_profiles,
)


CAMPAIGN_SCHEMA = "person_tts_profile_campaign_v1"
TASK_SCHEMA = "person_tts_profile_shard_task_v1"
RESULT_SCHEMA = "person_tts_policy_bank_entry_v1"
INDEX_SCHEMA = "person_tts_policy_experience_bank_v1"
QUEUE_STATES = ("pending", "running", "done", "failed")


def _infer_benchmark(discovery_datasets: Sequence[str]) -> str:
    prefixes = {str(ds).split("_")[0] for ds in discovery_datasets}
    if len(prefixes) != 1:
        raise ValueError(
            f"cannot infer single benchmark from mixed datasets: {list(discovery_datasets)}"
        )
    return prefixes.pop()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch() -> float:
    return time.time()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_int(*parts: object) -> int:
    text = "\u241f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_component(value: str, *, name: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value):
        raise ValueError(f"{name} contains an unsafe path character: {value!r}")
    return value


@contextlib.contextmanager
def _mkdir_lock(path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            path.mkdir(parents=True)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring lock: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.rmdir()


@dataclass(frozen=True)
class CampaignLayout:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "campaign.json"

    @property
    def cache_inventory(self) -> Path:
        return self.root / "cache_inventory.json"

    def registry(self, bucket: str, profile_id: str) -> Path:
        return self.root / "registry" / bucket / f"{profile_id}.json"

    def profile(self, bucket: str, profile_id: str) -> Path:
        return self.root / "profiles" / bucket / f"{profile_id}.json"

    def queue(self, state: str, bucket: str) -> Path:
        if state not in QUEUE_STATES:
            raise ValueError(f"unsupported queue state: {state}")
        return self.root / "queue" / state / bucket

    def active(self, state: str) -> Path:
        if state not in {"pending", "running"}:
            raise ValueError(f"unsupported active marker state: {state}")
        return self.root / "queue" / f"active_{state}"

    def bucket_lock(self, state: str, bucket: str) -> Path:
        return self.root / "queue" / "bucket_locks" / state / f"{bucket}.lock"

    def heartbeat(self, lease_id: str) -> Path:
        return self.root / "heartbeats" / f"{lease_id}.json"

    def progress(self, profile_id: str) -> Path:
        return self.root / "progress" / f"{profile_id}.json"

    def result(self, profile_id: str) -> Path:
        return self.root / "results" / "summaries" / f"{profile_id}.json"

    def policy(self, profile_id: str) -> Path:
        return self.root / "results" / "policies" / f"{profile_id}.py"


def _bucket(profile_id: str, bucket_count: int) -> str:
    width = max(2, len(f"{bucket_count - 1:x}"))
    return f"{_stable_int(profile_id) % bucket_count:0{width}x}"


def _manifest_settings(
    *,
    config_path: Path,
    config: Dict[str, Any],
    bucket_count: int,
    max_attempts: int,
    round_limit: int,
    round_start_mode: str,
    seed_count: int,
    model: str,
    evaluation_workers: int,
    agent_max_turns: Optional[int],
    agent_max_budget_usd: Optional[float],
    agent_timeout_seconds: float,
    agent_max_retries: int,
    agent_max_thinking_tokens: int,
    agent_max_output_tokens: Optional[int],
    inject_strategy_guide: bool = True,
    prompt_path: Optional[str] = None,
    warm_start_bank_path: Optional[str] = None,
) -> Dict[str, Any]:
    if bucket_count < 1 or bucket_count > 4096:
        raise ValueError("bucket_count must be in [1, 4096]")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if round_limit < 1 or round_limit > int(config["discovery_rounds"]):
        raise ValueError("round_limit must be positive and no larger than configured rounds")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    evaluation_workers = validate_evaluation_workers(evaluation_workers)
    if agent_timeout_seconds <= 0 or agent_max_retries < 0 or agent_max_thinking_tokens <= 0:
        raise ValueError("Agent timeout/thinking must be positive and retries non-negative")
    if agent_max_output_tokens is not None and agent_max_output_tokens <= 0:
        raise ValueError("Agent max output tokens must be positive when configured")
    if round_start_mode not in {"autotts_reset", "incumbent_refine"}:
        raise ValueError("invalid round_start_mode")
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "bucket_count": bucket_count,
        "max_attempts": max_attempts,
        "round_limit": round_limit,
        "round_start_mode": round_start_mode,
        "seed_count": seed_count,
        "model": model,
        "evaluation_workers": evaluation_workers,
        "agent_max_turns": agent_max_turns,
        "agent_max_budget_usd": agent_max_budget_usd,
        "agent_timeout_seconds": agent_timeout_seconds,
        "agent_max_retries": agent_max_retries,
        "agent_max_thinking_tokens": agent_max_thinking_tokens,
        "agent_max_output_tokens": agent_max_output_tokens,
        "publish_candidate": False,
        "discovery_datasets": list(config["discovery_datasets"]),
        "heldout_visible": False,
        "inject_strategy_guide": inject_strategy_guide,
        "prompt_path": prompt_path,
        "warm_start_bank_path": warm_start_bank_path,
    }


def _touch_active(layout: CampaignLayout, state: str, bucket: str) -> None:
    path = layout.active(state) / bucket
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _directory_has_json(path: Path) -> bool:
    if not path.is_dir():
        return False
    with os.scandir(path) as entries:
        return any(entry.is_file() and entry.name.endswith(".json") for entry in entries)


def _cleanup_active(layout: CampaignLayout, state: str, bucket: str) -> None:
    lock = layout.bucket_lock(state, bucket)
    try:
        with _mkdir_lock(lock, timeout_seconds=0.2):
            if not _directory_has_json(layout.queue(state, bucket)):
                with contextlib.suppress(FileNotFoundError):
                    (layout.active(state) / bucket).unlink()
    except TimeoutError:
        return


def initialize_campaign(
    *,
    campaign_root: Path,
    profiles_path: Path,
    config_path: Path,
    settings: Dict[str, Any],
    cache_inventory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    layout = CampaignLayout(campaign_root.resolve())
    profiles = load_requirement_profiles(profiles_path.resolve())
    layout.root.mkdir(parents=True, exist_ok=True)
    for state in QUEUE_STATES:
        (layout.root / "queue" / state).mkdir(parents=True, exist_ok=True)
    for state in ("pending", "running"):
        layout.active(state).mkdir(parents=True, exist_ok=True)

    lock = layout.root / "locks" / "initialize.lock"
    with _mkdir_lock(lock):
        manifest_value = {
            "schema_version": CAMPAIGN_SCHEMA,
            "created_at": _now(),
            **settings,
        }
        if cache_inventory is not None:
            cache_sha256 = str(cache_inventory["semantic_sha256"])
            manifest_value["cache_inventory_sha256"] = cache_sha256
            if layout.cache_inventory.is_file():
                recorded_inventory = _read_json(layout.cache_inventory)
                if recorded_inventory.get("semantic_sha256") != cache_sha256:
                    raise ValueError("replay cache inventory differs from immutable campaign")
            else:
                _atomic_json(layout.cache_inventory, cache_inventory)
        if layout.manifest.is_file():
            existing = _read_json(layout.manifest)
            comparable = {key: value for key, value in existing.items() if key != "created_at"}
            requested = {key: value for key, value in manifest_value.items() if key != "created_at"}
            # Backward compatibility: settings keys added to _manifest_settings
            # after a campaign was created are absent from its recorded
            # manifest. A None-valued request means the feature was not used,
            # which is exactly what an absent key also means, so treat the two
            # as equal instead of failing extension of pre-existing campaigns.
            for key, value in requested.items():
                if key not in comparable and value is None:
                    comparable[key] = None
            if comparable != requested:
                raise ValueError(
                    f"campaign settings differ from immutable manifest: recorded={comparable}, requested={requested}"
                )
        else:
            _atomic_json(layout.manifest, manifest_value)

        added: List[str] = []
        existing_ids: List[str] = []
        bucket_count = int(settings["bucket_count"])
        for profile in profiles:
            profile_id = _safe_component(profile.profile_id, name="profile_id")
            bucket = _bucket(profile_id, bucket_count)
            registry_path = layout.registry(bucket, profile_id)
            if registry_path.is_file():
                existing = _read_json(registry_path)
                if existing.get("profile_hash") != profile.fingerprint:
                    raise ValueError(
                        f"profile {profile_id!r} already exists with a different fingerprint"
                    )
                existing_ids.append(profile_id)
                continue

            single_profile = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "profiles": [profile.to_dict()],
            }
            task = {
                "schema_version": TASK_SCHEMA,
                "profile_id": profile_id,
                "profile_hash": profile.fingerprint,
                "bucket": bucket,
                "attempts": 0,
                "created_at": _now(),
            }
            with _mkdir_lock(layout.bucket_lock("pending", bucket)):
                _atomic_json(layout.profile(bucket, profile_id), single_profile)
                _atomic_json(registry_path, task)
                _atomic_json(layout.queue("pending", bucket) / f"{profile_id}.json", task)
                _touch_active(layout, "pending", bucket)
            added.append(profile_id)

        addition = {
            "timestamp": _now(),
            "profiles_path": str(profiles_path.resolve()),
            "profiles_sha256": _sha256(profiles_path.resolve()),
            "requested": len(profiles),
            "added": added,
            "already_present": existing_ids,
        }
        additions = layout.root / "additions.jsonl"
        additions.parent.mkdir(parents=True, exist_ok=True)
        with additions.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(addition, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return addition


def _verify_campaign_inputs(
    layout: CampaignLayout, manifest: Dict[str, Any]
) -> Dict[str, Any]:
    if "config_path" not in manifest or "config_sha256" not in manifest:
        return {"legacy_unfrozen": True}
    config_path = Path(str(manifest["config_path"])).resolve()
    if _sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("experiment config changed after campaign initialization")
    report: Dict[str, Any] = {"config_sha256": manifest["config_sha256"]}
    inventory_sha256 = manifest.get("cache_inventory_sha256")
    if inventory_sha256 is not None:
        if not layout.cache_inventory.is_file():
            raise RuntimeError("campaign cache inventory is missing")
        inventory = _read_json(layout.cache_inventory)
        if inventory.get("semantic_sha256") != inventory_sha256:
            raise ValueError("campaign cache inventory fingerprint mismatch")
        report["cache"] = verify_cache_inventory(load_config(config_path), inventory)
        report["cache_inventory_sha256"] = inventory_sha256
    return report


@dataclass
class ClaimStats:
    active_buckets_seen: int = 0
    buckets_probed: int = 0
    task_entries_probed: int = 0
    rename_conflicts: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dict(vars(self))


@dataclass
class Claim:
    layout: CampaignLayout
    task: Dict[str, Any]
    path: Path
    lease_id: str
    worker_id: str
    handle: Any
    stats: ClaimStats

    def close(self) -> None:
        if self.handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _ordered_active_buckets(
    layout: CampaignLayout, *, worker_id: str, claim_sequence: int
) -> List[str]:
    root = layout.active("pending")
    if not root.is_dir():
        return []
    with os.scandir(root) as entries:
        names = [entry.name for entry in entries if entry.is_file()]
    names.sort(key=lambda name: _stable_int(worker_id, claim_sequence, name))
    return names


def claim_task(
    *,
    layout: CampaignLayout,
    worker_id: str,
    claim_sequence: int,
    max_bucket_probes: int,
    max_task_probes: int,
) -> tuple[Optional[Claim], ClaimStats]:
    if max_bucket_probes < 1 or max_task_probes < 1:
        raise ValueError("claim probe limits must be positive")
    stats = ClaimStats()
    buckets = _ordered_active_buckets(
        layout, worker_id=worker_id, claim_sequence=claim_sequence
    )
    stats.active_buckets_seen = len(buckets)
    for bucket in buckets[:max_bucket_probes]:
        stats.buckets_probed += 1
        pending_dir = layout.queue("pending", bucket)
        candidates: List[str] = []
        if pending_dir.is_dir():
            with os.scandir(pending_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".json"):
                        candidates.append(entry.name)
                        if len(candidates) >= max_task_probes:
                            break
        candidates.sort(key=lambda name: _stable_int(worker_id, claim_sequence, name))
        for name in candidates:
            stats.task_entries_probed += 1
            source = pending_dir / name
            profile_id = name[:-5]
            lease_id = uuid.uuid4().hex
            destination = layout.queue("running", bucket) / f"{profile_id}.{lease_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(source, destination)
            except FileNotFoundError:
                stats.rename_conflicts += 1
                continue
            except OSError:
                stats.rename_conflicts += 1
                continue

            handle = destination.open("r+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            task = json.load(handle)
            task["attempts"] = int(task.get("attempts", 0)) + 1
            task["status"] = "running"
            task["lease_id"] = lease_id
            task["worker_id"] = worker_id
            task["claimed_at"] = _now()
            handle.seek(0)
            handle.truncate()
            json.dump(task, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            _touch_active(layout, "running", bucket)
            _atomic_json(
                layout.heartbeat(lease_id),
                {
                    "lease_id": lease_id,
                    "worker_id": worker_id,
                    "profile_id": profile_id,
                    "heartbeat_epoch": _epoch(),
                    "heartbeat_at": _now(),
                    "status": "claimed",
                },
            )
            _cleanup_active(layout, "pending", bucket)
            return (
                Claim(
                    layout=layout,
                    task=task,
                    path=destination,
                    lease_id=lease_id,
                    worker_id=worker_id,
                    handle=handle,
                    stats=stats,
                ),
                stats,
            )
        if not candidates:
            _cleanup_active(layout, "pending", bucket)
    return None, stats


def _rewrite_locked(claim: Claim, value: Dict[str, Any]) -> None:
    claim.handle.seek(0)
    claim.handle.truncate()
    json.dump(value, claim.handle, ensure_ascii=False, indent=2)
    claim.handle.write("\n")
    claim.handle.flush()
    os.fsync(claim.handle.fileno())


def _transition(claim: Claim, state: str, value: Dict[str, Any]) -> Path:
    profile_id = str(claim.task["profile_id"])
    bucket = str(claim.task["bucket"])
    destination = claim.layout.queue(state, bucket) / f"{profile_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _rewrite_locked(claim, value)
    os.replace(claim.path, destination)
    with contextlib.suppress(FileNotFoundError):
        claim.layout.heartbeat(claim.lease_id).unlink()
    claim.close()
    _cleanup_active(claim.layout, "running", bucket)
    return destination


def complete_claim(claim: Claim, result: Dict[str, Any]) -> Path:
    value = {
        **claim.task,
        "status": "done",
        "completed_at": _now(),
        "result": result,
    }
    return _transition(claim, "done", value)


def fail_claim(claim: Claim, error: Dict[str, Any], *, max_attempts: int) -> Path:
    attempts = int(claim.task["attempts"])
    retry = attempts < max_attempts
    value = {
        **claim.task,
        "status": "pending" if retry else "failed",
        "last_failure_at": _now(),
        "last_error": error,
    }
    state = "pending" if retry else "failed"
    destination = _transition(claim, state, value)
    if retry:
        _touch_active(claim.layout, "pending", str(claim.task["bucket"]))
    return destination


def retry_backoff_seconds(
    *, task_id: str, attempt: int, base_seconds: float, max_seconds: float
) -> float:
    if attempt < 1:
        raise ValueError("retry attempt must be positive")
    if base_seconds < 0 or max_seconds < 0 or max_seconds < base_seconds:
        raise ValueError("retry backoff requires 0 <= base_seconds <= max_seconds")
    if base_seconds == 0:
        return 0.0
    jitter = 0.8 + (_stable_int("retry", task_id, attempt) % 4001) / 10000.0
    return min(max_seconds, base_seconds * (2 ** (attempt - 1)) * jitter)


def prepare_retry_backoff(
    claim: Claim,
    error: Dict[str, Any],
    *,
    base_seconds: float,
    max_seconds: float,
) -> tuple[float, float]:
    task_id = str(claim.task.get("task_id", claim.task["profile_id"]))
    delay = retry_backoff_seconds(
        task_id=task_id,
        attempt=int(claim.task["attempts"]),
        base_seconds=base_seconds,
        max_seconds=max_seconds,
    )
    not_before = _epoch() + delay
    claim.task["last_failure_at"] = _now()
    claim.task["last_error"] = error
    claim.task["retry_backoff_seconds"] = delay
    claim.task["retry_not_before_epoch"] = not_before
    _rewrite_locked(claim, claim.task)
    return delay, not_before


def wait_for_retry_backoff(
    claim: Claim, *, delay_seconds: float, heartbeat_interval_seconds: float
) -> bool:
    heartbeat = ClaimHeartbeat(claim, heartbeat_interval_seconds)
    heartbeat.start()
    try:
        heartbeat_lost = heartbeat.lost_event.wait(delay_seconds)
    finally:
        heartbeat.stop()
    claim.task.pop("retry_backoff_seconds", None)
    claim.task.pop("retry_not_before_epoch", None)
    return heartbeat_lost


def _running_candidates(
    layout: CampaignLayout,
    *,
    max_bucket_probes: int,
    max_task_probes: int,
    reaper_id: str,
    reap_sequence: int,
) -> Iterator[Path]:
    active = layout.active("running")
    if not active.is_dir():
        return
    with os.scandir(active) as entries:
        buckets = sorted(entry.name for entry in entries if entry.is_file())
    if buckets:
        start = (
            _stable_int("reaper", reaper_id) + reap_sequence * max_bucket_probes
        ) % len(buckets)
        buckets = buckets[start:] + buckets[:start]
    for bucket in buckets[:max_bucket_probes]:
        root = layout.queue("running", bucket)
        seen = 0
        if root.is_dir():
            with os.scandir(root) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".json"):
                        yield Path(entry.path)
                        seen += 1
                        if seen >= max_task_probes:
                            break
        if seen == 0:
            _cleanup_active(layout, "running", bucket)


def reap_stale_claims(
    *,
    layout: CampaignLayout,
    lease_timeout_seconds: float,
    max_attempts: int,
    max_bucket_probes: int,
    max_task_probes: int,
    reaper_id: Optional[str] = None,
    reap_sequence: int = 0,
) -> Dict[str, int]:
    if lease_timeout_seconds <= 0:
        raise ValueError("lease_timeout_seconds must be positive")
    report = {"probed": 0, "locked": 0, "live": 0, "requeued": 0, "failed": 0}
    for path in _running_candidates(
        layout,
        max_bucket_probes=max_bucket_probes,
        max_task_probes=max_task_probes,
        reaper_id=reaper_id or f"{socket.gethostname()}.{os.getpid()}",
        reap_sequence=reap_sequence,
    ):
        report["probed"] += 1
        handle = path.open("r+", encoding="utf-8")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report["locked"] += 1
                continue
            claim_data = json.load(handle)
            lease_id = str(claim_data.get("lease_id", ""))
            heartbeat_path = layout.heartbeat(lease_id)
            heartbeat_epoch = path.stat().st_mtime
            if heartbeat_path.is_file():
                with contextlib.suppress(Exception):
                    heartbeat_epoch = float(_read_json(heartbeat_path)["heartbeat_epoch"])
            if _epoch() - heartbeat_epoch < lease_timeout_seconds:
                report["live"] += 1
                continue
            claim = Claim(
                layout=layout,
                task=claim_data,
                path=path,
                lease_id=lease_id,
                worker_id=str(claim_data.get("worker_id", "unknown")),
                handle=handle,
                stats=ClaimStats(),
            )
            handle = None
            destination = fail_claim(
                claim,
                {
                    "type": "StaleLease",
                    "message": f"lease heartbeat exceeded {lease_timeout_seconds}s",
                    "reaped_at": _now(),
                },
                max_attempts=max_attempts,
            )
            report["requeued" if destination.parent.parent.name == "pending" else "failed"] += 1
        finally:
            if handle is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
    return report


class ClaimHeartbeat:
    def __init__(self, claim: Claim, interval_seconds: float):
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.claim = claim
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds + 2.0)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            if not self.claim.path.is_file():
                self.lost_event.set()
                return
            try:
                _atomic_json(
                    self.claim.layout.heartbeat(self.claim.lease_id),
                    {
                        "lease_id": self.claim.lease_id,
                        "worker_id": self.claim.worker_id,
                        "profile_id": self.claim.task["profile_id"],
                        "heartbeat_epoch": _epoch(),
                        "heartbeat_at": _now(),
                        "status": "running",
                    },
                )
            except Exception:
                self.lost_event.set()
                return


def _run_id(manifest: Dict[str, Any], profile_id: str) -> str:
    campaign_id = _safe_component(str(manifest["campaign_id"]), name="campaign_id")
    return f"{campaign_id}--{_safe_component(profile_id, name='profile_id')}"


def _run_root(manifest: Dict[str, Any], profile_id: str) -> Path:
    config = load_config(manifest["config_path"])
    return Path(config["_root"]) / "artifacts" / "discovery" / _run_id(manifest, profile_id)


def archive_interrupted_agent_round(run_root: Path, *, lease_id: str) -> Optional[Path]:
    rounds_root = run_root / "rounds"
    if not rounds_root.is_dir():
        return None
    for round_root in sorted(rounds_root.glob("round_*")):
        if (round_root / "complete.json").is_file():
            continue
        state_path = round_root / "state.json"
        if not state_path.is_file():
            continue
        state = _read_json(state_path)
        status = str(state.get("status", ""))
        if status not in {"agent_running", "agent_failed", "agent_rejected"}:
            return None
        heartbeat_path = round_root / "agent_heartbeat.json"
        heartbeat = _read_json(heartbeat_path) if heartbeat_path.is_file() else {}
        if status == "agent_running" and heartbeat.get("status") == "complete":
            return None
        archive = (
            run_root
            / "interrupted_rounds"
            / round_root.name
            / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{lease_id}"
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.rename(round_root, archive)
        _atomic_json(
            archive / "scheduler_recovery.json",
            {
                "archived_at": _now(),
                "reason": "incomplete Agent call has no safe SDK session resume",
                "logical_round": int(state.get("round", -1)),
                "previous_status": status,
                "previous_heartbeat_status": heartbeat.get("status"),
                "replacement_semantics": "retry same logical round; incomplete proposal is never evaluated",
            },
        )
        return archive
    return None


def discovery_command(
    *,
    manifest: Dict[str, Any],
    profile_path: Path,
    profile_id: str,
    warm_start_seed_path: Optional[Path] = None,
) -> List[str]:
    config_root = Path(manifest["config_path"]).resolve().parent.parent
    command = [
        sys.executable,
        str(config_root / "scripts" / "run_discovery.py"),
        "--config",
        str(Path(manifest["config_path"]).resolve()),
        "--run-id",
        _run_id(manifest, profile_id),
        "--round-limit",
        str(manifest["round_limit"]),
        "--round-start-mode",
        str(manifest["round_start_mode"]),
        "--seed-count",
        str(manifest["seed_count"]),
        "--profiles",
        str(profile_path.resolve()),
        "--model",
        str(manifest["model"]),
        "--agent-timeout-seconds",
        str(manifest["agent_timeout_seconds"]),
        "--agent-max-retries",
        str(manifest["agent_max_retries"]),
        "--agent-max-thinking-tokens",
        str(manifest["agent_max_thinking_tokens"]),
        "--evaluation-workers",
        str(manifest["evaluation_workers"]),
    ]
    if manifest.get("agent_max_output_tokens") is not None:
        command.extend(
            ("--agent-max-output-tokens", str(manifest["agent_max_output_tokens"]))
        )
    if manifest.get("agent_max_turns") is not None:
        command.extend(("--agent-max-turns", str(manifest["agent_max_turns"])))
    if manifest.get("agent_max_budget_usd") is not None:
        command.extend(("--agent-max-budget-usd", str(manifest["agent_max_budget_usd"])))
    if not manifest.get("inject_strategy_guide", True):
        command.append("--no-strategy-guide")
    if manifest.get("prompt_path") is not None:
        command.extend(("--prompt", str(manifest["prompt_path"])))
    if warm_start_seed_path is not None:
        command.extend(("--warm-start-seed", str(warm_start_seed_path.resolve())))
    return command


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{uuid.uuid4().hex}")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def collect_profile_result(
    *, layout: CampaignLayout, manifest: Dict[str, Any], task: Dict[str, Any]
) -> Dict[str, Any]:
    profile_id = str(task["profile_id"])
    run_root = _run_root(manifest, profile_id)
    complete_path = run_root / "complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"discovery did not produce complete.json: {run_root}")
    complete = _read_json(complete_path)
    candidate = Path(str(complete["final_candidate"])).resolve()
    validate_candidate(candidate, allow_beta=False)
    candidate_sha = _sha256(candidate)
    if candidate_sha != complete["final_candidate_sha256"]:
        raise RuntimeError("final candidate hash does not match discovery complete.json")
    policy_path = layout.policy(profile_id)
    _copy_atomic(candidate, policy_path)

    initial = _read_json(run_root / "initial_incumbent" / "complete.json")
    initial_utility = float(initial["summary"]["utility"])
    rounds: List[Dict[str, Any]] = []
    incumbent_utility = initial_utility
    for round_complete_path in sorted((run_root / "rounds").glob("round_*/complete.json")):
        round_complete = _read_json(round_complete_path)
        incumbent_after = round_complete.get("incumbent_after_summary")
        if isinstance(incumbent_after, dict):
            incumbent_utility = float(incumbent_after["utility"])
        rounds.append(
            {
                "round": int(round_complete["round"]),
                "proposal_utility": float(round_complete["summary"]["utility"]),
                "accepted": bool(round_complete["accepted"]),
                "incumbent_utility": incumbent_utility,
                "candidate_sha256": round_complete["candidate_sha256"],
            }
        )
    if len(rounds) != int(manifest["round_limit"]):
        raise RuntimeError(
            f"expected {manifest['round_limit']} completed rounds for {profile_id}, found {len(rounds)}"
        )
    retrieval_record_path = run_root / "retrieval_record.json"
    warm_start_meta = None
    if retrieval_record_path.is_file():
        warm_start_meta = _read_json(retrieval_record_path)
    result = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "profile_id": profile_id,
        "profile_hash": task["profile_hash"],
        "run_id": _run_id(manifest, profile_id),
        "run_root": str(run_root),
        "seed_count": int(manifest["seed_count"]),
        "initial_utility": initial_utility,
        "final_utility": incumbent_utility,
        "rounds": rounds,
        "policy_path": str(policy_path),
        "policy_sha256": candidate_sha,
        "collected_at": _now(),
        "warm_start": warm_start_meta,
    }
    _atomic_json(layout.result(profile_id), result)
    return result


def execute_claim(
    *,
    claim: Claim,
    manifest: Dict[str, Any],
    heartbeat_interval_seconds: float,
    shard_timeout_seconds: Optional[float],
) -> Dict[str, Any]:
    _verify_campaign_inputs(claim.layout, manifest)
    profile_id = str(claim.task["profile_id"])
    bucket = str(claim.task["bucket"])
    profile_path = claim.layout.profile(bucket, profile_id)
    run_root = _run_root(manifest, profile_id)
    archived = archive_interrupted_agent_round(run_root, lease_id=claim.lease_id)

    warm_start_seed_path: Optional[Path] = None
    bank_path_str = manifest.get("warm_start_bank_path")
    if bank_path_str is not None:
        from discovery.warm_start import load_experience_bank, retrieve_seed

        bank = load_experience_bank(Path(bank_path_str))
        target_profiles = load_requirement_profiles(profile_path)
        if len(target_profiles) != 1:
            raise RuntimeError(
                f"campaign shard must have exactly 1 profile, got {len(target_profiles)}"
            )
        target_profile = target_profiles[0]
        target_benchmark = _infer_benchmark(manifest["discovery_datasets"])
        config = load_config(manifest["config_path"])
        retrieval = retrieve_seed(
            bank=bank,
            target_profile=target_profile,
            target_benchmark=target_benchmark,
            target_config=config,
        )
        if retrieval is not None:
            seed_staging = run_root / "warm_start_staging"
            seed_staging.mkdir(parents=True, exist_ok=True)
            seed_copy = seed_staging / "seed.py"
            _copy_atomic(retrieval.source_entry.policy_path, seed_copy)
            if _sha256(seed_copy) != retrieval.source_entry.policy_sha256:
                raise RuntimeError("warm start seed staging hash mismatch")
            warm_start_seed_path = seed_copy
            _atomic_json(run_root / "retrieval_record.json", {
                "source_id": retrieval.source_entry.source_id,
                "distance": retrieval.distance,
                "per_dim_sq": list(retrieval.per_dim_sq),
                "mu": list(retrieval.mu),
                "scale": list(retrieval.scale),
                "compatible_count": retrieval.compatible_count,
                "target_raw": list(retrieval.target_raw),
                "source_raw": list(retrieval.source_raw),
                "seed_sha256": retrieval.source_entry.policy_sha256,
            })
        else:
            _atomic_json(run_root / "retrieval_record.json", {
                "source_id": None,
                "reason": "no compatible entries in experience bank",
            })

    log_path = (
        claim.layout.root
        / "logs"
        / f"{profile_id}.attempt_{claim.task['attempts']}.{claim.lease_id}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = discovery_command(
        manifest=manifest,
        profile_path=profile_path,
        profile_id=profile_id,
        warm_start_seed_path=warm_start_seed_path,
    )
    _atomic_json(
        claim.layout.progress(profile_id),
        {
            "profile_id": profile_id,
            "status": "running",
            "worker_id": claim.worker_id,
            "lease_id": claim.lease_id,
            "attempt": claim.task["attempts"],
            "started_at": _now(),
            "run_root": str(run_root),
            "archived_interrupted_round": str(archived) if archived else None,
            "log_path": str(log_path),
        },
    )
    heartbeat = ClaimHeartbeat(claim, heartbeat_interval_seconds)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(Path(manifest["config_path"]).resolve().parent.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        heartbeat.start()
        try:
            while process.poll() is None:
                if heartbeat.lost_event.wait(1.0):
                    process.terminate()
                    raise RuntimeError("claim heartbeat failed or lease ownership was lost")
                if (
                    shard_timeout_seconds is not None
                    and time.monotonic() - started > shard_timeout_seconds
                ):
                    process.terminate()
                    raise TimeoutError(
                        f"profile shard exceeded {shard_timeout_seconds}s outer timeout"
                    )
            if process.returncode != 0:
                raise RuntimeError(
                    f"discovery subprocess exited with code {process.returncode}; log={log_path}"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
                if process.poll() is None:
                    process.kill()
            heartbeat.stop()
    _verify_campaign_inputs(claim.layout, manifest)
    result = collect_profile_result(layout=claim.layout, manifest=manifest, task=claim.task)
    _atomic_json(
        claim.layout.progress(profile_id),
        {
            "profile_id": profile_id,
            "status": "complete",
            "worker_id": claim.worker_id,
            "attempt": claim.task["attempts"],
            "completed_at": _now(),
            "result_path": str(claim.layout.result(profile_id)),
            "final_utility": result["final_utility"],
        },
    )
    return result


def run_worker(
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
    if manifest.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported campaign manifest")
    _verify_campaign_inputs(layout, manifest)
    config = load_config(manifest["config_path"])
    project_root = Path(config["_root"]).parent.resolve()
    try:
        preflight_project_grader(project_root)
        preflight_claude_sdk()
    except Exception as exc:
        raise RuntimeError(
            "worker preflight failed before shard claim: production grader or Claude "
            f"Agent SDK dependencies are unavailable: {exc}"
        ) from exc
    identity = _safe_component(
        worker_id or f"{socket.gethostname()}.{os.getpid()}", name="worker_id"
    )
    max_attempts = int(manifest["max_attempts"])
    if (
        retry_backoff_base_seconds < 0
        or retry_backoff_max_seconds < retry_backoff_base_seconds
    ):
        raise ValueError("retry backoff requires 0 <= base <= max")
    completed = 0
    failures = 0
    claim_sequence = 0
    idle_started: Optional[float] = None
    aggregate_probe_stats = ClaimStats()
    while True:
        reap_stale_claims(
            layout=layout,
            lease_timeout_seconds=lease_timeout_seconds,
            max_attempts=max_attempts,
            max_bucket_probes=reap_bucket_probes,
            max_task_probes=reap_task_probes,
            reaper_id=identity,
            reap_sequence=claim_sequence,
        )
        claim, stats = claim_task(
            layout=layout,
            worker_id=identity,
            claim_sequence=claim_sequence,
            max_bucket_probes=max_bucket_probes,
            max_task_probes=max_task_probes,
        )
        claim_sequence += 1
        for key, value in vars(stats).items():
            setattr(aggregate_probe_stats, key, getattr(aggregate_probe_stats, key) + value)
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
            result = execute_claim(
                claim=claim,
                manifest=manifest,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
                shard_timeout_seconds=shard_timeout_seconds,
            )
            complete_claim(
                claim,
                {
                    "summary_path": str(layout.result(str(claim.task["profile_id"]))),
                    "policy_path": result["policy_path"],
                    "policy_sha256": result["policy_sha256"],
                    "final_utility": result["final_utility"],
                },
            )
            completed += 1
        except Exception as exc:
            failures += 1
            error = {"type": type(exc).__name__, "message": str(exc), "timestamp": _now()}
            will_retry = int(claim.task["attempts"]) < max_attempts
            delay = not_before = None
            if will_retry:
                delay, not_before = prepare_retry_backoff(
                    claim,
                    error,
                    base_seconds=retry_backoff_base_seconds,
                    max_seconds=retry_backoff_max_seconds,
                )
            _atomic_json(
                layout.progress(str(claim.task["profile_id"])),
                {
                    "profile_id": claim.task["profile_id"],
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
            fail_claim(claim, error, max_attempts=max_attempts)
        if once:
            break
    return {
        "worker_id": identity,
        "completed": completed,
        "failures": failures,
        "claim_cycles": claim_sequence,
        "probe_stats": aggregate_probe_stats.to_dict(),
    }


def _count_json_files(root: Path) -> int:
    count = 0
    if not root.is_dir():
        return 0
    for bucket in os.scandir(root):
        if not bucket.is_dir():
            continue
        with os.scandir(bucket.path) as entries:
            count += sum(entry.is_file() and entry.name.endswith(".json") for entry in entries)
    return count


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _running_detail(
    *,
    layout: CampaignLayout,
    manifest: Dict[str, Any],
    claim_path: Path,
    expected_problem_count: Optional[int],
) -> Dict[str, Any]:
    claim = _read_json(claim_path)
    profile_id = str(claim["profile_id"])
    run_root = _run_root(manifest, profile_id)
    completed_rounds = len(list((run_root / "rounds").glob("round_*/complete.json")))
    detail: Dict[str, Any] = {
        "profile_id": profile_id,
        "worker_id": claim.get("worker_id"),
        "attempt": claim.get("attempts"),
        "claimed_at": claim.get("claimed_at"),
        "completed_rounds": completed_rounds,
        "requested_rounds": int(manifest["round_limit"]),
    }
    incomplete = [
        path
        for path in sorted((run_root / "rounds").glob("round_*"))
        if not (path / "complete.json").is_file()
    ]
    if incomplete:
        round_root = incomplete[0]
        state_path = round_root / "state.json"
        state = _read_json(state_path) if state_path.is_file() else {}
        detail["current_round"] = state.get("round")
        detail["phase"] = state.get("status")
        heartbeat_path = round_root / "agent_heartbeat.json"
        if heartbeat_path.is_file():
            heartbeat = _read_json(heartbeat_path)
            detail["agent"] = {
                key: heartbeat.get(key)
                for key in (
                    "status",
                    "elapsed_seconds",
                    "tool_call_count",
                    "num_turns",
                    "total_cost_usd",
                )
            }
        evaluation = round_root / "evaluation" / "episodes.jsonl"
        if evaluation.is_file():
            detail["evaluation_episodes"] = _line_count(evaluation)
            if expected_problem_count is not None:
                detail["evaluation_episodes_expected"] = (
                    int(manifest["seed_count"]) * expected_problem_count
                )
    elif not (run_root / "initial_incumbent" / "complete.json").is_file():
        detail["phase"] = "initial_incumbent"
        evaluation = run_root / "initial_incumbent" / "evaluation" / "episodes.jsonl"
        detail["evaluation_episodes"] = _line_count(evaluation)
        if expected_problem_count is not None:
            detail["evaluation_episodes_expected"] = (
                int(manifest["seed_count"]) * expected_problem_count
            )
    elif (run_root / "complete.json").is_file():
        detail["phase"] = "collecting"
    else:
        detail["phase"] = "between_rounds"
    return detail


def campaign_status(campaign_root: Path, *, details_limit: int = 20) -> Dict[str, Any]:
    if details_limit < 0:
        raise ValueError("details_limit must be non-negative")
    layout = CampaignLayout(campaign_root.resolve())
    manifest = _read_json(layout.manifest)
    counts = {
        state: _count_json_files(layout.root / "queue" / state)
        for state in QUEUE_STATES
    }
    total = _count_json_files(layout.root / "registry")
    utilities: List[float] = []
    summaries = layout.root / "results" / "summaries"
    if summaries.is_dir():
        for entry in os.scandir(summaries):
            if entry.is_file() and entry.name.endswith(".json"):
                utilities.append(float(_read_json(Path(entry.path))["final_utility"]))
    running_details: List[Dict[str, Any]] = []
    if details_limit:
        running_root = layout.root / "queue" / "running"
        if running_root.is_dir():
            expected_problem_count: Optional[int] = None
            if counts["running"]:
                config = load_config(Path(manifest["config_path"]).resolve())
                configured_problem_count = config.get(
                    "expected_problem_count_by_split", {}
                ).get("discovery")
                if configured_problem_count is not None:
                    expected_problem_count = int(configured_problem_count)
            for bucket in sorted(path for path in running_root.iterdir() if path.is_dir()):
                for claim_path in sorted(bucket.glob("*.json")):
                    running_details.append(
                        _running_detail(
                            layout=layout,
                            manifest=manifest,
                            claim_path=claim_path,
                            expected_problem_count=expected_problem_count,
                        )
                    )
                    if len(running_details) >= details_limit:
                        break
                if len(running_details) >= details_limit:
                    break
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "profiles_total": total,
        "queue": counts,
        "accounted": sum(counts.values()),
        "active_pending_buckets": len(list(layout.active("pending").glob("*"))),
        "active_running_buckets": len(list(layout.active("running").glob("*"))),
        "results": len(utilities),
        "utility": (
            {
                "min": min(utilities),
                "mean": sum(utilities) / len(utilities),
                "max": max(utilities),
            }
            if utilities
            else None
        ),
        "running_details": running_details,
        "running_details_truncated": counts["running"] > len(running_details),
        "consistent": total == sum(counts.values()),
    }


def retry_failed_tasks(
    campaign_root: Path,
    *,
    profile_ids: Optional[Sequence[str]] = None,
    reset_attempts: bool = False,
) -> Dict[str, Any]:
    layout = CampaignLayout(campaign_root.resolve())
    selected = (
        {_safe_component(value, name="profile_id") for value in profile_ids}
        if profile_ids is not None
        else None
    )
    retried: List[str] = []
    failed_root = layout.root / "queue" / "failed"
    if failed_root.is_dir():
        for bucket_dir in sorted(path for path in failed_root.iterdir() if path.is_dir()):
            bucket = bucket_dir.name
            for failed_path in sorted(bucket_dir.glob("*.json")):
                profile_id = failed_path.stem
                if selected is not None and profile_id not in selected:
                    continue
                with _mkdir_lock(layout.bucket_lock("pending", bucket)):
                    value = _read_json(failed_path)
                    value["status"] = "pending"
                    value["manual_retry_at"] = _now()
                    value["manual_retry_previous_attempts"] = int(value.get("attempts", 0))
                    if reset_attempts:
                        value["attempts"] = 0
                    with failed_path.open("w", encoding="utf-8") as handle:
                        json.dump(value, handle, ensure_ascii=False, indent=2)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    _touch_active(layout, "pending", bucket)
                    destination = layout.queue("pending", bucket) / failed_path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(failed_path, destination)
                retried.append(profile_id)
    missing = sorted(selected - set(retried)) if selected is not None else []
    if missing:
        raise ValueError(f"requested profiles are not in failed state: {missing}")
    return {"retried": retried, "count": len(retried), "reset_attempts": reset_attempts}


def collect_policy_bank(
    campaign_root: Path, *, allow_partial: bool = False
) -> Dict[str, Any]:
    layout = CampaignLayout(campaign_root.resolve())
    manifest = _read_json(layout.manifest)
    _verify_campaign_inputs(layout, manifest)
    status = campaign_status(campaign_root, details_limit=0)
    unfinished = {
        state: int(status["queue"][state])
        for state in ("pending", "running", "failed")
        if int(status["queue"][state])
    }
    if unfinished and not allow_partial:
        raise RuntimeError(
            f"refusing to collect an incomplete policy bank: {unfinished}; "
            "use --allow-partial only for an explicitly partial diagnostic index"
        )
    entries: List[Dict[str, Any]] = []
    summaries = layout.root / "results" / "summaries"
    if summaries.is_dir():
        for path in sorted(summaries.glob("*.json")):
            result = _read_json(path)
            policy_path = Path(str(result["policy_path"]))
            if not policy_path.is_file() or _sha256(policy_path) != result["policy_sha256"]:
                raise RuntimeError(f"missing or mismatched policy artifact for {result['profile_id']}")
            entries.append(
                {
                    "profile_id": result["profile_id"],
                    "profile_hash": result["profile_hash"],
                    "policy_path": str(policy_path),
                    "policy_sha256": result["policy_sha256"],
                    "initial_utility": result["initial_utility"],
                    "final_utility": result["final_utility"],
                    "rounds": result["rounds"],
                }
            )
    index = {
        "schema_version": INDEX_SCHEMA,
        "campaign_id": manifest["campaign_id"],
        "generated_at": _now(),
        "config_sha256": manifest.get("config_sha256"),
        "cache_inventory_sha256": manifest.get("cache_inventory_sha256"),
        "profiles": entries,
        "profile_count": len(entries),
        "complete": not unfinished and len(entries) == int(status["profiles_total"]),
        "queue_at_collection": status["queue"],
    }
    _atomic_json(layout.root / "results" / "index.json", index)
    return index
