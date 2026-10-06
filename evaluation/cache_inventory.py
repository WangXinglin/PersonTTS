from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from config_utils import model_weights, resolve_from_root
from replay.adapter import load_problem


CACHE_INVENTORY_SCHEMA = "person_tts_cache_inventory_v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_paths(config: Mapping[str, Any], split: str) -> list[Path]:
    if split not in {"discovery", "heldout"}:
        raise ValueError("split must be discovery or heldout")
    cache_root = resolve_from_root(dict(config), str(config["private_cache_root"]))
    paths: list[Path] = []
    seen_indices: set[tuple[str, int]] = set()
    for dataset in config[f"{split}_datasets"]:
        dataset_root = cache_root / str(dataset)
        dataset_paths = list(dataset_root.glob("*.json"))
        try:
            dataset_paths.sort(key=lambda path: int(path.stem))
        except ValueError as exc:
            raise ValueError(f"non-numeric replay cache filename under {dataset_root}") from exc
        if not dataset_paths:
            raise RuntimeError(f"no replay caches found for configured dataset {dataset!r}")
        for path in dataset_paths:
            key = (str(dataset), int(path.stem))
            if key in seen_indices:
                raise ValueError(f"duplicate replay cache key: {key}")
            seen_indices.add(key)
            paths.append(path)
    expected = config.get("expected_problem_count_by_split", {}).get(split)
    if expected is not None and len(paths) != int(expected):
        raise ValueError(
            f"{split} cache inventory requires {int(expected)} problems, found {len(paths)}"
        )
    return paths


def build_cache_inventory(
    config: Mapping[str, Any],
    *,
    split: str,
    require_latency: bool,
) -> Dict[str, Any]:
    cache_root = resolve_from_root(dict(config), str(config["private_cache_root"]))
    expected_models = sorted(model_weights(dict(config)))
    entries = []
    for path in _cache_paths(config, split):
        before = path.stat()
        content_sha256 = _file_sha256(path)
        problem = load_problem(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"replay cache changed while inventorying: {path}")
        expected_dataset = path.parent.name
        expected_index = int(path.stem)
        if problem.dataset != expected_dataset or int(problem.index) != expected_index:
            raise ValueError(
                f"cache identity/path mismatch: {path} contains "
                f"{problem.dataset}:{problem.index}"
            )
        actual_models = sorted(str(name) for name in problem.models)
        if actual_models != expected_models:
            raise ValueError(
                f"cache model set mismatch for {path}: "
                f"expected={expected_models}, actual={actual_models}"
            )
        if require_latency and not problem.latency_available:
            raise ValueError(f"latency-aware cache v2 required: {path}")
        entries.append(
            {
                "relative_path": path.relative_to(cache_root).as_posix(),
                "dataset": problem.dataset,
                "index": int(problem.index),
                "size": int(after.st_size),
                "mtime_ns": int(after.st_mtime_ns),
                "content_sha256": content_sha256,
                "models": actual_models,
                "latency_available": bool(problem.latency_available),
            }
        )
    semantic = {
        "schema_version": CACHE_INVENTORY_SCHEMA,
        "split": split,
        "datasets": [str(value) for value in config[f"{split}_datasets"]],
        "expected_models": expected_models,
        "entries": [
            {key: value for key, value in entry.items() if key != "mtime_ns"}
            for entry in entries
        ],
    }
    return {
        **semantic,
        "cache_root": str(cache_root),
        "entries": entries,
        "semantic_sha256": _canonical_sha256(semantic),
    }


def verify_cache_inventory(
    config: Mapping[str, Any], inventory: Mapping[str, Any]
) -> Dict[str, int]:
    if inventory.get("schema_version") != CACHE_INVENTORY_SCHEMA:
        raise ValueError("unsupported cache inventory schema")
    split = str(inventory["split"])
    current_paths = _cache_paths(config, split)
    cache_root = resolve_from_root(dict(config), str(config["private_cache_root"]))
    expected_entries = {
        str(entry["relative_path"]): entry for entry in inventory["entries"]
    }
    current_relative = {path.relative_to(cache_root).as_posix() for path in current_paths}
    if current_relative != set(expected_entries):
        missing = sorted(set(expected_entries) - current_relative)
        extra = sorted(current_relative - set(expected_entries))
        raise ValueError(
            f"replay cache file set changed: missing={missing[:5]}, extra={extra[:5]}"
        )

    semantic = {
        "schema_version": inventory["schema_version"],
        "split": split,
        "datasets": inventory["datasets"],
        "expected_models": inventory["expected_models"],
        "entries": [
            {key: value for key, value in entry.items() if key != "mtime_ns"}
            for entry in inventory["entries"]
        ],
    }
    if _canonical_sha256(semantic) != inventory.get("semantic_sha256"):
        raise ValueError("cache inventory document fingerprint mismatch")

    hashed = 0
    for relative, entry in expected_entries.items():
        path = cache_root / relative
        stat = path.stat()
        if int(stat.st_size) != int(entry["size"]):
            raise ValueError(f"replay cache size changed: {path}")
        if int(stat.st_mtime_ns) != int(entry["mtime_ns"]):
            hashed += 1
            if _file_sha256(path) != entry["content_sha256"]:
                raise ValueError(f"replay cache content changed: {path}")
    return {"files": len(current_paths), "content_hashes_rechecked": hashed}
