"""Experience-bank loading and warm-start seed retrieval.

Retrieval algorithm (fixed):
  x(u) = [accuracy_min, ln(latency_max), ln(compute_max)]
  Filter: same benchmark + compatible models/policy_config.
  Standardize: fit mu/sigma on source-only (ddof=0); zero-variance → scale=1.
  Distance: Euclidean on standardized features, equal weights.
  Top-1; ties broken by stable source_id sort.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BANK_SCHEMA = "person_tts_experience_bank_v1"


def _canon_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    campaign_id: str
    profile_id: str
    benchmark: str
    accuracy_min: float
    latency_max: float
    compute_max: float
    constraint_specs: Tuple[Dict[str, Any], ...]
    policy_path: Path
    policy_sha256: str
    satisfaction_rate: float
    models_hash: str
    policy_config_hash: str


@dataclass(frozen=True)
class ExperienceBank:
    entries: Tuple[SourceEntry, ...]
    path: Path
    schema_version: str


@dataclass(frozen=True)
class RetrievalResult:
    source_entry: SourceEntry
    distance: float
    per_dim_sq: Tuple[float, float, float]
    mu: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    compatible_count: int
    target_raw: Tuple[float, float, float]
    source_raw: Tuple[float, float, float]


def _parse_entry(raw: Mapping[str, Any]) -> SourceEntry:
    constraints = raw["constraints"]
    acc_min = float(constraints["accuracy_min"])
    lat_max = float(constraints["latency_max"])
    comp_max = float(constraints["compute_max"])
    if not math.isfinite(acc_min) or acc_min < 0.0 or acc_min > 1.0:
        raise ValueError(f"accuracy_min must be in [0,1], got {acc_min}")
    if not math.isfinite(lat_max) or lat_max <= 0.0:
        raise ValueError(f"latency_max must be > 0, got {lat_max}")
    if not math.isfinite(comp_max) or comp_max <= 0.0:
        raise ValueError(f"compute_max must be > 0, got {comp_max}")
    return SourceEntry(
        source_id=str(raw["source_id"]),
        campaign_id=str(raw["campaign_id"]),
        profile_id=str(raw["profile_id"]),
        benchmark=str(raw["benchmark"]),
        accuracy_min=acc_min,
        latency_max=lat_max,
        compute_max=comp_max,
        constraint_specs=tuple(raw.get("constraint_specs", ())),
        policy_path=Path(raw["policy_path"]),
        policy_sha256=str(raw["policy_sha256"]),
        satisfaction_rate=float(raw.get("satisfaction_rate", 0.0)),
        models_hash=_canon_hash(raw["models"]),
        policy_config_hash=_canon_hash(raw["candidate_policy_config"]),
    )


def load_experience_bank(path: Path) -> ExperienceBank:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("schema_version", "")
    if version != BANK_SCHEMA:
        raise ValueError(f"expected schema {BANK_SCHEMA!r}, got {version!r}")
    valid: List[SourceEntry] = []
    for i, raw in enumerate(data.get("entries", [])):
        try:
            entry = _parse_entry(raw)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("skipping entry %d: parse error: %s", i, exc)
            continue
        if not entry.policy_path.is_file():
            logger.warning(
                "excluding %s: policy file missing: %s",
                entry.source_id, entry.policy_path,
            )
            continue
        actual_hash = _sha256(entry.policy_path)
        if actual_hash != entry.policy_sha256:
            logger.warning(
                "excluding %s: policy hash mismatch (expected %s, got %s)",
                entry.source_id, entry.policy_sha256, actual_hash,
            )
            continue
        valid.append(entry)
    return ExperienceBank(
        entries=tuple(valid), path=path.resolve(), schema_version=version,
    )


def _extract_features(acc_min: float, lat_max: float, comp_max: float) -> np.ndarray:
    return np.array([acc_min, math.log(lat_max), math.log(comp_max)], dtype=np.float64)


def _is_compatible(
    entry: SourceEntry,
    *,
    target_benchmark: str,
    target_models_hash: str,
    target_policy_config_hash: str,
) -> bool:
    return (
        entry.benchmark == target_benchmark
        and entry.models_hash == target_models_hash
        and entry.policy_config_hash == target_policy_config_hash
    )


def retrieve_seed(
    bank: ExperienceBank,
    target_profile: Any,
    *,
    target_benchmark: str,
    target_config: Dict[str, Any],
) -> Optional[RetrievalResult]:
    target_models_hash = _canon_hash(target_config["models"])
    target_policy_config_hash = _canon_hash(target_config["candidate_policy_config"])

    acc_constraint = next(
        c for c in target_profile.constraints if c.metric == "accuracy"
    )
    lat_constraint = next(
        c for c in target_profile.constraints if c.metric == "latency_s"
    )
    comp_constraint = next(
        c for c in target_profile.constraints if c.metric == "weighted_compute"
    )
    target_acc = float(acc_constraint.value)
    target_lat = float(lat_constraint.value)
    target_comp = float(comp_constraint.value)

    if target_lat <= 0.0 or target_comp <= 0.0:
        raise ValueError(
            f"target latency_max ({target_lat}) and compute_max ({target_comp}) must be > 0"
        )

    compatible = [
        e for e in bank.entries
        if _is_compatible(
            e,
            target_benchmark=target_benchmark,
            target_models_hash=target_models_hash,
            target_policy_config_hash=target_policy_config_hash,
        )
    ]
    if not compatible:
        logger.info("no compatible entries in experience bank; falling back to template")
        return None

    source_features = np.array(
        [_extract_features(e.accuracy_min, e.latency_max, e.compute_max) for e in compatible],
        dtype=np.float64,
    )
    target_features = _extract_features(target_acc, target_lat, target_comp)

    mu = source_features.mean(axis=0)
    sigma = source_features.std(axis=0, ddof=0)
    scale = np.where(sigma > 0.0, sigma, 1.0)

    z_sources = (source_features - mu) / scale
    z_target = (target_features - mu) / scale

    diffs_sq = (z_sources - z_target) ** 2
    distances = np.sqrt(diffs_sq.sum(axis=1))

    sorted_indices = sorted(
        range(len(compatible)),
        key=lambda i: (distances[i], compatible[i].source_id),
    )
    best_idx = sorted_indices[0]
    best_entry = compatible[best_idx]
    best_dist = float(distances[best_idx])
    best_per_dim = tuple(float(v) for v in diffs_sq[best_idx])

    return RetrievalResult(
        source_entry=best_entry,
        distance=best_dist,
        per_dim_sq=best_per_dim,  # type: ignore[arg-type]
        mu=tuple(float(v) for v in mu),  # type: ignore[arg-type]
        scale=tuple(float(v) for v in scale),  # type: ignore[arg-type]
        compatible_count=len(compatible),
        target_raw=(target_acc, target_lat, target_comp),
        source_raw=(best_entry.accuracy_min, best_entry.latency_max, best_entry.compute_max),
    )


def retrieve_seed_dry_run(
    bank_path: Path,
    profiles_path: Path,
    *,
    target_benchmark: str,
    target_config_path: Path,
) -> List[Dict[str, Any]]:
    from config_utils import load_config
    from evaluation.requirements import load_requirement_profiles

    bank = load_experience_bank(bank_path)
    config = load_config(str(target_config_path))
    profiles = load_requirement_profiles(profiles_path)
    results = []
    for profile in profiles:
        result = retrieve_seed(
            bank, profile, target_benchmark=target_benchmark, target_config=config,
        )
        if result is None:
            results.append({
                "target_profile_id": profile.profile_id,
                "matched": False,
                "reason": "no compatible entries",
            })
        else:
            results.append({
                "target_profile_id": profile.profile_id,
                "matched": True,
                "source_id": result.source_entry.source_id,
                "distance": result.distance,
                "per_dim_sq": list(result.per_dim_sq),
                "mu": list(result.mu),
                "scale": list(result.scale),
                "compatible_count": result.compatible_count,
                "target_raw": list(result.target_raw),
                "source_raw": list(result.source_raw),
                "source_satisfaction_rate": result.source_entry.satisfaction_rate,
                "seed_sha256": result.source_entry.policy_sha256,
            })
    return results
