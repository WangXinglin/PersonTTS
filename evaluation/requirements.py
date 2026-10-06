from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROFILE_SCHEMA_VERSION = "person_tts_requirement_profiles_v1"
UTILITY_SCHEMA_VERSION = "person_tts_seed_batch_utility_v1"
SUPPORTED_METRICS = {"accuracy", "latency_s", "weighted_compute"}
SUPPORTED_AGGREGATIONS = {
    "accuracy": {"mean"},
    "latency_s": {"mean", "batch", "max"},
    "weighted_compute": {"mean", "batch", "max"},
}
SUPPORTED_OPERATORS = {"<=", ">="}
_PROFILE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Constraint:
    metric: str
    aggregation: str
    op: str
    value: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Constraint":
        constraint = cls(
            metric=str(raw.get("metric", "")),
            aggregation=str(raw.get("aggregation", "")),
            op=str(raw.get("op", "")),
            value=float(raw.get("value")),
        )
        if constraint.metric not in SUPPORTED_METRICS:
            raise ValueError(f"unsupported requirement metric: {constraint.metric!r}")
        if constraint.aggregation not in SUPPORTED_AGGREGATIONS[constraint.metric]:
            raise ValueError(
                f"unsupported aggregation for {constraint.metric}: "
                f"{constraint.aggregation!r}"
            )
        if constraint.op not in SUPPORTED_OPERATORS:
            raise ValueError(f"unsupported requirement operator: {constraint.op!r}")
        if not math.isfinite(constraint.value) or constraint.value < 0.0:
            raise ValueError("requirement values must be finite and non-negative")
        if constraint.metric == "accuracy" and constraint.value > 1.0:
            raise ValueError("accuracy requirement must be in [0, 1]")
        if constraint.metric == "accuracy" and constraint.op != ">=":
            raise ValueError("accuracy is a minimum requirement and must use >=")
        if constraint.metric != "accuracy" and constraint.op != "<=":
            raise ValueError(f"{constraint.metric} is a maximum requirement and must use <=")
        return constraint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "aggregation": self.aggregation,
            "op": self.op,
            "value": self.value,
        }


@dataclass(frozen=True)
class RequirementProfile:
    profile_id: str
    weight: float
    constraints: tuple[Constraint, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RequirementProfile":
        profile_id = str(raw.get("id", ""))
        if not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError(
                "requirement profile id must contain only letters, digits, '.', '_', or '-'"
            )
        weight = float(raw.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("requirement profile weight must be finite and positive")
        raw_constraints = raw.get("constraints")
        if not isinstance(raw_constraints, list):
            raise ValueError("requirement profile constraints must be a list")
        constraints = tuple(Constraint.from_dict(value) for value in raw_constraints)
        metrics = [constraint.metric for constraint in constraints]
        if len(metrics) != len(set(metrics)):
            raise ValueError(f"duplicate constraints in profile {profile_id!r}")
        if set(metrics) != SUPPORTED_METRICS:
            raise ValueError(
                f"profile {profile_id!r} must define exactly one hard requirement for "
                f"each of {sorted(SUPPORTED_METRICS)!r}"
            )
        return cls(profile_id=profile_id, weight=weight, constraints=constraints)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.profile_id,
            "weight": self.weight,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()

    def policy_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload.update(
            {constraint.metric: constraint.to_dict() for constraint in self.constraints}
        )
        return payload


def load_requirement_profiles(path: Path) -> List[RequirementProfile]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"requirement profile file must use schema {PROFILE_SCHEMA_VERSION!r}"
        )
    values = raw.get("profiles")
    if not isinstance(values, list) or not values:
        raise ValueError("requirement profile file must contain a non-empty profiles list")
    profiles = [RequirementProfile.from_dict(value) for value in values]
    ids = [profile.profile_id for profile in profiles]
    if len(ids) != len(set(ids)):
        raise ValueError("requirement profile ids must be unique")
    return profiles


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot aggregate an empty batch")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _aggregate(values: Sequence[float], aggregation: str) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty batch")
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "batch":
        return _percentile(values, 0.95)
    if aggregation == "max":
        return max(values)
    raise ValueError(f"unsupported aggregation: {aggregation!r}")


def _constraint_values(
    records: Sequence[Mapping[str, Any]], constraint: Constraint
) -> List[float]:
    if constraint.metric == "accuracy":
        return [float(bool(record["correct"])) for record in records]
    return [float(record["cost"][constraint.metric]) for record in records]


def _sqlite_aggregate(
    connection: sqlite3.Connection,
    *,
    profile_id: str,
    seed: int,
    metric: str,
    aggregation: str,
) -> float:
    column = {
        "accuracy": "correct",
        "latency_s": "latency_s",
        "weighted_compute": "weighted_compute",
    }[metric]
    parameters = (profile_id, seed)
    where = "WHERE profile_id = ? AND seed = ?"
    if aggregation == "mean":
        row = connection.execute(
            f"SELECT AVG({column}) FROM episodes {where}", parameters
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError("cannot aggregate an empty batch")
        return float(row[0])
    if aggregation == "max":
        row = connection.execute(
            f"SELECT MAX({column}) FROM episodes {where}", parameters
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError("cannot aggregate an empty batch")
        return float(row[0])
    if aggregation != "batch":
        raise ValueError(f"unsupported aggregation: {aggregation!r}")
    count = int(
        connection.execute(
            f"SELECT COUNT(*) FROM episodes {where}", parameters
        ).fetchone()[0]
    )
    if not count:
        raise ValueError("cannot aggregate an empty batch")
    position = (count - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)

    def ordered_value(offset: int) -> float:
        row = connection.execute(
            f"SELECT {column} FROM episodes {where} "
            f"ORDER BY {column} LIMIT 1 OFFSET ?",
            (*parameters, offset),
        ).fetchone()
        if row is None:
            raise RuntimeError("missing percentile row")
        return float(row[0])

    lower_value = ordered_value(lower)
    if lower == upper:
        return lower_value
    fraction = position - lower
    return lower_value * (1.0 - fraction) + ordered_value(upper) * fraction


def evaluate_seed_batch_utility(
    records: Iterable[Mapping[str, Any]],
    profiles: Sequence[RequirementProfile],
    *,
    expected_problem_keys: Sequence[str],
    expected_seeds: Sequence[int] | None = None,
) -> Dict[str, Any]:
    expected = set(expected_problem_keys)
    if not expected:
        raise ValueError("expected_problem_keys must not be empty")
    profile_by_id = {profile.profile_id: profile for profile in profiles}
    if len(profile_by_id) != len(profiles) or not profiles:
        raise ValueError("profiles must be non-empty with unique ids")

    connection = sqlite3.connect("")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE episodes ("
        "profile_id TEXT, seed INTEGER, problem_key TEXT, correct REAL, "
        "latency_s REAL, weighted_compute REAL, "
        "PRIMARY KEY (profile_id, seed, problem_key))"
    )
    try:
        for record in records:
            profile_id = str(record.get("profile_id", ""))
            if profile_id not in profile_by_id:
                raise ValueError(f"episode references unknown profile: {profile_id!r}")
            expected_hash = profile_by_id[profile_id].fingerprint
            if record.get("profile_hash") != expected_hash:
                raise ValueError(f"profile hash mismatch for {profile_id!r}")
            seed = int(record["seed"])
            problem_key = f"{record['dataset']}:{int(record['index'])}"
            try:
                connection.execute(
                    "INSERT INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        profile_id,
                        seed,
                        problem_key,
                        float(bool(record["correct"])),
                        float(record["cost"]["latency_s"]),
                        float(record["cost"]["weighted_compute"]),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"duplicate problems in profile={profile_id}, seed={seed}"
                ) from exc
        connection.commit()
        grouped = [
            (str(profile_id), int(seed))
            for profile_id, seed in connection.execute(
                "SELECT DISTINCT profile_id, seed FROM episodes ORDER BY profile_id, seed"
            )
        ]
        required_seeds = (
            {int(seed) for seed in expected_seeds}
            if expected_seeds is not None
            else {seed for _, seed in grouped}
        )
        if expected_seeds is not None and len(required_seeds) != len(expected_seeds):
            raise ValueError("expected_seeds must not contain duplicates")
        for profile in profiles:
            actual_seeds = {
                seed for profile_id, seed in grouped if profile_id == profile.profile_id
            }
            if actual_seeds != required_seeds:
                raise ValueError(
                    f"incomplete seed set for profile={profile.profile_id}: "
                    f"missing={sorted(required_seeds - actual_seeds)}, "
                    f"extra={sorted(actual_seeds - required_seeds)}"
                )

        batch_results: List[Dict[str, Any]] = []
        for profile_id, seed in grouped:
            actual = {
                str(row[0])
                for row in connection.execute(
                    "SELECT problem_key FROM episodes "
                    "WHERE profile_id = ? AND seed = ?",
                    (profile_id, seed),
                )
            }
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"incomplete seed batch for profile={profile_id}, seed={seed}: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            profile = profile_by_id[profile_id]
            constraint_results = []
            for constraint in profile.constraints:
                observed = _sqlite_aggregate(
                    connection,
                    profile_id=profile_id,
                    seed=seed,
                    metric=constraint.metric,
                    aggregation=constraint.aggregation,
                )
                if constraint.op == ">=":
                    satisfied = observed >= constraint.value
                    margin = observed - constraint.value
                else:
                    satisfied = observed <= constraint.value
                    margin = constraint.value - observed
                constraint_results.append(
                    {
                        **constraint.to_dict(),
                        "observed": observed,
                        "satisfied": satisfied,
                        "margin": margin,
                    }
                )
            batch_results.append(
                {
                    "profile_id": profile_id,
                    "profile_hash": profile.fingerprint,
                    "seed": seed,
                    "problems": len(actual),
                    "satisfied": all(item["satisfied"] for item in constraint_results),
                    "constraints": constraint_results,
                }
            )
    finally:
        connection.close()

    profile_results: Dict[str, Any] = {}
    weighted_satisfied = 0.0
    weighted_batches = 0.0
    for profile in profiles:
        batches = [
            batch for batch in batch_results if batch["profile_id"] == profile.profile_id
        ]
        if not batches:
            raise ValueError(f"no batches evaluated for profile {profile.profile_id!r}")
        satisfied_batches = sum(int(batch["satisfied"]) for batch in batches)
        constraint_summary: Dict[str, Any] = {}
        for constraint in profile.constraints:
            results = [
                next(
                    item
                    for item in batch["constraints"]
                    if item["metric"] == constraint.metric
                )
                for batch in batches
            ]
            constraint_summary[constraint.metric] = {
                "satisfied_batches": sum(int(item["satisfied"]) for item in results),
                "satisfaction_rate": sum(int(item["satisfied"]) for item in results)
                / len(results),
                "mean_margin": sum(float(item["margin"]) for item in results)
                / len(results),
            }
        profile_results[profile.profile_id] = {
            "weight": profile.weight,
            "batches": len(batches),
            "satisfied_batches": satisfied_batches,
            "satisfaction_rate": satisfied_batches / len(batches),
            "constraints": constraint_summary,
        }
        weighted_satisfied += profile.weight * satisfied_batches
        weighted_batches += profile.weight * len(batches)

    return {
        "schema_version": UTILITY_SCHEMA_VERSION,
        "utility": weighted_satisfied / weighted_batches,
        "profiles": profile_results,
        "batch_count": len(batch_results),
        "expected_problems_per_batch": len(expected),
        "expected_seeds": sorted(required_seeds),
        "batches": batch_results,
    }
