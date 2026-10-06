from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from evaluation.requirements import PROFILE_SCHEMA_VERSION, RequirementProfile


CALIBRATION_SCHEMA_VERSION = "person_tts_profile_calibration_v1"
RANGE_SCHEMA_VERSION = "person_tts_profile_ranges_v1"
ANCHOR_SCHEMA_VERSION = "person_tts_profile_anchors_v1"
QUANTILES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)


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
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "quantiles": {
            f"q{round(fraction * 100):02d}": _percentile(values, fraction)
            for fraction in QUANTILES
        },
    }


def _batch_aggregate(values: Sequence[float], aggregation: str) -> float:
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "batch":
        return _percentile(values, 0.95)
    if aggregation == "max":
        return max(values)
    raise ValueError(f"unsupported aggregation: {aggregation}")


def _record_group(record: Mapping[str, Any]) -> str:
    if "profile_id" in record:
        return f"profile={record['profile_id']}@{record.get('profile_hash', 'missing')}"
    if "beta" in record:
        return f"beta={float(record['beta']):.4f}"
    return "default"


def calibrate_episode_records(
    sources: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    expected_problem_keys: Sequence[str],
) -> Dict[str, Any]:
    expected = set(expected_problem_keys)
    if not expected:
        raise ValueError("expected_problem_keys must not be empty")
    batch_records: Dict[tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for source_name, records in sources.items():
        if not source_name:
            raise ValueError("calibration source names must not be empty")
        for record in records:
            batch_records[
                (source_name, _record_group(record), int(record["seed"]))
            ].append(record)
    if not batch_records:
        raise ValueError("no calibration records were supplied")

    batches: List[Dict[str, Any]] = []
    for (source_name, group, seed), records in sorted(batch_records.items()):
        actual = [f"{record['dataset']}:{int(record['index'])}" for record in records]
        if len(actual) != len(set(actual)):
            raise ValueError(
                f"duplicate problems in calibration batch {source_name}/{group}/seed={seed}"
            )
        if set(actual) != expected:
            missing = sorted(expected - set(actual))
            extra = sorted(set(actual) - expected)
            raise ValueError(
                f"incomplete calibration batch {source_name}/{group}/seed={seed}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        latency_values = []
        compute_values = []
        for record in records:
            cost = record.get("cost")
            if not isinstance(cost, Mapping) or "latency_s" not in cost:
                raise ValueError(
                    "calibration requires latency-aware replay records with cost.latency_s"
                )
            latency_values.append(float(cost["latency_s"]))
            compute_values.append(float(cost["weighted_compute"]))
        batches.append(
            {
                "source": source_name,
                "group": group,
                "seed": seed,
                "problems": len(records),
                "accuracy": sum(int(bool(record["correct"])) for record in records)
                / len(records),
                "latency_s": {
                    aggregation: _batch_aggregate(latency_values, aggregation)
                    for aggregation in ("mean", "batch", "max")
                },
                "weighted_compute": {
                    aggregation: _batch_aggregate(compute_values, aggregation)
                    for aggregation in ("mean", "batch", "max")
                },
            }
        )

    distributions: Dict[str, Any] = {
        "accuracy": {"mean": _distribution([batch["accuracy"] for batch in batches])}
    }
    for metric in ("latency_s", "weighted_compute"):
        distributions[metric] = {
            aggregation: _distribution(
                [float(batch[metric][aggregation]) for batch in batches]
            )
            for aggregation in ("mean", "batch", "max")
        }
    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "expected_problems_per_batch": len(expected),
        "expected_problem_keys_sha256": _canonical_sha256(sorted(expected)),
        "source_count": len(sources),
        "batch_count": len(batches),
        "distributions": distributions,
        "batches": batches,
    }
    report["calibration_sha256"] = _canonical_sha256(report)
    return report


def range_template(calibration: Mapping[str, Any]) -> Dict[str, Any]:
    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration schema")
    distributions = calibration["distributions"]
    return {
        "schema_version": RANGE_SCHEMA_VERSION,
        "calibration_sha256": calibration["calibration_sha256"],
        "approved": False,
        "generation": {
            "method": "independent_uniform",
            "seed": None,
            "count": None,
            "profile_weight": 1.0,
        },
        "ranges": {
            "accuracy": {
                "aggregation": "mean",
                "min": None,
                "max": None,
                "observed": distributions["accuracy"]["mean"],
            },
            "latency_s": {
                "aggregation": "batch",
                "min": None,
                "max": None,
                "observed_by_aggregation": distributions["latency_s"],
            },
            "weighted_compute": {
                "aggregation": "mean",
                "min": None,
                "max": None,
                "observed_by_aggregation": distributions["weighted_compute"],
            },
        },
    }


def _validated_range(raw: Mapping[str, Any], metric: str) -> tuple[str, float, float]:
    aggregation = str(raw.get("aggregation", ""))
    allowed = {"mean"} if metric == "accuracy" else {"mean", "batch", "max"}
    if aggregation not in allowed:
        raise ValueError(f"invalid {metric} aggregation: {aggregation!r}")
    low = float(raw.get("min"))
    high = float(raw.get("max"))
    if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high < low:
        raise ValueError(f"invalid approved range for {metric}: [{low}, {high}]")
    if metric == "accuracy" and high > 1.0:
        raise ValueError("accuracy range must stay in [0, 1]")
    return aggregation, low, high


def _validated_anchor_set(
    anchor_set: Mapping[str, Any],
    validated_ranges: Mapping[str, tuple[str, float, float]],
) -> tuple[List[Dict[str, Any]], str]:
    if anchor_set.get("schema_version") != ANCHOR_SCHEMA_VERSION:
        raise ValueError("unsupported profile anchor schema")
    raw_anchors = anchor_set.get("anchors")
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise ValueError("profile anchor set must contain a non-empty anchors list")
    anchors_sha256 = _canonical_sha256(raw_anchors)
    declared_hash = anchor_set.get("anchors_sha256")
    if declared_hash is not None and str(declared_hash) != anchors_sha256:
        raise ValueError("profile anchor set hash does not match its anchors")
    identifiers = set()
    anchors: List[Dict[str, Any]] = []
    for raw in raw_anchors:
        if not isinstance(raw, Mapping):
            raise ValueError("each profile anchor must be a mapping")
        anchor_id = str(raw.get("id", ""))
        if not anchor_id or anchor_id in identifiers:
            raise ValueError(f"invalid or duplicate profile anchor id: {anchor_id!r}")
        identifiers.add(anchor_id)
        values = {
            metric: float(raw.get(metric))
            for metric in ("accuracy", "latency_s", "weighted_compute")
        }
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"profile anchor {anchor_id!r} has invalid metrics")
        anchor = {
            "id": anchor_id,
            "source": str(raw.get("source", "unknown")),
            **values,
        }
        if "metadata" in raw:
            anchor["metadata"] = raw["metadata"]
        anchors.append(anchor)
    accuracy_low = validated_ranges["accuracy"][1]
    latency_high = validated_ranges["latency_s"][2]
    compute_high = validated_ranges["weighted_compute"][2]
    feasible = [
        anchor
        for anchor in anchors
        if anchor["accuracy"] >= accuracy_low
        and anchor["latency_s"] <= latency_high
        and anchor["weighted_compute"] <= compute_high
    ]
    if not feasible:
        raise ValueError("no profile anchors are feasible within the approved ranges")
    return feasible, anchors_sha256


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = (
        float(left["accuracy"]) >= float(right["accuracy"])
        and float(left["latency_s"]) <= float(right["latency_s"])
        and float(left["weighted_compute"]) <= float(right["weighted_compute"])
    )
    strictly_better = (
        float(left["accuracy"]) > float(right["accuracy"])
        or float(left["latency_s"]) < float(right["latency_s"])
        or float(left["weighted_compute"]) < float(right["weighted_compute"])
    )
    return no_worse and strictly_better


def _pareto_front(anchors: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(anchor)
        for index, anchor in enumerate(anchors)
        if not any(
            _dominates(other, anchor)
            for other_index, other in enumerate(anchors)
            if other_index != index
        )
    ]


def _normalized_coordinate(value: float, low: float, high: float, *, log: bool) -> float:
    if high <= low:
        return 0.0
    clipped = min(high, max(low, value))
    if log and low > 0:
        return (math.log(clipped) - math.log(low)) / (math.log(high) - math.log(low))
    return (clipped - low) / (high - low)


def _anchor_coordinate(
    anchor: Mapping[str, Any],
    validated_ranges: Mapping[str, tuple[str, float, float]],
) -> tuple[float, float, float]:
    return (
        _normalized_coordinate(
            float(anchor["accuracy"]),
            validated_ranges["accuracy"][1],
            validated_ranges["accuracy"][2],
            log=False,
        ),
        _normalized_coordinate(
            float(anchor["latency_s"]),
            validated_ranges["latency_s"][1],
            validated_ranges["latency_s"][2],
            log=True,
        ),
        _normalized_coordinate(
            float(anchor["weighted_compute"]),
            validated_ranges["weighted_compute"][1],
            validated_ranges["weighted_compute"][2],
            log=True,
        ),
    )


def _maximin_order(
    anchors: Sequence[Dict[str, Any]],
    validated_ranges: Mapping[str, tuple[str, float, float]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    remaining = list(anchors)
    rng.shuffle(remaining)
    selected = [remaining.pop()]
    coordinates = {
        str(anchor["id"]): _anchor_coordinate(anchor, validated_ranges)
        for anchor in anchors
    }

    def distance_squared(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        left_coordinate = coordinates[str(left["id"])]
        right_coordinate = coordinates[str(right["id"])]
        return sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left_coordinate, right_coordinate)
        )

    minimum_distance = {
        str(anchor["id"]): distance_squared(anchor, selected[0])
        for anchor in remaining
    }
    while remaining:
        best_index = max(
            range(len(remaining)),
            key=lambda index: minimum_distance[str(remaining[index]["id"])],
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        minimum_distance.pop(str(chosen["id"]))
        for anchor in remaining:
            anchor_id = str(anchor["id"])
            minimum_distance[anchor_id] = min(
                minimum_distance[anchor_id], distance_squared(anchor, chosen)
            )
    return selected


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    if high <= low:
        return low
    if low <= 0:
        return rng.uniform(low, high)
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _anchor_satisfies(
    anchor: Mapping[str, Any], constraints: Sequence[Mapping[str, Any]]
) -> bool:
    values = {
        "accuracy": float(anchor["accuracy"]),
        "latency_s": float(anchor["latency_s"]),
        "weighted_compute": float(anchor["weighted_compute"]),
    }
    for constraint in constraints:
        value = values[str(constraint["metric"])]
        threshold = float(constraint["value"])
        if constraint["op"] == ">=" and value < threshold:
            return False
        if constraint["op"] == "<=" and value > threshold:
            return False
    return True


def generate_profiles(
    approved_ranges: Mapping[str, Any],
    anchor_set: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if approved_ranges.get("schema_version") != RANGE_SCHEMA_VERSION:
        raise ValueError("unsupported profile range schema")
    if approved_ranges.get("approved") is not True:
        raise ValueError("profile ranges must be explicitly marked approved=true")
    generation = approved_ranges.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("generation settings are required")
    method = str(generation.get("method", ""))
    if method not in {"independent_uniform", "pareto_anchor_jitter"}:
        raise ValueError(f"unsupported profile generation method: {method!r}")
    seed = int(generation.get("seed"))
    count = int(generation.get("count"))
    weight = float(generation.get("profile_weight", 1.0))
    if count < 1:
        raise ValueError("profile count must be positive")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("profile_weight must be finite and positive")
    ranges = approved_ranges.get("ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("ranges mapping is required")
    validated = {
        metric: _validated_range(ranges[metric], metric)
        for metric in ("accuracy", "latency_s", "weighted_compute")
    }
    rng = random.Random(seed)
    profiles = []
    provenance: Dict[str, Any] = {}
    sampling: Dict[str, Any] = {
        "method": method,
        "seed": seed,
        "count": count,
        "profile_weight": weight,
    }
    anchor_set_sha256 = ""

    if method == "independent_uniform":
        for index in range(count):
            constraints = []
            for metric in ("accuracy", "latency_s", "weighted_compute"):
                aggregation, low, high = validated[metric]
                constraints.append(
                    {
                        "metric": metric,
                        "aggregation": aggregation,
                        "op": ">=" if metric == "accuracy" else "<=",
                        "value": rng.uniform(low, high),
                    }
                )
            profile = RequirementProfile.from_dict(
                {
                    "id": f"profile_{index:04d}",
                    "weight": weight,
                    "constraints": constraints,
                }
            )
            profiles.append(profile.to_dict())
    else:
        if not isinstance(anchor_set, Mapping):
            raise ValueError("pareto_anchor_jitter requires a profile anchor set")
        feasible, anchor_set_sha256 = _validated_anchor_set(anchor_set, validated)
        expected_anchor_hash = approved_ranges.get("anchor_set_sha256")
        if expected_anchor_hash is None:
            raise ValueError(
                "pareto_anchor_jitter ranges must declare anchor_set_sha256"
            )
        if str(expected_anchor_hash) != anchor_set_sha256:
            raise ValueError(
                "approved ranges anchor_set_sha256 does not match the supplied anchors"
            )
        pareto = _pareto_front(feasible)
        ordered = _maximin_order(pareto, validated, rng)
        accuracy_slack = float(generation.get("accuracy_slack_abs", 0.05))
        latency_multiplier = float(generation.get("latency_slack_multiplier", 1.25))
        compute_multiplier = float(
            generation.get("weighted_compute_slack_multiplier", 1.25)
        )
        if not math.isfinite(accuracy_slack) or not 0 <= accuracy_slack <= 1:
            raise ValueError("accuracy_slack_abs must be in [0, 1]")
        if (
            not math.isfinite(latency_multiplier)
            or latency_multiplier < 1
            or not math.isfinite(compute_multiplier)
            or compute_multiplier < 1
        ):
            raise ValueError("resource slack multipliers must be finite and at least 1")
        sampling.update(
            {
                "candidate_anchor_count": len(feasible),
                "pareto_anchor_count": len(pareto),
                "selected_unique_anchor_count": min(count, len(ordered)),
                "selection": "seeded_maximin_in_normalized_accuracy_log_latency_log_compute",
                "accuracy_slack_abs": accuracy_slack,
                "latency_slack_multiplier": latency_multiplier,
                "weighted_compute_slack_multiplier": compute_multiplier,
                "feasibility_guarantee": "selected_anchor_satisfies_all_constraints",
            }
        )
        for index in range(count):
            anchor = ordered[index % len(ordered)]
            accuracy_aggregation, accuracy_low, accuracy_high = validated["accuracy"]
            latency_aggregation, latency_low, latency_high = validated["latency_s"]
            compute_aggregation, compute_low, compute_high = validated[
                "weighted_compute"
            ]
            accuracy_upper = min(accuracy_high, float(anchor["accuracy"]))
            accuracy_lower = max(accuracy_low, accuracy_upper - accuracy_slack)
            latency_lower = max(latency_low, float(anchor["latency_s"]))
            latency_upper = min(
                latency_high, max(latency_lower, latency_lower * latency_multiplier)
            )
            compute_lower = max(compute_low, float(anchor["weighted_compute"]))
            compute_upper = min(
                compute_high, max(compute_lower, compute_lower * compute_multiplier)
            )
            constraints = [
                {
                    "metric": "accuracy",
                    "aggregation": accuracy_aggregation,
                    "op": ">=",
                    "value": rng.uniform(accuracy_lower, accuracy_upper),
                },
                {
                    "metric": "latency_s",
                    "aggregation": latency_aggregation,
                    "op": "<=",
                    "value": _log_uniform(rng, latency_lower, latency_upper),
                },
                {
                    "metric": "weighted_compute",
                    "aggregation": compute_aggregation,
                    "op": "<=",
                    "value": _log_uniform(rng, compute_lower, compute_upper),
                },
            ]
            if not _anchor_satisfies(anchor, constraints):
                raise AssertionError("generated profile is not satisfied by its anchor")
            profile_id = f"profile_{index:04d}"
            profile = RequirementProfile.from_dict(
                {
                    "id": profile_id,
                    "weight": weight,
                    "constraints": constraints,
                }
            )
            profiles.append(profile.to_dict())
            provenance[profile_id] = {
                "anchor_id": anchor["id"],
                "anchor_source": anchor["source"],
                "anchor_metrics": {
                    metric: float(anchor[metric])
                    for metric in ("accuracy", "latency_s", "weighted_compute")
                },
                "anchor_metadata": anchor.get("metadata"),
            }

    result = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "source_ranges_sha256": _canonical_sha256(approved_ranges),
        "calibration_sha256": str(approved_ranges.get("calibration_sha256", "")),
        "sampling": sampling,
        "profiles": profiles,
    }
    if anchor_set_sha256:
        result["anchor_set_sha256"] = anchor_set_sha256
        result["profile_provenance"] = provenance
        result["profile_provenance_sha256"] = _canonical_sha256(provenance)
    result["profiles_sha256"] = _canonical_sha256(profiles)
    return result


def write_generated_profiles(path: Path, profiles: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != profiles:
            raise ValueError(
                f"refusing to overwrite a different persisted profile sample: {path}"
            )
        return
    _atomic_json(path, profiles)


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic_json(path, value)
