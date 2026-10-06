from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from discovery.profile_campaign import (
    CampaignLayout,
    _atomic_json,
    _atomic_text,
    _read_json,
    _sha256,
    collect_policy_bank,
)
from evaluation.baseline_campaign import collect_baseline_index
from evaluation.requirements import (
    RequirementProfile,
    evaluate_seed_batch_utility,
    load_requirement_profiles,
)


COMPARISON_SCHEMA = "person_tts_baseline_comparison_v1"
METRICS = ("accuracy", "latency_s", "weighted_compute")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            values.append(value)
    return values


def _validate_profile_set(
    profiles: Sequence[RequirementProfile], policy_index: Mapping[str, Any]
) -> None:
    expected = {profile.profile_id: profile.fingerprint for profile in profiles}
    actual = {
        str(entry["profile_id"]): str(entry["profile_hash"])
        for entry in policy_index["profiles"]
    }
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        )
        raise ValueError(
            f"PersonalTTS campaign/profile mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}, changed={changed[:5]}"
        )


def _personaltts_profile_details(
    campaign_root: Path,
    profiles: Sequence[RequirementProfile],
    *,
    expected_seeds: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    layout = CampaignLayout(campaign_root.resolve())
    details: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        result_path = layout.result(profile.profile_id)
        if not result_path.is_file():
            raise RuntimeError(f"missing PersonalTTS result: {profile.profile_id}")
        result = _read_json(result_path)
        if result.get("profile_hash") != profile.fingerprint:
            raise ValueError(f"PersonalTTS profile fingerprint mismatch: {profile.profile_id}")
        run_root = Path(result["run_root"])
        initial = _read_json(run_root / "initial_incumbent" / "complete.json")["summary"]
        current = initial
        for path in sorted((run_root / "rounds").glob("round_*/complete.json")):
            complete = _read_json(path)
            incumbent = complete.get("incumbent_after_summary")
            if not isinstance(incumbent, dict):
                raise ValueError(f"missing incumbent summary: {path}")
            current = incumbent
        if set(current.get("profiles", {})) != {profile.profile_id}:
            raise ValueError(
                f"PersonalTTS shard is not single-profile: {profile.profile_id}"
            )
        if list(current.get("expected_seeds", [])) != list(expected_seeds):
            raise ValueError(f"PersonalTTS seed set mismatch: {profile.profile_id}")
        if int(current.get("expected_problems_per_batch", -1)) <= 0:
            raise ValueError(f"invalid PersonalTTS problem count: {profile.profile_id}")
        if not math.isclose(
            float(current["utility"]), float(result["final_utility"]), abs_tol=1e-12
        ):
            raise ValueError(f"PersonalTTS final utility mismatch: {profile.profile_id}")
        details[profile.profile_id] = current["profiles"][profile.profile_id]
    return details


def _baseline_records(
    baseline_id: str,
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
) -> tuple[List[Dict[str, Any]], List[str]]:
    by_seed: Dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if entry["baseline_id"] != baseline_id:
            continue
        seed = int(entry["seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate baseline shard: {baseline_id}, seed={seed}")
        by_seed[seed] = entry
    if set(by_seed) != set(expected_seeds):
        raise ValueError(
            f"incomplete baseline seeds for {baseline_id}: "
            f"expected={list(expected_seeds)}, actual={sorted(by_seed)}"
        )

    records: List[Dict[str, Any]] = []
    problem_keys: List[str] | None = None
    seen = set()
    for seed in expected_seeds:
        entry = by_seed[seed]
        episodes_path = Path(str(entry["episodes_path"]))
        if _sha256(episodes_path) != entry["episodes_sha256"]:
            raise RuntimeError(f"baseline shard hash mismatch: {entry['task_id']}")
        manifest = _read_json(Path(entry["output_dir"]) / "evaluation_manifest.json")
        if manifest.get("seeds") != [seed]:
            raise ValueError(f"baseline shard seed manifest mismatch: {entry['task_id']}")
        keys = [str(value) for value in manifest["problem_keys"]]
        if problem_keys is None:
            problem_keys = keys
        elif problem_keys != keys:
            raise ValueError(f"problem ordering differs in baseline shard: {entry['task_id']}")
        shard_records = _read_jsonl(episodes_path)
        if len(shard_records) != len(keys):
            raise ValueError(f"baseline shard episode count mismatch: {entry['task_id']}")
        for record in shard_records:
            key = (str(record["dataset"]), int(record["index"]), int(record["seed"]))
            if int(record["seed"]) != seed or key in seen:
                raise ValueError(f"duplicate or wrong-seed baseline episode: {key}")
            seen.add(key)
            records.append(record)
    assert problem_keys is not None
    return records, problem_keys


def _score_baseline_profiles(
    records: Sequence[Mapping[str, Any]],
    profiles: Sequence[RequirementProfile],
    *,
    problem_keys: Sequence[str],
    expected_seeds: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    details = {}
    for profile in profiles:
        personalized = [
            {
                "dataset": record["dataset"],
                "index": record["index"],
                "seed": record["seed"],
                "profile_id": profile.profile_id,
                "profile_hash": profile.fingerprint,
                "correct": record["correct"],
                "cost": record["cost"],
            }
            for record in records
        ]
        utility = evaluate_seed_batch_utility(
            personalized,
            [profile],
            expected_problem_keys=problem_keys,
            expected_seeds=expected_seeds,
        )
        details[profile.profile_id] = utility["profiles"][profile.profile_id]
    return details


def _summarize_method(
    details: Mapping[str, Mapping[str, Any]],
    profiles: Sequence[RequirementProfile],
) -> Dict[str, Any]:
    weighted_satisfied = weighted_batches = 0.0
    satisfied = batches = 0
    constraints: Dict[str, Dict[str, float]] = {}
    for profile in profiles:
        value = details[profile.profile_id]
        profile_batches = int(value["batches"])
        profile_satisfied = int(value["satisfied_batches"])
        batches += profile_batches
        satisfied += profile_satisfied
        weighted_batches += profile.weight * profile_batches
        weighted_satisfied += profile.weight * profile_satisfied
        for metric in METRICS:
            item = value["constraints"][metric]
            aggregate = constraints.setdefault(
                metric,
                {"weighted_satisfied": 0.0, "weighted_batches": 0.0, "weighted_margin": 0.0},
            )
            aggregate["weighted_satisfied"] += profile.weight * int(
                item["satisfied_batches"]
            )
            aggregate["weighted_batches"] += profile.weight * profile_batches
            aggregate["weighted_margin"] += profile.weight * profile_batches * float(
                item["mean_margin"]
            )
    return {
        "utility": weighted_satisfied / weighted_batches,
        "unweighted_satisfaction_rate": satisfied / batches,
        "satisfied_batches": satisfied,
        "batch_count": batches,
        "profile_count": len(profiles),
        "constraints": {
            metric: {
                "satisfaction_rate": value["weighted_satisfied"]
                / value["weighted_batches"],
                "mean_margin": value["weighted_margin"] / value["weighted_batches"],
            }
            for metric, value in constraints.items()
        },
    }


def _csv_text(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def analyze_campaigns(
    *,
    personal_campaign_root: Path,
    baseline_campaign_root: Path,
    profiles_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    profiles = load_requirement_profiles(profiles_path.resolve())
    personal_root = personal_campaign_root.resolve()
    baseline_root = baseline_campaign_root.resolve()
    personal_manifest = _read_json(personal_root / "campaign.json")
    baseline_manifest = _read_json(baseline_root / "campaign.json")
    if personal_manifest["config_sha256"] != baseline_manifest["config_sha256"]:
        raise ValueError("baseline and PersonalTTS campaigns use different experiment configs")
    if personal_manifest.get("cache_inventory_sha256") != baseline_manifest.get(
        "cache_inventory_sha256"
    ):
        raise ValueError("baseline and PersonalTTS campaigns use different replay caches")
    if baseline_manifest.get("split") != "discovery":
        raise ValueError("this comparison requires a discovery-split baseline campaign")
    if personal_manifest.get("heldout_visible") is not False:
        raise ValueError("PersonalTTS campaign does not record held-out isolation")
    seed_count = int(personal_manifest["seed_count"])
    if seed_count != int(baseline_manifest["seed_count"]):
        raise ValueError("baseline and PersonalTTS seed counts differ")
    expected_seeds = list(range(seed_count))

    policy_index = collect_policy_bank(personal_root)
    if not policy_index["complete"]:
        raise RuntimeError("PersonalTTS policy bank is incomplete")
    _validate_profile_set(profiles, policy_index)
    baseline_index = collect_baseline_index(baseline_root)
    if not baseline_index["complete"]:
        raise RuntimeError("baseline campaign is incomplete")

    method_details: Dict[str, Dict[str, Dict[str, Any]]] = {}
    baseline_specs = {spec["id"]: spec for spec in baseline_index["baselines"]}
    problem_keys: List[str] | None = None
    for baseline_id in baseline_specs:
        records, keys = _baseline_records(
            baseline_id,
            baseline_index["entries"],
            expected_seeds=expected_seeds,
        )
        if problem_keys is None:
            problem_keys = keys
        elif problem_keys != keys:
            raise ValueError("baseline methods use different problem sets")
        method_details[baseline_id] = _score_baseline_profiles(
            records,
            profiles,
            problem_keys=keys,
            expected_seeds=expected_seeds,
        )
    method_details["personaltts"] = _personaltts_profile_details(
        personal_root, profiles, expected_seeds=expected_seeds
    )

    methods = {
        method_id: _summarize_method(details, profiles)
        for method_id, details in method_details.items()
    }
    best_baseline = max(baseline_specs, key=lambda key: methods[key]["utility"])
    comparisons = {
        "best_baseline": best_baseline,
        "best_baseline_utility": methods[best_baseline]["utility"],
        "personaltts_utility": methods["personaltts"]["utility"],
        "personaltts_delta": methods["personaltts"]["utility"]
        - methods[best_baseline]["utility"],
    }
    profile_comparisons = {}
    for profile in profiles:
        rates = {
            method_id: float(details[profile.profile_id]["satisfaction_rate"])
            for method_id, details in method_details.items()
        }
        profile_best = max(baseline_specs, key=lambda key: rates[key])
        profile_comparisons[profile.profile_id] = {
            "methods": rates,
            "best_baseline": profile_best,
            "best_baseline_rate": rates[profile_best],
            "personaltts_rate": rates["personaltts"],
            "personaltts_delta": rates["personaltts"] - rates[profile_best],
        }

    report = {
        "schema_version": COMPARISON_SCHEMA,
        "profiles_path": str(profiles_path.resolve()),
        "profiles_sha256": _sha256(profiles_path.resolve()),
        "personal_campaign_root": str(personal_root),
        "baseline_campaign_root": str(baseline_root),
        "config_sha256": personal_manifest["config_sha256"],
        "split": baseline_manifest["split"],
        "seed_count": seed_count,
        "problem_count": len(problem_keys or []),
        "baseline_specs": baseline_specs,
        "methods": methods,
        "comparison": comparisons,
        "profile_comparisons": profile_comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "comparison.json", report)

    method_rows = []
    for method_id, value in methods.items():
        method_rows.append(
            {
                "method": method_id,
                "utility": value["utility"],
                "unweighted_satisfaction_rate": value["unweighted_satisfaction_rate"],
                "satisfied_batches": value["satisfied_batches"],
                "batch_count": value["batch_count"],
                "accuracy_requirement_rate": value["constraints"]["accuracy"]["satisfaction_rate"],
                "latency_requirement_rate": value["constraints"]["latency_s"]["satisfaction_rate"],
                "weighted_compute_requirement_rate": value["constraints"]["weighted_compute"]["satisfaction_rate"],
            }
        )
    method_fields = list(method_rows[0])
    _atomic_text(output_dir / "method_summary.csv", _csv_text(method_rows, method_fields))

    profile_rows = []
    for profile in profiles:
        for method_id, details in method_details.items():
            value = details[profile.profile_id]
            profile_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "method": method_id,
                    "satisfaction_rate": value["satisfaction_rate"],
                    "satisfied_batches": value["satisfied_batches"],
                    "batch_count": value["batches"],
                    "accuracy_requirement_rate": value["constraints"]["accuracy"]["satisfaction_rate"],
                    "latency_requirement_rate": value["constraints"]["latency_s"]["satisfaction_rate"],
                    "weighted_compute_requirement_rate": value["constraints"]["weighted_compute"]["satisfaction_rate"],
                }
            )
    profile_fields = list(profile_rows[0])
    _atomic_text(
        output_dir / "profile_comparison.csv", _csv_text(profile_rows, profile_fields)
    )

    lines = [
        "# PersonalTTS and baseline hard-requirement comparison",
        "",
        f"Profiles: {len(profiles)}; seeds per profile: {seed_count}; problems per seed: {len(problem_keys or [])}.",
        "",
        "| Method | Utility | Satisfied batches | Accuracy req. | Latency req. | Compute req. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['method']} | {row['utility']:.4f} | "
            f"{row['satisfied_batches']}/{row['batch_count']} | "
            f"{row['accuracy_requirement_rate']:.4f} | "
            f"{row['latency_requirement_rate']:.4f} | "
            f"{row['weighted_compute_requirement_rate']:.4f} |"
        )
    lines.extend(
        (
            "",
            f"Best baseline: `{best_baseline}` ({methods[best_baseline]['utility']:.4f}).",
            f"PersonalTTS delta: {comparisons['personaltts_delta']:+.4f}.",
            "",
        )
    )
    _atomic_text(output_dir / "comparison.md", "\n".join(lines))
    return report
