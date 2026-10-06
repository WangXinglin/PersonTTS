#!/usr/bin/env python3
"""Merge sharded AutoTTS-baseline evaluation summaries into one summary.json.

Each shard's per-profile satisfied_batches/batches counts are computed
independently (a "batch" == one seed, scored entirely within that seed), so
merging by summation is exact -- no approximation, unlike naively averaging
percentile-based stats across shards.

Usage:
    python scripts/merge_autotts_baseline_shards.py \
        --shards-dir artifacts/heldout_evaluation/autotts_aime/all_profiles/autotts_aime_beta05/AIME_26/shards \
        --output artifacts/heldout_evaluation/autotts_aime/all_profiles/autotts_aime_beta05/AIME_26/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def merge_profile(acc: Dict[str, Any], shard: Dict[str, Any]) -> None:
    acc["weight"] = shard["weight"]
    acc["batches"] = acc.get("batches", 0) + shard["batches"]
    acc["satisfied_batches"] = acc.get("satisfied_batches", 0) + shard["satisfied_batches"]
    for metric, stats in shard["constraints"].items():
        c = acc.setdefault("constraints", {}).setdefault(metric, {
            "satisfied_batches": 0, "batches": 0, "margin_sum": 0.0,
        })
        c["satisfied_batches"] += stats["satisfied_batches"]
        c["batches"] += shard["batches"]
        c["margin_sum"] += stats["mean_margin"] * shard["batches"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded seed-batch evaluation summaries")
    parser.add_argument("--shards-dir", required=True, help="Directory containing shard_label subdirs")
    parser.add_argument("--output", required=True, help="Output merged summary.json path")
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    shard_dirs = sorted(p for p in shards_dir.iterdir() if p.is_dir())
    if not shard_dirs:
        raise SystemExit(f"no shard directories found under {shards_dir}")

    merged_profiles: Dict[str, Dict[str, Any]] = {}
    all_seeds: list[int] = []
    schema_version = None
    problems_per_batch = None

    for shard_dir in shard_dirs:
        summary_path = shard_dir / "summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"shard missing summary.json: {summary_path}")
        with open(summary_path, encoding="utf-8") as f:
            shard_summary = json.load(f)
        schema_version = shard_summary["schema_version"]
        problems_per_batch = shard_summary["expected_problems_per_batch"]
        all_seeds.extend(shard_summary["expected_seeds"])
        for pid, pdata in shard_summary["profiles"].items():
            merge_profile(merged_profiles.setdefault(pid, {}), pdata)

    if len(set(all_seeds)) != len(all_seeds):
        raise SystemExit(f"duplicate seeds across shards detected: {len(all_seeds)} entries, {len(set(all_seeds))} unique")

    weighted_satisfied = 0.0
    weighted_batches = 0.0
    out_profiles: Dict[str, Any] = {}
    for pid, acc in merged_profiles.items():
        weight = acc["weight"]
        batches = acc["batches"]
        satisfied = acc["satisfied_batches"]
        weighted_satisfied += weight * satisfied
        weighted_batches += weight * batches
        constraints_out = {}
        for metric, c in acc["constraints"].items():
            constraints_out[metric] = {
                "satisfied_batches": c["satisfied_batches"],
                "satisfaction_rate": c["satisfied_batches"] / c["batches"],
                "mean_margin": c["margin_sum"] / c["batches"],
            }
        out_profiles[pid] = {
            "weight": weight,
            "batches": batches,
            "satisfied_batches": satisfied,
            "satisfaction_rate": satisfied / batches,
            "constraints": constraints_out,
        }

    merged = {
        "schema_version": schema_version,
        "utility": weighted_satisfied / weighted_batches,
        "profiles": out_profiles,
        "batch_count": len(all_seeds),
        "expected_problems_per_batch": problems_per_batch,
        "expected_seeds": sorted(all_seeds),
        "mode": "personalized_hard_requirements",
        "merged_from_shards": [str(p.name) for p in shard_dirs],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(shard_dirs)} shards ({len(all_seeds)} seeds total) -> {output_path}")
    print(f"utility: {merged['utility']:.4f}")


if __name__ == "__main__":
    main()
