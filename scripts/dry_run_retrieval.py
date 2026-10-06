#!/usr/bin/env python3
"""Dry-run warm-start retrieval: show which source profile each target would match.

Usage:
    python scripts/dry_run_retrieval.py \
        --bank artifacts/experience_banks/aime_bank_v1.json \
        --target-profiles profiles/pilot_aime_20/profiles.json \
        --target-config config/experiment_aime.json \
        --target-benchmark AIME
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery.warm_start import retrieve_seed_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run warm-start seed retrieval")
    parser.add_argument("--bank", required=True, help="Experience bank JSON")
    parser.add_argument("--target-profiles", required=True, help="Target profiles JSON")
    parser.add_argument("--target-config", required=True, help="Target experiment config")
    parser.add_argument("--target-benchmark", required=True, help="Target benchmark label")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    results = retrieve_seed_dry_run(
        bank_path=Path(args.bank),
        profiles_path=Path(args.target_profiles),
        target_benchmark=args.target_benchmark,
        target_config_path=Path(args.target_config),
    )

    print(f"{'Target':<16} {'Matched Source':<45} {'Dist':>8} {'AccΔ²':>8} {'LatΔ²':>8} {'CmpΔ²':>8}")
    print("-" * 100)
    for r in results:
        if r["matched"]:
            dims = r["per_dim_sq"]
            print(
                f"{r['target_profile_id']:<16} "
                f"{r['source_id']:<45} "
                f"{r['distance']:>8.4f} "
                f"{dims[0]:>8.4f} {dims[1]:>8.4f} {dims[2]:>8.4f}"
            )
        else:
            print(f"{r['target_profile_id']:<16} (no match: {r['reason']})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
