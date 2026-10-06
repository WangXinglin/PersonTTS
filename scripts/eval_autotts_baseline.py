#!/usr/bin/env python3
"""Evaluate AutoTTS discovery policies on the heldout test set.

Usage:
    # After AutoTTS discovery completes:
    .venv/bin/python scripts/eval_autotts_baseline.py \
        --run-id autotts_beta05 \
        --campaign autotts_aime \
        --dataset AIME_26 \
        --workers 60 --seed-count 1000
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config
from evaluation.requirements import load_requirement_profiles
from evaluation.runner import evaluate

PROFILES_PATH = ROOT / "profiles" / "pilot_aime_20" / "profiles.json"

DATASET_PROBLEM_COUNT_SINGLE = 30

SPLIT_MAP = {
    "AIME_26": "heldout",
}


def _prepare_config(config, dataset, split):
    cfg = copy.deepcopy(config)
    datasets = [dataset]
    problem_count = DATASET_PROBLEM_COUNT_SINGLE * len(datasets)
    dataset_key = f"{split}_datasets"
    cfg[dataset_key] = datasets
    counts = cfg.get("expected_problem_count_by_split", {})
    counts[split] = problem_count
    cfg["expected_problem_count_by_split"] = counts
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AutoTTS baseline policy")
    parser.add_argument("--run-id", required=True,
                        help="AutoTTS discovery run ID (e.g. autotts_beta05)")
    parser.add_argument("--campaign", required=True,
                        help="Campaign name for output directory grouping")
    parser.add_argument("--dataset", required=True,
                        choices=["AIME_26"])
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--seed-count", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First seed of this shard (seeds evaluated: [seed-start, seed-start+seed-count))")
    parser.add_argument("--shard-label", default=None,
                        help="Output subdirectory name for this shard (default: no sharding, use dataset dir directly)")
    parser.add_argument("--profile", default=None,
                        help="Single profile ID to evaluate (default: all profiles)")
    parser.add_argument("--config", default=None,
                        help="Experiment config path (default: config/experiment_aime.json)")
    parser.add_argument("--profiles", default=None,
                        help="Requirement profiles path (default: profiles/pilot_aime_20/profiles.json)")
    args = parser.parse_args()

    complete_path = ROOT / "artifacts" / "discovery" / args.run_id / "complete.json"
    if not complete_path.is_file():
        raise SystemExit(
            f"Discovery not complete: {complete_path}\n"
            f"Run AutoTTS discovery first with: BETA=... bash scripts/run_autotts_baseline.sh"
        )

    complete = json.loads(complete_path.read_text())
    policy_path = Path(complete["final_candidate"])
    if not policy_path.is_file():
        raise SystemExit(f"Final candidate policy not found: {policy_path}")

    print(f"policy: {policy_path}")
    print(f"dataset: {args.dataset}")

    config = load_config(args.config) if args.config else load_config()
    split = SPLIT_MAP[args.dataset]
    cfg = _prepare_config(config, args.dataset, split)

    profiles_path = Path(args.profiles) if args.profiles else PROFILES_PATH
    all_profiles = load_requirement_profiles(profiles_path)
    if args.profile:
        target_profiles = [p for p in all_profiles if p.profile_id == args.profile]
        if not target_profiles:
            raise SystemExit(f"profile {args.profile!r} not found")
        profile_label = args.profile
    else:
        target_profiles = all_profiles
        profile_label = "all_profiles"
    output_dir = (ROOT / "artifacts" / "heldout_evaluation" / args.campaign
                  / profile_label / args.run_id / args.dataset)
    if args.shard_label:
        output_dir = output_dir / "shards" / args.shard_label
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_seeds = range(args.seed_start, args.seed_start + args.seed_count)
    print(f"seeds: [{shard_seeds.start}, {shard_seeds.stop})")

    summary = evaluate(
        config=cfg,
        output_dir=output_dir,
        split=split,
        policy_path=policy_path,
        seeds=list(shard_seeds),
        workers=args.workers,
        requirement_profiles=target_profiles,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
