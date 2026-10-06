from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config
from evaluation.baseline_campaign import load_baseline_specs
from evaluation.requirements import load_requirement_profiles
from evaluation.runner import evaluate
from policies.seeds import BASELINE_REGISTRY

PROFILES_PATH = ROOT / "profiles" / "pilot_aime_20" / "profiles.json"
BASELINE_SPECS_PATH = ROOT / "config" / "baseline_specs.json"

VALID_POLICY_TYPES = ("optimized", "asc", "esc", "parallel_probe")
VALID_DATASETS = ("AIME_26", "AIME_24", "AIME_25")

BASELINE_ALGORITHM_MAP = {
    "asc": "asc_configured_4d",
    "esc": "esc_configured_4d",
    "parallel_probe": "parallel_probe_sr",
}

DATASET_PROBLEM_COUNT_SINGLE = 30


def _prepare_config(config, dataset=None, split="heldout"):
    if dataset is None:
        return config, split
    cfg = copy.deepcopy(config)
    datasets = [dataset]
    problem_count = DATASET_PROBLEM_COUNT_SINGLE * len(datasets)
    dataset_key = f"{split}_datasets"
    cfg[dataset_key] = datasets
    counts = cfg.get("expected_problem_count_by_split", {})
    counts[split] = problem_count
    cfg["expected_problem_count_by_split"] = counts
    return cfg, split


def _resolve_policy(args, campaign_dir):
    policy_path = None
    policy_class = None
    policy_config = None

    if args.policy_type == "optimized":
        profile_id = args.profile
        policy_path = campaign_dir / "results" / "policies" / f"{profile_id}.py"
        if not policy_path.is_file():
            raise SystemExit(
                f"optimized policy not found: {policy_path}\n"
                f"Has profile {profile_id} completed discovery?"
            )
    elif args.policy_type in BASELINE_ALGORITHM_MAP:
        algorithm = BASELINE_ALGORITHM_MAP[args.policy_type]
        policy_class = BASELINE_REGISTRY[algorithm]
        specs = {s["id"]: s for s in load_baseline_specs(BASELINE_SPECS_PATH)}
        if args.policy_type not in specs:
            raise SystemExit(
                f"baseline spec {args.policy_type!r} not found in {BASELINE_SPECS_PATH}"
            )
        policy_config = specs[args.policy_type]["policy_config"]
    else:
        raise SystemExit(f"unknown policy type: {args.policy_type}")

    return policy_path, policy_class, policy_config


def _output_dir(campaign, profile_or_all, policy_type, dataset=None, shard_label=None):
    parts = [ROOT, "artifacts", "heldout_evaluation", campaign, profile_or_all, policy_type]
    if dataset is not None:
        parts.append(dataset)
    if shard_label is not None:
        parts += ["shards", shard_label]
    return Path(*parts)


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else load_config()
    profiles_path = Path(args.profiles) if args.profiles else PROFILES_PATH
    all_profiles = load_requirement_profiles(profiles_path)
    campaign_dir = ROOT / "artifacts" / "profile_discovery" / args.campaign

    if args.all_profiles:
        if args.policy_type == "optimized":
            raise SystemExit("--all-profiles cannot be used with --policy-type optimized")
        target_profiles = all_profiles
        profile_label = "all_profiles"
    else:
        if not args.profile:
            raise SystemExit("either --profile or --all-profiles is required")
        target = [p for p in all_profiles if p.profile_id == args.profile]
        if not target:
            raise SystemExit(f"profile {args.profile!r} not found")
        target_profiles = target
        profile_label = args.profile

    split = getattr(args, "split", "heldout")
    cfg, effective_split = _prepare_config(config, args.dataset, split)
    output_dir = _output_dir(args.campaign, profile_label, args.policy_type, args.dataset,
                              shard_label=args.shard_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_path, policy_class, policy_config = _resolve_policy(args, campaign_dir)

    shard_seeds = range(args.seed_start, args.seed_start + args.seed_count)
    summary = evaluate(
        config=cfg,
        output_dir=output_dir,
        split=effective_split,
        policy_path=policy_path,
        policy_class=policy_class,
        policy_config=policy_config,
        seeds=list(shard_seeds),
        workers=args.workers,
        requirement_profiles=target_profiles,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_collect(args: argparse.Namespace) -> None:
    base = ROOT / "artifacts" / "heldout_evaluation" / args.campaign
    if not base.is_dir():
        raise SystemExit(f"campaign directory not found: {base}")

    rows = []
    for summary_path in sorted(base.rglob("summary.json")):
        rel = summary_path.relative_to(base)
        parts = rel.parts
        if len(parts) < 3:
            continue
        profile_or_all = parts[0]
        policy_type = parts[1]
        dataset = parts[2] if len(parts) == 4 else "combined"

        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARN: skipping {summary_path}: {exc}", file=sys.stderr)
            continue

        for pid, pdata in data.get("profiles", {}).items():
            rows.append({
                "profile_id": pid,
                "policy_type": policy_type,
                "dataset": dataset,
                "satisfaction_rate": pdata.get("satisfaction_rate"),
                "satisfied_batches": pdata.get("satisfied_batches"),
                "batches": pdata.get("batches"),
                "constraints": {
                    c: {"satisfaction_rate": v.get("satisfaction_rate"),
                        "mean_margin": v.get("mean_margin")}
                    for c, v in pdata.get("constraints", {}).items()
                },
            })

    if not rows:
        print("No completed shards found.", file=sys.stderr)
        return

    report = {
        "schema_version": "person_tts_heldout_report_v2",
        "campaign": args.campaign,
        "record_count": len(rows),
        "results": rows,
    }
    report_path = base / "heldout_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Report written to {report_path} ({len(rows)} records)\n")

    _print_main_table(rows)


def _print_main_table(rows):
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        key = (r["policy_type"], r["dataset"])
        if r["satisfaction_rate"] is not None:
            agg[key].append(r["satisfaction_rate"])

    datasets = sorted({r["dataset"] for r in rows})
    policy_types = ["optimized", "asc", "esc", "parallel_probe"]
    policy_types += sorted(p for p in {k[0] for k in agg} - set(policy_types))
    policy_types = [p for p in policy_types if any((p, d) in agg for d in datasets)]

    header_parts = [f"{'Policy':<30}"]
    for ds in datasets:
        header_parts.append(f"{ds:>12}")
    header = "  ".join(header_parts)
    print(header)
    print("-" * len(header))

    for pt in policy_types:
        parts = [f"{pt:<30}"]
        for ds in datasets:
            vals = agg.get((pt, ds), [])
            if vals:
                mean_sat = sum(vals) / len(vals)
                parts.append(f"{mean_sat:>12.4f}")
            else:
                parts.append(f"{'N/A':>12}")
        print("  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policies on the held-out test set (AIME_26)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Execute one evaluation shard")
    run_parser.add_argument("--campaign", required=True)
    run_parser.add_argument("--profile", default=None,
                            help="Profile ID (required for optimized, optional for baselines)")
    run_parser.add_argument("--all-profiles", action="store_true",
                            help="Evaluate with all profiles (baselines only)")
    run_parser.add_argument("--policy-type", required=True, choices=VALID_POLICY_TYPES)
    run_parser.add_argument("--dataset", default=None, choices=VALID_DATASETS,
                            help="Evaluate on a single dataset or group (default: combined)")
    run_parser.add_argument("--split", default="heldout", choices=("heldout", "discovery"),
                            help="Which data split to evaluate on")
    run_parser.add_argument("--workers", type=int, default=60)
    run_parser.add_argument("--seed-count", type=int, default=1000)
    run_parser.add_argument("--seed-start", type=int, default=0,
                            help="First seed in this shard's range (for seed-range sharding)")
    run_parser.add_argument("--shard-label", default=None,
                            help="Shard label; writes output under .../<policy_type>/[dataset/]shards/<label>/")
    run_parser.add_argument("--config", default=None,
                            help="Experiment config path (default: config/experiment_aime.json)")
    run_parser.add_argument("--profiles", default=None,
                            help="Requirement profiles path (default: profiles/pilot_aime_20/profiles.json)")

    collect_parser = sub.add_parser("collect", help="Aggregate results and print table")
    collect_parser.add_argument("--campaign", required=True)

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    main()
