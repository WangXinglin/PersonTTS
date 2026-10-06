from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config
from evaluation.baseline_campaign import (
    baseline_campaign_settings,
    baseline_campaign_status,
    collect_baseline_index,
    initialize_baseline_campaign,
    retry_failed_baseline_tasks,
    run_baseline_worker,
)
from discovery.profile_campaign import CampaignLayout, _read_json, reap_stale_claims
from evaluation.cache_inventory import build_cache_inventory


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _root(args: argparse.Namespace) -> Path:
    if args.campaign_root:
        return Path(args.campaign_root).resolve()
    if not getattr(args, "campaign_id", None):
        raise ValueError("provide --campaign-root or --campaign-id")
    return (ROOT / "artifacts" / "baseline_campaigns" / args.campaign_id).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-machine scheduler for baseline-by-seed evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--campaign-root", default=None)
    init.add_argument("--config", default=str(ROOT / "config" / "experiment_aime.json"))
    init.add_argument("--specs", default=str(ROOT / "config" / "baseline_specs.json"))
    init.add_argument("--split", choices=("discovery", "heldout"), default="discovery")
    init.add_argument("--seed-count", type=_positive, default=100)
    init.add_argument("--bucket-count", type=_positive, default=256)
    init.add_argument("--max-attempts", type=_positive, default=3)
    init.add_argument("--evaluation-workers", type=_positive, default=4)

    worker = commands.add_parser("worker")
    worker.add_argument("--campaign-id", default=None)
    worker.add_argument("--campaign-root", default=None)
    worker.add_argument("--worker-id", default=os.environ.get("TRIAL_ID"))
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--idle-exit-seconds", type=float, default=120.0)
    worker.add_argument("--heartbeat-interval-seconds", type=float, default=30.0)
    worker.add_argument("--lease-timeout-seconds", type=float, default=300.0)
    worker.add_argument("--shard-timeout-seconds", type=float, default=None)
    worker.add_argument("--max-bucket-probes", type=_positive, default=32)
    worker.add_argument("--max-task-probes", type=_positive, default=16)
    worker.add_argument("--reap-bucket-probes", type=_positive, default=4)
    worker.add_argument("--reap-task-probes", type=_positive, default=8)
    worker.add_argument("--retry-backoff-base-seconds", type=float, default=60.0)
    worker.add_argument("--retry-backoff-max-seconds", type=float, default=900.0)

    status = commands.add_parser("status")
    status.add_argument("--campaign-id", default=None)
    status.add_argument("--campaign-root", default=None)
    status.add_argument("--details-limit", type=int, default=20)

    reap = commands.add_parser("reap")
    reap.add_argument("--campaign-id", default=None)
    reap.add_argument("--campaign-root", default=None)
    reap.add_argument("--lease-timeout-seconds", type=float, default=300.0)
    reap.add_argument("--max-bucket-probes", type=_positive, default=32)
    reap.add_argument("--max-task-probes", type=_positive, default=16)

    retry = commands.add_parser("retry-failed")
    retry.add_argument("--campaign-id", default=None)
    retry.add_argument("--campaign-root", default=None)
    retry.add_argument("--tasks", default=None)
    retry.add_argument("--reset-attempts", action="store_true")

    collect = commands.add_parser("collect")
    collect.add_argument("--campaign-id", default=None)
    collect.add_argument("--campaign-root", default=None)
    collect.add_argument("--allow-partial", action="store_true")

    args = parser.parse_args()
    campaign_root = _root(args)
    if args.command == "init":
        settings = baseline_campaign_settings(
            config_path=Path(args.config).resolve(),
            specs_path=Path(args.specs).resolve(),
            campaign_id=args.campaign_id,
            seed_count=args.seed_count,
            bucket_count=args.bucket_count,
            max_attempts=args.max_attempts,
            evaluation_workers=args.evaluation_workers,
            split=args.split,
        )
        cache_inventory = build_cache_inventory(
            load_config(Path(args.config).resolve()),
            split=args.split,
            require_latency=True,
        )
        report = initialize_baseline_campaign(
            campaign_root=campaign_root,
            settings=settings,
            cache_inventory=cache_inventory,
        )
    elif args.command == "worker":
        report = run_baseline_worker(
            campaign_root=campaign_root,
            worker_id=args.worker_id,
            once=args.once,
            idle_exit_seconds=args.idle_exit_seconds,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            lease_timeout_seconds=args.lease_timeout_seconds,
            shard_timeout_seconds=args.shard_timeout_seconds,
            max_bucket_probes=args.max_bucket_probes,
            max_task_probes=args.max_task_probes,
            reap_bucket_probes=args.reap_bucket_probes,
            reap_task_probes=args.reap_task_probes,
            retry_backoff_base_seconds=args.retry_backoff_base_seconds,
            retry_backoff_max_seconds=args.retry_backoff_max_seconds,
        )
    elif args.command == "status":
        report = baseline_campaign_status(campaign_root, details_limit=args.details_limit)
    elif args.command == "reap":
        layout = CampaignLayout(campaign_root)
        manifest = _read_json(layout.manifest)
        report = reap_stale_claims(
            layout=layout,
            lease_timeout_seconds=args.lease_timeout_seconds,
            max_attempts=int(manifest["max_attempts"]),
            max_bucket_probes=args.max_bucket_probes,
            max_task_probes=args.max_task_probes,
        )
    elif args.command == "retry-failed":
        report = retry_failed_baseline_tasks(
            campaign_root,
            task_ids=args.tasks.split(",") if args.tasks else None,
            reset_attempts=args.reset_attempts,
        )
    else:
        report = collect_baseline_index(
            campaign_root, allow_partial=args.allow_partial
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
