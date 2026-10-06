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
from evaluation.cache_inventory import build_cache_inventory
from discovery.profile_campaign import (
    CampaignLayout,
    _manifest_settings,
    campaign_status,
    collect_policy_bank,
    initialize_campaign,
    reap_stale_claims,
    run_worker,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _campaign_root(args: argparse.Namespace) -> Path:
    if args.campaign_root:
        return Path(args.campaign_root).resolve()
    if not getattr(args, "campaign_id", None):
        raise ValueError("provide --campaign-root or --campaign-id")
    return (ROOT / "artifacts" / "profile_discovery" / args.campaign_id).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-machine scheduler for one-policy-per-profile discovery"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create or incrementally extend a campaign")
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--campaign-root", default=None)
    init.add_argument("--profiles", required=True)
    init.add_argument("--config", default=str(ROOT / "config" / "experiment_aime.json"))
    init.add_argument("--bucket-count", type=_positive, default=256)
    init.add_argument("--max-attempts", type=_positive, default=3)
    init.add_argument("--round-limit", type=_positive, default=None)
    init.add_argument(
        "--round-start-mode",
        choices=("autotts_reset", "incumbent_refine"),
        default=None,
    )
    init.add_argument("--seed-count", type=_positive, default=None)
    init.add_argument("--model", default="claude-sonnet-4-6")
    init.add_argument("--evaluation-workers", type=_positive, default=None)
    init.add_argument("--agent-max-turns", type=_positive, default=None)
    init.add_argument("--agent-max-budget-usd", type=float, default=None)
    init.add_argument("--agent-timeout-seconds", type=float, default=3600.0)
    init.add_argument("--agent-max-retries", type=int, default=0)
    init.add_argument("--agent-max-thinking-tokens", type=_positive, default=4000)
    init.add_argument("--agent-max-output-tokens", type=_positive, default=None)
    init.add_argument(
        "--no-strategy-guide",
        action="store_true",
        help="Do not inject STRATEGY_GUIDE.md (ablation control group)",
    )
    init.add_argument(
        "--prompt",
        default=None,
        help="Path to a custom discovery prompt file",
    )
    init.add_argument(
        "--warm-start-bank",
        default=None,
        help="Path to experience bank JSON for warm-start retrieval (OFF by default)",
    )

    worker = subparsers.add_parser("worker", help="Claim and execute profile shards")
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

    status = subparsers.add_parser("status", help="Report exact campaign state")
    status.add_argument("--campaign-id", default=None)
    status.add_argument("--campaign-root", default=None)
    status.add_argument("--details-limit", type=int, default=20)

    reap = subparsers.add_parser("reap", help="Boundedly recover stale worker claims")
    reap.add_argument("--campaign-id", default=None)
    reap.add_argument("--campaign-root", default=None)
    reap.add_argument("--lease-timeout-seconds", type=float, default=300.0)
    reap.add_argument("--max-bucket-probes", type=_positive, default=32)
    reap.add_argument("--max-task-probes", type=_positive, default=16)

    collect = subparsers.add_parser("collect", help="Build verified policy-bank index")
    collect.add_argument("--campaign-id", default=None)
    collect.add_argument("--campaign-root", default=None)
    collect.add_argument("--allow-partial", action="store_true")

    retry = subparsers.add_parser("retry-failed", help="Explicitly requeue failed shards")
    retry.add_argument("--campaign-id", default=None)
    retry.add_argument("--campaign-root", default=None)
    retry.add_argument("--profiles", default=None, help="Optional comma-separated profile IDs")
    retry.add_argument("--reset-attempts", action="store_true")

    args = parser.parse_args()
    campaign_root = _campaign_root(args)
    if args.command == "init":
        config_path = Path(args.config).resolve()
        config = load_config(config_path)
        settings = _manifest_settings(
            config_path=config_path,
            config=config,
            bucket_count=args.bucket_count,
            max_attempts=args.max_attempts,
            round_limit=args.round_limit or int(config["discovery_rounds"]),
            round_start_mode=args.round_start_mode
            or str(config.get("round_start_mode", "autotts_reset")),
            seed_count=args.seed_count or int(config["shuffle_seeds"]),
            model=args.model,
            evaluation_workers=args.evaluation_workers
            or int(config.get("evaluation_workers", 1)),
            agent_max_turns=args.agent_max_turns,
            agent_max_budget_usd=args.agent_max_budget_usd,
            agent_timeout_seconds=args.agent_timeout_seconds,
            agent_max_retries=args.agent_max_retries,
            agent_max_thinking_tokens=args.agent_max_thinking_tokens,
            agent_max_output_tokens=args.agent_max_output_tokens,
            inject_strategy_guide=not args.no_strategy_guide,
            prompt_path=(
                str(Path(args.prompt).resolve()) if args.prompt else None
            ),
            warm_start_bank_path=(
                str(Path(args.warm_start_bank).resolve())
                if args.warm_start_bank
                else None
            ),
        )
        settings["campaign_id"] = args.campaign_id
        cache_inventory = build_cache_inventory(
            config, split="discovery", require_latency=True
        )
        report = initialize_campaign(
            campaign_root=campaign_root,
            profiles_path=Path(args.profiles).resolve(),
            config_path=config_path,
            settings=settings,
            cache_inventory=cache_inventory,
        )
    elif args.command == "worker":
        report = run_worker(
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
        report = campaign_status(campaign_root, details_limit=args.details_limit)
    elif args.command == "reap":
        layout = CampaignLayout(campaign_root)
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
        report = reap_stale_claims(
            layout=layout,
            lease_timeout_seconds=args.lease_timeout_seconds,
            max_attempts=int(manifest["max_attempts"]),
            max_bucket_probes=args.max_bucket_probes,
            max_task_probes=args.max_task_probes,
        )
    elif args.command == "collect":
        report = collect_policy_bank(campaign_root, allow_partial=args.allow_partial)
    else:
        from discovery.profile_campaign import retry_failed_tasks

        report = retry_failed_tasks(
            campaign_root,
            profile_ids=args.profiles.split(",") if args.profiles else None,
            reset_attempts=args.reset_attempts,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
