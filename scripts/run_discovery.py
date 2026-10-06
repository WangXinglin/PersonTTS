from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config
from discovery.workflow import run_discovery


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable Claude policy discovery")
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment_aime.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-limit", type=int, default=None, help="Pilot-only round cap")
    parser.add_argument(
        "--round-start-mode",
        choices=("autotts_reset", "incumbent_refine"),
        default=None,
        help="Round candidate start: shared template or current accepted incumbent",
    )
    parser.add_argument("--seed-count", type=int, default=None, help="Pilot-only seed cap")
    parser.add_argument("--betas", default=None, help="Pilot-only comma-separated beta subset")
    parser.add_argument(
        "--profiles",
        default=None,
        help="Persisted requirement profile JSON; replaces the beta grid",
    )
    parser.add_argument("--model", default=None, help="Optional Claude model passed to the SDK")
    parser.add_argument(
        "--publish-candidate",
        action="store_true",
        help="Copy the final discovered policy to policies/candidate.py",
    )
    parser.add_argument("--agent-max-turns", type=int, default=None)
    parser.add_argument("--agent-max-budget-usd", type=float, default=None)
    parser.add_argument("--agent-timeout-seconds", type=float, default=None)
    parser.add_argument("--agent-max-retries", type=int, default=0)
    parser.add_argument(
        "--agent-max-thinking-tokens",
        type=int,
        default=None,
        help="Positive per-response thinking token cap forwarded to Claude Code",
    )
    parser.add_argument(
        "--agent-max-output-tokens",
        type=int,
        default=None,
        help="Positive total response token cap forwarded to Claude Code",
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=None,
        help="Parallel episode workers inside each discovery-round evaluation",
    )
    parser.add_argument(
        "--baseline-summary",
        default=None,
        help="Optional aggregate seed summary exposed as a sanitized reference.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Path to a custom discovery prompt (default: discovery/prompts/discovery.md)",
    )
    parser.add_argument(
        "--no-strategy-guide",
        action="store_true",
        help="Do not inject STRATEGY_GUIDE.md into the agent workspace (for ablation experiments)",
    )
    parser.add_argument(
        "--warm-start-seed",
        default=None,
        help="Path to a warm-start seed policy retrieved from an experience bank",
    )
    args = parser.parse_args()
    if args.profiles and args.betas:
        parser.error("--profiles and --betas are mutually exclusive")
    config = load_config(args.config)
    betas = [float(value) for value in args.betas.split(",")] if args.betas else None
    run_root = run_discovery(
        config=config,
        run_id=args.run_id,
        round_limit=args.round_limit,
        seed_count=args.seed_count,
        beta_grid=betas,
        model=args.model,
        publish_candidate=args.publish_candidate,
        agent_max_turns=args.agent_max_turns,
        agent_max_budget_usd=args.agent_max_budget_usd,
        agent_timeout_seconds=args.agent_timeout_seconds,
        agent_max_retries=args.agent_max_retries,
        agent_max_thinking_tokens=args.agent_max_thinking_tokens,
        agent_max_output_tokens=args.agent_max_output_tokens,
        baseline_summary_path=(
            Path(args.baseline_summary).resolve()
            if args.baseline_summary
            else None
        ),
        evaluation_workers=args.evaluation_workers,
        requirement_profiles_path=(
            Path(args.profiles).resolve() if args.profiles else None
        ),
        round_start_mode=args.round_start_mode,
        prompt_path=(
            Path(args.prompt).resolve() if args.prompt else None
        ),
        inject_strategy_guide=not args.no_strategy_guide,
        warm_start_seed_path=(
            Path(args.warm_start_seed).resolve() if args.warm_start_seed else None
        ),
    )
    print(json.dumps({"status": "complete", "run_root": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()
