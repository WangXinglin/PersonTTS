from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config
from evaluation.baseline_campaign import load_baseline_specs
from evaluation.runner import evaluate
from policies.seeds import BASELINE_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one baseline/seed shard")
    parser.add_argument("--config", required=True)
    parser.add_argument("--specs", required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split", choices=("discovery", "heldout"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", required=True, type=float)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    specs = {spec["id"]: spec for spec in load_baseline_specs(Path(args.specs))}
    try:
        spec = specs[args.baseline_id]
    except KeyError as exc:
        raise ValueError(f"unknown baseline id: {args.baseline_id}") from exc
    summary = evaluate(
        config=load_config(args.config),
        output_dir=Path(args.output_dir).resolve(),
        split=args.split,
        policy_class=BASELINE_REGISTRY[spec["algorithm"]],
        policy_config=spec["policy_config"],
        seeds=[args.seed],
        beta_grid=[args.beta],
        workers=args.workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
