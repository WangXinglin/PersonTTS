from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, resolve_from_root
from replay.adapter import load_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate selected compact replay files")
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment_aime.json"))
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--indices", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    cache_root = resolve_from_root(config, config["private_cache_root"])
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    indices = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
    records = []
    for dataset in datasets:
        for index in indices:
            path = cache_root / dataset / f"{index}.json"
            problem = load_problem(path)
            records.append(
                {
                    "dataset": dataset,
                    "index": index,
                    "models": len(problem.models),
                    "branches": sum(len(branches) for branches in problem.models.values()),
                    "stages": sum(
                        len(branch.stages)
                        for branches in problem.models.values()
                        for branch in branches
                    ),
                    "checkpoints": sum(
                        len(stage.checkpoints)
                        for branches in problem.models.values()
                        for branch in branches
                        for stage in branch.stages
                    ),
                    "raw_tokens": sum(
                        checkpoint.raw_tokens
                        for branches in problem.models.values()
                        for branch in branches
                        for stage in branch.stages
                        for checkpoint in stage.checkpoints
                    )
                    + sum(
                        stage.feedback_tokens
                        for branches in problem.models.values()
                        for branch in branches
                        for stage in branch.stages
                    ),
                }
            )
    print(json.dumps({"status": "ok", "files": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
