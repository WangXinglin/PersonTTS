from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config_utils import load_config
from evaluation.runner import evaluate
from evaluation.requirements import RequirementProfile, load_requirement_profiles
from policies.seeds import BASELINE_REGISTRY


def _csv_values(raw: str, converter) -> list:
    values = [converter(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("grid arguments must contain at least one value")
    return values


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _profile_name(algorithm: str, model: str, width: int, depth: int, refines: int) -> str:
    safe_model = model.replace("/", "_")
    return f"{algorithm}__{safe_model}__w{width}__d{depth}__r{refines}"


def _write_or_validate_profile(path: Path, profile: Dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != profile:
            raise ValueError(
                f"baseline output already contains a different configuration: {path}"
            )
        return
    _atomic_json(path, profile)


def run_grid(
    *,
    config: Dict[str, Any],
    output_root: Path,
    split: str,
    algorithms: Iterable[str],
    models: Iterable[str],
    widths: Iterable[int],
    depths: Iterable[int],
    refine_rounds: Iterable[int],
    seed_count: int,
    beta_grid: List[float],
    confidence_threshold: float,
    min_samples: int,
    window_size: int,
    warmup_checkpoints: int,
    stable_window: int,
    prune_patience: int,
    max_weighted_compute: float | None,
    workers: int | None = None,
    requirement_profiles: List[RequirementProfile] | None = None,
) -> Dict[str, Any]:
    matrix_path = output_root / "matrix_summary.json"
    results: Dict[str, Any] = (
        json.loads(matrix_path.read_text(encoding="utf-8"))
        if matrix_path.is_file()
        else {}
    )
    for algorithm, model, width, depth, refines in product(
        algorithms, models, widths, depths, refine_rounds
    ):
        try:
            policy_class = BASELINE_REGISTRY[algorithm]
        except KeyError as exc:
            raise ValueError(f"unknown baseline: {algorithm}") from exc
        policy_config = {
            "resources": {
                "model_id": model,
                "max_width": int(width),
                "checkpoints_per_stage": int(depth),
                "max_refine_rounds": int(refines),
            },
            "asc": {
                "confidence_threshold": confidence_threshold,
                "min_samples": min_samples,
            },
            "esc": {"window_size": window_size},
            "parallel_probe": {
                "warmup_checkpoints": warmup_checkpoints,
                "stable_window": stable_window,
                "prune_patience": prune_patience,
            },
        }
        # Validate before creating any output for an invalid configuration.
        policy_class(config=policy_config)
        profile_name = _profile_name(algorithm, model, width, depth, refines)
        profile_root = output_root / profile_name
        _write_or_validate_profile(
            profile_root / "profile.json",
            {"algorithm": algorithm, **policy_config, "split": split},
        )
        summary = evaluate(
            config=config,
            policy_class=policy_class,
            policy_config=policy_config,
            output_dir=profile_root,
            split=split,
            seed_count=seed_count,
            beta_grid=beta_grid,
            max_weighted_compute=max_weighted_compute,
            workers=workers,
            requirement_profiles=requirement_profiles,
        )
        results[profile_name] = summary
        _atomic_json(matrix_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit four-dimensional baseline grids")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[1] / "config/experiment_aime.json")
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=["discovery", "heldout"], default="discovery")
    parser.add_argument(
        "--algorithms",
        default="asc_configured_4d,esc_configured_4d,parallel_probe_sr",
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--widths", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--refine-rounds", required=True)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--betas", default="0.0")
    parser.add_argument(
        "--profiles",
        default=None,
        help="Requirement profile JSON; evaluates hard-requirement utility instead of beta",
    )
    parser.add_argument("--asc-threshold", type=float, default=0.95)
    parser.add_argument("--asc-min-samples", type=int, default=2)
    parser.add_argument("--esc-window-size", type=int, default=5)
    parser.add_argument("--pp-warmup", type=int, default=10)
    parser.add_argument("--pp-stable-window", type=int, default=14)
    parser.add_argument("--pp-prune-patience", type=int, default=8)
    parser.add_argument("--max-weighted-compute", type=float, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Episode worker processes per baseline profile",
    )
    args = parser.parse_args()
    profiles = (
        load_requirement_profiles(Path(args.profiles).resolve())
        if args.profiles
        else None
    )

    results = run_grid(
        config=load_config(args.config),
        output_root=Path(args.output_root).resolve(),
        split=args.split,
        algorithms=_csv_values(args.algorithms, str),
        models=_csv_values(args.models, str),
        widths=_csv_values(args.widths, int),
        depths=_csv_values(args.depths, int),
        refine_rounds=_csv_values(args.refine_rounds, int),
        seed_count=args.seed_count,
        beta_grid=None if profiles is not None else _csv_values(args.betas, float),
        confidence_threshold=args.asc_threshold,
        min_samples=args.asc_min_samples,
        window_size=args.esc_window_size,
        warmup_checkpoints=args.pp_warmup,
        stable_window=args.pp_stable_window,
        prune_patience=args.pp_prune_patience,
        max_weighted_compute=args.max_weighted_compute,
        workers=args.workers,
        requirement_profiles=profiles,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
