from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "experiment_aime.json"
MAX_EVALUATION_WORKERS = 60


def validate_evaluation_workers(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_workers must be an integer in [1, 60]") from exc
    if isinstance(value, bool) or parsed < 1 or parsed > MAX_EVALUATION_WORKERS:
        raise ValueError("evaluation_workers must be in [1, 60]")
    return parsed


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "person_tts_autoresearch_config_v1":
        raise ValueError(f"Unsupported experiment config: {config_path}")
    if "evaluation_workers" in config:
        config["evaluation_workers"] = validate_evaluation_workers(
            config["evaluation_workers"]
        )
    config["_config_path"] = str(config_path)
    config["_root"] = str(ROOT)
    return config


def resolve_from_root(config: Dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_root"]) / path
    return path.resolve()


def model_weights(config: Dict[str, Any]) -> Dict[str, float]:
    return {str(item["name"]): float(item["weight"]) for item in config["models"]}
