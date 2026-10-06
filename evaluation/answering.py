from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple


# The grader lives in `eval/`. It sits beside the package root here; in the
# original tree it sat one level above it. Try both rather than pinning one,
# so the same file works in either layout.
def _resolve_eval_root(project_root: Path) -> Path:
    candidates = (project_root / "eval", Path(__file__).resolve().parents[1] / "eval")
    for candidate in candidates:
        if (candidate / "sal" / "utils" / "math.py").is_file():
            return candidate
    return candidates[0]


@lru_cache(maxsize=4)
def _load_project_grader(project_root: Path):
    eval_root = _resolve_eval_root(project_root)
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    try:
        from sal.utils.math import extract_answer
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import the existing personTTS answer extractor from {eval_root}"
        ) from exc
    grader_path = eval_root / "evaluation" / "grader.py"
    module_name = "person_tts_existing_math_grader"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, grader_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load the existing personTTS grader: {grader_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    math_equal = getattr(module, "math_equal", None)
    if math_equal is None:
        raise RuntimeError(f"math_equal is missing from {grader_path}")
    return extract_answer, math_equal


def preflight_project_grader(project_root: Path) -> None:
    """Load the production grader before a worker claims any shard."""
    _load_project_grader(project_root)


def normalize_prediction(raw_prediction: Any, data_name: str, project_root: Path) -> Tuple[str, bool]:
    if raw_prediction is None:
        return "", True
    text = str(raw_prediction).strip()
    if not text or text.upper() == "ERR":
        return "", True
    lowered = text.lower()
    looks_full = any(
        marker in text or marker in lowered
        for marker in ("\\box", "\\boxed", "<think>", "</think>", "final answer", "\n")
    )
    if looks_full:
        extract_answer, _ = _load_project_grader(project_root)
        try:
            extracted = extract_answer(text, data_name)
            if extracted is not None:
                text = str(extracted).strip()
        except Exception:
            pass
    return text, not bool(text)


def is_correct(raw_prediction: Any, gold: Any, data_name: str, project_root: Path) -> bool:
    prediction, is_null = normalize_prediction(raw_prediction, data_name, project_root)
    if is_null:
        return False
    _, math_equal = _load_project_grader(project_root)
    try:
        return bool(math_equal(str(prediction), str(gold), timeout=True))
    except Exception:
        return False
