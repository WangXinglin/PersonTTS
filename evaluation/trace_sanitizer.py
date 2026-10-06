from __future__ import annotations

from typing import Any, Dict, Iterable, List


ANSWER_KEYS = {"answer"}
_OBSERVATION_FIELDS = (
    "stage_index",
    "stage_kind",
    "checkpoint_index",
    "stage_terminal",
    "can_continue",
    "can_refine",
    "pruned",
)


def public_structural_trace(
    trace: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    public: List[Dict[str, Any]] = []
    for source in trace:
        event = {
            key: source[key]
            for key in ("step", "action", "model", "branch_id", "answer")
            if key in source
        }
        observation = source.get("observation")
        if isinstance(observation, dict):
            for key in _OBSERVATION_FIELDS:
                if key in observation:
                    event[key] = observation[key]
        public.append(event)
    return public


def _anonymize_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    labels: Dict[str, str] = {}

    def label(value: Any) -> Any:
        if value in (None, "", "ERR"):
            return value
        text = str(value)
        if text not in labels:
            labels[text] = f"A{len(labels)}"
        return labels[text]

    def walk(value: Any, key: str = "") -> Any:
        if key in ANSWER_KEYS:
            return label(value)
        if isinstance(value, dict):
            return {child_key: walk(child, child_key) for child_key, child in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(events)


def anonymize_full_trace(trace: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return every executed public event for queryable full-trace analysis."""
    return _anonymize_events(public_structural_trace(trace))

