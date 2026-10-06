from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_utils import resolve_from_root, validate_evaluation_workers
from discovery.audit import assert_clean_audit, audit_tool_calls
from discovery.claude_proposer import require_claude_sdk, run_claude_round
from discovery.validator import (
    CandidateValidationError,
    validate_candidate,
    validate_candidate_source,
)
from evaluation.runner import evaluate
from evaluation.requirements import RequirementProfile, load_requirement_profiles
from evaluation.trace_analysis import (
    PUBLIC_TRACE_FILENAME,
    TRACE_SUMMARY_FILENAME,
    summarize_trace_records,
)
ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "discovery.md"
CANDIDATE_TEMPLATE_PATH = ROOT / "policies" / "candidate_template.py"
ROUND_START_MODES = {"autotts_reset", "incumbent_refine"}
API_PATH = Path(__file__).resolve().parent / "POLICY_API.md"
SEED_POLICY_PATH = ROOT / "policies" / "seeds.py"
POLICY_BASE_PATH = ROOT / "policies" / "base.py"
TRACE_ANALYZER_PATH = ROOT / "evaluation" / "trace_analysis.py"
TRACE_GUIDE_PATH = Path(__file__).resolve().parent / "TRACE_GUIDE.md"
STRATEGY_GUIDE_PATH = Path(__file__).resolve().parent / "strategy_guide" / "STRATEGY_GUIDE.md"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _agent_failure_context(event_log: Path, exc: Exception) -> Dict[str, Any]:
    """Classify SDK/CLI failures without persisting model or credential payloads."""
    context: Dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "exit_code": getattr(exc, "exit_code", None),
        "stderr_present": event_log.with_name("agent_stderr.jsonl").is_file(),
        "request_seed_present": event_log.with_name("agent_request_seed.json").is_file(),
    }
    tail: List[Dict[str, Any]] = []
    if event_log.is_file():
        with event_log.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tail.append(event)
                if len(tail) > 8:
                    tail.pop(0)
    context["last_events"] = [
        {
            "timestamp": event.get("timestamp"),
            "message_type": event.get("message_type"),
            "stream_event_type": event.get("stream_event_type"),
        }
        for event in tail
    ]
    operation_timeout = any(
        str(block.get("text", "")).strip() == "API Error: The operation timed out."
        for event in tail
        for block in event.get("content", [])
        if isinstance(block, dict) and block.get("type") == "TextBlock"
    )
    context["failure_class"] = (
        "upstream_operation_timeout"
        if operation_timeout
        else "claude_sdk_or_cli_failure"
    )
    return context


def _tool_calls_from_event_log(path: Path) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    denied: Dict[str, str] = {}
    if not path.is_file():
        return calls
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("message_type") == "PreToolUseDenied":
                denied[str(event.get("tool_use_id") or "")] = str(
                    event.get("reason") or "denied by pre-tool guard"
                )
            for block in event.get("content", []):
                if (
                    isinstance(block, dict)
                    and "name" in block
                    and "input" in block
                ):
                    calls.append(
                        {
                            "tool": str(block["name"]),
                            "input": block["input"],
                            "id": block.get("id"),
                        }
                    )
    for call in calls:
        call_id = str(call.get("id") or "")
        if call_id in denied:
            call["blocked"] = True
            call["block_reason"] = denied[call_id]
    return calls


def _recoverable_budget_result(heartbeat: Dict[str, Any]) -> bool:
    budget = heartbeat.get("max_budget_usd")
    cost = heartbeat.get("total_cost_usd")
    return (
        heartbeat.get("status") == "failed"
        and heartbeat.get("last_message_type") == "ResultMessage"
        and isinstance(budget, (int, float))
        and isinstance(cost, (int, float))
        and budget > 0
        and cost >= budget
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_atomic(source: Path, destination: Path) -> None:
    _atomic_text(destination, source.read_text(encoding="utf-8"))


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, destination)


def _publish_candidate(source: Path, destination: Path, *, enabled: bool) -> None:
    if enabled:
        _copy_atomic(source, destination)


def _proposal_improves_utility(
    proposal: Dict[str, Any], incumbent: Dict[str, Any]
) -> bool:
    return float(proposal["utility"]) > float(incumbent["utility"])


def _completed_history(rounds_root: Path) -> List[Dict[str, Any]]:
    history = []
    for complete_path in sorted(rounds_root.glob("round_*/complete.json")):
        complete = _read_json(complete_path)
        history.append(
            {
                "round": complete["round"],
                "candidate_sha256": complete["candidate_sha256"],
                "summary": complete["summary"],
                "accepted": complete.get("accepted", True),
                "acceptance_rule": complete.get("acceptance_rule", "legacy_unconditional"),
                "incumbent_after_summary": complete.get("incumbent_after_summary"),
            }
        )
    return history


def _initial_feedback() -> Dict[str, Any]:
    return {
        "summary": None,
        "note": "No earlier discovery round exists. Start from the supplied valid seed policy.",
    }


def _aggregate_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    if summary.get("mode") == "personalized_hard_requirements":
        return {
            "mode": "personalized_hard_requirements",
            "utility": float(summary["utility"]),
            "batch_count": int(summary["batch_count"]),
            "expected_problems_per_batch": int(summary["expected_problems_per_batch"]),
            "profiles": summary["profiles"],
        }
    groups = list(summary["overall"].values())
    episodes = sum(int(group["episodes"]) for group in groups)
    if episodes <= 0:
        raise ValueError("Reference summary contains no episodes")
    correct = sum(
        round(float(group["accuracy"]) * int(group["episodes"]))
        for group in groups
    )
    return {
        "episodes": episodes,
        "correct": correct,
        "accuracy": correct / episodes,
        "mean_raw_tokens": sum(
            float(group["avg_raw_tokens"]) * int(group["episodes"])
            for group in groups
        )
        / episodes,
        "mean_weighted_compute": sum(
            float(group["avg_weighted_compute"]) * int(group["episodes"])
            for group in groups
        )
        / episodes,
    }


def _with_reference_baseline(
    feedback: Dict[str, Any], reference_summary: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if reference_summary is None:
        return {
            key: value
            for key, value in feedback.items()
            if key != "decision_trace_samples"
        }
    baseline = _aggregate_summary(reference_summary)
    reference: Dict[str, Any] = {
        "scope": "Same discovery caches, beta grid, and shuffle seeds.",
        "aggregate": baseline,
    }
    current_summary = feedback.get("summary")
    if isinstance(current_summary, dict):
        current = _aggregate_summary(current_summary)
        reference["current_round_aggregate"] = current
        if current.get("mode") == "personalized_hard_requirements":
            if baseline.get("mode") != "personalized_hard_requirements":
                raise ValueError("personalized discovery requires a personalized baseline summary")
            reference["current_minus_baseline"] = {
                "utility": current["utility"] - baseline["utility"]
            }
        else:
            reference["current_minus_baseline"] = {
                "correct": current["correct"] - baseline["correct"],
                "accuracy": current["accuracy"] - baseline["accuracy"],
                "mean_raw_tokens": current["mean_raw_tokens"] - baseline["mean_raw_tokens"],
                "mean_weighted_compute": (
                    current["mean_weighted_compute"] - baseline["mean_weighted_compute"]
                ),
                "weighted_compute_relative_change": (
                    current["mean_weighted_compute"] / baseline["mean_weighted_compute"] - 1.0
                ),
            }
    result = {
        "summary": feedback.get("summary"),
        "reference_baseline": reference,
    }
    for key, value in feedback.items():
        if key not in result and key not in {
            "decision_trace_samples",
            "reference_baseline",
        }:
            result[key] = value
    return result


def _agent_system_prompt(workspace: Path) -> str:
    allowed_root = str(workspace.resolve())
    return (
        "You are the analysis-and-implementation Agent for one AutoTTS-style "
        "policy-discovery round. Analyze the supplied aggregate feedback and "
        "sanitized executed traces, then edit candidate.py in this same session. "
        f"You may write only candidate.py. Your only permitted workspace is {allowed_root}. "
        "For structured file tools, use absolute paths only under that exact workspace. "
        "For Bash, the working directory is already set: use relative paths only. "
        "Never use /home/work or inspect any parent, root, or other filesystem path."
    )


def _prepare_trace_inputs(
    *,
    workspace: Path,
    rounds_root: Path,
    reference_baseline_path: Optional[Path],
) -> List[str]:
    copied: List[str] = []
    for source, name in (
        (TRACE_ANALYZER_PATH, "TRACE_ANALYZER.py"),
        (TRACE_GUIDE_PATH, "TRACE_GUIDE.md"),
    ):
        _copy_atomic(source, workspace / name)
        copied.append(name)

    latest_trace: Optional[Path] = None
    latest_summary: Optional[Path] = None
    for complete_path in sorted(rounds_root.glob("round_*/complete.json")):
        round_name = complete_path.parent.name
        evaluation_root = complete_path.parent / "evaluation"
        trace_source = evaluation_root / PUBLIC_TRACE_FILENAME
        summary_source = evaluation_root / TRACE_SUMMARY_FILENAME
        if not trace_source.is_file() or not summary_source.is_file():
            continue
        trace_name = f"TRACE_HISTORY/{round_name}.jsonl"
        summary_name = f"TRACE_HISTORY/{round_name}.summary.json"
        _copy_file_atomic(trace_source, workspace / trace_name)
        _copy_file_atomic(summary_source, workspace / summary_name)
        copied.extend((trace_name, summary_name))
        latest_trace = trace_source
        latest_summary = summary_source

    if latest_trace is None or latest_summary is None:
        _atomic_text(workspace / "SEARCH_TRACES.jsonl", "")
        _atomic_json(workspace / "TRACE_SUMMARY.json", summarize_trace_records([]))
    else:
        _copy_file_atomic(latest_trace, workspace / "SEARCH_TRACES.jsonl")
        _copy_file_atomic(latest_summary, workspace / "TRACE_SUMMARY.json")
    copied.extend(("SEARCH_TRACES.jsonl", "TRACE_SUMMARY.json"))

    if reference_baseline_path is not None:
        baseline_root = reference_baseline_path.parent
        baseline_trace = baseline_root / PUBLIC_TRACE_FILENAME
        baseline_summary = baseline_root / TRACE_SUMMARY_FILENAME
        if baseline_trace.is_file() and baseline_summary.is_file():
            _copy_file_atomic(baseline_trace, workspace / "BASELINE_TRACES.jsonl")
            _copy_file_atomic(
                baseline_summary, workspace / "BASELINE_TRACE_SUMMARY.json"
            )
            copied.extend(
                ("BASELINE_TRACES.jsonl", "BASELINE_TRACE_SUMMARY.json")
            )
    return copied



def _prepare_candidate_history(*, workspace: Path, rounds_root: Path) -> List[str]:
    copied: List[str] = []
    for complete_path in sorted(rounds_root.glob("round_*/complete.json")):
        proposal = complete_path.parent / "candidate.py"
        if not proposal.is_file():
            continue
        name = f"CANDIDATE_HISTORY/{complete_path.parent.name}.py"
        _copy_file_atomic(proposal, workspace / name)
        copied.append(name)
    return copied


def _copy_public_policy_inputs(
    *,
    workspace: Path,
    candidate_source: Path,
    feedback: Dict[str, Any],
    history: List[Dict[str, Any]],
    requirement_profiles_path: Optional[Path],
    inject_strategy_guide: bool = True,
    warm_start_note: Optional[str] = None,
) -> List[str]:
    _copy_atomic(candidate_source, workspace / "candidate.py")
    _copy_atomic(API_PATH, workspace / "POLICY_API.md")
    _copy_atomic(SEED_POLICY_PATH, workspace / "SEED_POLICY.py")
    _copy_atomic(POLICY_BASE_PATH, workspace / "POLICY_BASE.py")
    _atomic_json(workspace / "feedback.json", feedback)
    _atomic_json(workspace / "history.json", history)
    names = [
        "candidate.py",
        "POLICY_API.md",
        "SEED_POLICY.py",
        "POLICY_BASE.py",
        "feedback.json",
        "history.json",
    ]
    if requirement_profiles_path is not None:
        _copy_atomic(requirement_profiles_path, workspace / "REQUIREMENT_PROFILES.json")
        names.append("REQUIREMENT_PROFILES.json")
    if inject_strategy_guide and STRATEGY_GUIDE_PATH.is_file():
        _copy_atomic(STRATEGY_GUIDE_PATH, workspace / "STRATEGY_GUIDE.md")
        names.append("STRATEGY_GUIDE.md")
    if warm_start_note is not None:
        _atomic_text(workspace / "WARM_START_NOTE.md", warm_start_note)
        names.append("WARM_START_NOTE.md")
    return names


def _round_candidate_source(
    *,
    round_start_mode: str,
    current_candidate: Path,
    reset_seed_path: Optional[Path] = None,
) -> Path:
    if round_start_mode == "autotts_reset":
        return reset_seed_path if reset_seed_path is not None else CANDIDATE_TEMPLATE_PATH
    if round_start_mode == "incumbent_refine":
        return current_candidate
    raise ValueError(f"unsupported round_start_mode: {round_start_mode}")


def _prepare_round(
    *,
    round_index: int,
    round_root: Path,
    rounds_root: Path,
    candidate_source: Path,
    round_start_mode: str,
    previous_feedback: Optional[Path],
    reference_baseline: Optional[Dict[str, Any]],
    reference_baseline_path: Optional[Path],
    requirement_profiles_path: Optional[Path] = None,
    inject_strategy_guide: bool = True,
    warm_start_note: Optional[str] = None,
) -> Dict[str, Any]:
    workspace = round_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    feedback = _read_json(previous_feedback) if previous_feedback else _initial_feedback()
    feedback = _with_reference_baseline(feedback, reference_baseline)
    names = _copy_public_policy_inputs(
        workspace=workspace,
        candidate_source=candidate_source,
        feedback=feedback,
        history=_completed_history(rounds_root),
        requirement_profiles_path=requirement_profiles_path,
        inject_strategy_guide=inject_strategy_guide,
        warm_start_note=warm_start_note,
    )
    names.extend(
        _prepare_trace_inputs(
            workspace=workspace,
            rounds_root=rounds_root,
            reference_baseline_path=reference_baseline_path,
        )
    )
    names.extend(
        _prepare_candidate_history(workspace=workspace, rounds_root=rounds_root)
    )
    immutable_names = [name for name in names if name != "candidate.py"]
    state = {
        "round": round_index,
        "status": "prepared",
        "workspace": str(workspace),
        "round_start_mode": round_start_mode,
        "candidate_input_sha256": _sha256(workspace / "candidate.py"),
        "immutable_hashes": {
            name: _sha256(workspace / name) for name in immutable_names
        },
        "trace_inputs": sorted(
            name for name in names if name.endswith(".jsonl")
        ),
        "candidate_history": sorted(
            name for name in names if name.startswith("CANDIDATE_HISTORY/")
        ),
    }
    _atomic_json(round_root / "state.json", state)
    return state


def _assert_inputs_unchanged(workspace: Path, hashes: Dict[str, str]) -> None:
    changed = [
        name
        for name, expected in hashes.items()
        if not (workspace / name).is_file() or _sha256(workspace / name) != expected
    ]
    if changed:
        raise RuntimeError(f"Claude modified read-only discovery inputs: {changed}")


def _agent_allowed_tools(config: Dict[str, Any]) -> List[str]:
    allowed = [
        tool
        for tool in config["claude_allowed_tools"]
        if tool in {"Read", "Grep", "Glob", "Bash", "Edit", "Write"}
    ]
    if "Bash" not in allowed or not {"Edit", "Write"}.intersection(allowed):
        raise ValueError("Single-session discovery requires Bash and Edit or Write")
    return allowed


def _is_direct_trace_access(tool: str, tool_input: Any) -> bool:
    if tool not in {"Read", "Grep"} or not isinstance(tool_input, dict):
        return False
    key = "file_path" if tool == "Read" else "path"
    raw = tool_input.get(key)
    return isinstance(raw, str) and Path(raw).suffix.lower() == ".jsonl"


def _audit_agent_calls(
    *, config: Dict[str, Any], calls: List[Dict[str, Any]], workspace: Path
) -> Dict[str, Any]:
    blocked_calls = [call for call in calls if call.get("blocked")]
    executed_calls = [call for call in calls if not call.get("blocked")]
    report = audit_tool_calls(
        executed_calls,
        allowed_tools=_agent_allowed_tools(config),
        forbidden_paths=list(config["claude_forbidden_paths"])
        + [
            "../",
            str(resolve_from_root(config, config["source_output_root"])),
            str(resolve_from_root(config, config["private_cache_root"])),
        ],
        workspace=workspace,
        allowed_write_paths=[workspace / "candidate.py"],
        read_only_bash_scripts=["TRACE_ANALYZER.py"],
    )
    for index, call in enumerate(executed_calls):
        if _is_direct_trace_access(str(call.get("tool", "")), call.get("input", {})):
            report["violations"].append(
                {
                    "call_index": index,
                    "tool": "Read",
                    "reason": "raw_trace_direct_read_not_allowed",
                }
            )
    report["status"] = "ok" if not report["violations"] else "rejected"
    report["blocked_before_execution"] = [
        {
            "tool": call.get("tool"),
            "id": call.get("id"),
            "reason": call.get("block_reason"),
        }
        for call in blocked_calls
    ]
    report["attempted_tool_call_count"] = len(calls)
    return report


def _agent_paths(round_root: Path) -> Dict[str, Path]:
    return {
        "events": round_root / "agent_events.jsonl",
        "heartbeat": round_root / "agent_heartbeat.json",
        "stderr": round_root / "agent_stderr.jsonl",
        "tool_calls": round_root / "agent_tool_calls.json",
        "audit": round_root / "agent_audit.json",
        "failure": round_root / "agent_failure.json",
    }


def _recover_agent_calls(
    *, round_index: int, round_root: Path, state_status: str
) -> tuple[List[Dict[str, Any]], bool]:
    paths = _agent_paths(round_root)
    heartbeat = _read_json(paths["heartbeat"]) if paths["heartbeat"].is_file() else {}
    if heartbeat.get("status") == "complete":
        if paths["tool_calls"].is_file():
            return list(_read_json(paths["tool_calls"])), False
        calls = _tool_calls_from_event_log(paths["events"])
        if calls:
            _atomic_json(paths["tool_calls"], calls)
            return calls, False
        raise RuntimeError(
            f"Round {round_index} Agent completed, but no tool calls could be "
            "recovered from its event log."
        )
    if state_status == "agent_failed" and _recoverable_budget_result(heartbeat):
        calls = _tool_calls_from_event_log(paths["events"])
        if not calls:
            raise RuntimeError(
                f"Round {round_index} reached the Claude budget, but no persisted "
                "tool calls could be recovered."
            )
        _atomic_json(paths["tool_calls"], calls)
        return calls, True
    raise RuntimeError(
        f"Round {round_index} has an interrupted or failed Claude Agent call. "
        "Refusing to issue a second call for this round; inspect and explicitly "
        "archive the round first."
    )


def _candidate_source_from_tool_input(
    *, workspace: Path, tool: str, tool_input: Any
) -> tuple[Optional[str], Optional[str]]:
    if tool not in {"Write", "Edit"} or not isinstance(tool_input, dict):
        return None, None
    target = tool_input.get("file_path")
    if not isinstance(target, str):
        return None, "Candidate write is missing file_path"
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = workspace / target_path
    if target_path.resolve() != (workspace / "candidate.py").resolve():
        return None, None
    if tool == "Write":
        content = tool_input.get("content")
        if not isinstance(content, str):
            return None, "Candidate Write requires string content"
        return content, None

    old = tool_input.get("old_string")
    new = tool_input.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        return None, "Candidate Edit requires non-empty old_string and string new_string"
    current = target_path.read_text(encoding="utf-8")
    occurrences = current.count(old)
    replace_all = bool(tool_input.get("replace_all", False))
    if occurrences == 0:
        return None, "Candidate Edit old_string was not found"
    if occurrences > 1 and not replace_all:
        return None, "Candidate Edit old_string is not unique"
    return (
        current.replace(old, new) if replace_all else current.replace(old, new, 1),
        None,
    )


def _candidate_validation_guard(
    *, workspace: Path, tool: str, tool_input: Any, allow_beta: bool = True
) -> Optional[str]:
    source, input_error = _candidate_source_from_tool_input(
        workspace=workspace, tool=tool, tool_input=tool_input
    )
    if input_error is not None:
        return input_error
    if source is None:
        return None
    try:
        validate_candidate_source(
            source,
            path=str((workspace / "candidate.py").resolve()),
            allow_beta=allow_beta,
        )
    except CandidateValidationError as exc:
        return str(exc)
    return None


def _run_agent_round(
    *,
    config: Dict[str, Any],
    round_root: Path,
    workspace: Path,
    model: Optional[str],
    max_turns: Optional[int],
    max_budget_usd: Optional[float],
    timeout_seconds: Optional[float],
    max_retries: int,
    max_thinking_tokens: Optional[int],
    max_output_tokens: Optional[int],
    allow_beta: bool = True,
    prompt_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    paths = _agent_paths(round_root)

    def tool_guard(tool: str, tool_input: Any) -> Optional[str]:
        report = _audit_agent_calls(
            config=config,
            calls=[{"tool": tool, "input": tool_input}],
            workspace=workspace,
        )
        if report["status"] != "ok":
            reasons = sorted(
                {str(violation["reason"]) for violation in report["violations"]}
            )
            return (
                "Tool call denied by the discovery boundary: "
                + ", ".join(reasons)
                + ". Use TRACE_ANALYZER.py for JSONL traces and only the "
                "documented workspace-local tools and write target."
            )
        validation_error = _candidate_validation_guard(
            workspace=workspace,
            tool=tool,
            tool_input=tool_input,
            allow_beta=allow_beta,
        )
        if validation_error is not None:
            return (
                "Candidate write denied before execution. Fix every validator error "
                "and write candidate.py again. "
                + validation_error
            )
        return None

    return asyncio.run(
        run_claude_round(
            prompt=(prompt_path or DISCOVERY_PROMPT_PATH).read_text(encoding="utf-8"),
            system_prompt=_agent_system_prompt(workspace),
            cwd=workspace,
            allowed_tools=_agent_allowed_tools(config),
            event_log=paths["events"],
            model=model,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_thinking_tokens=max_thinking_tokens,
            max_output_tokens=max_output_tokens,
            heartbeat_path=paths["heartbeat"],
            stderr_log=paths["stderr"],
            tool_guard=tool_guard,
        )
    )


def run_discovery(
    *,
    config: Dict[str, Any],
    run_id: str,
    round_limit: Optional[int] = None,
    seed_count: Optional[int] = None,
    beta_grid: Optional[List[float]] = None,
    model: Optional[str] = None,
    publish_candidate: bool = False,
    agent_max_turns: Optional[int] = None,
    agent_max_budget_usd: Optional[float] = None,
    agent_timeout_seconds: Optional[float] = None,
    agent_max_retries: int = 0,
    agent_max_thinking_tokens: Optional[int] = None,
    agent_max_output_tokens: Optional[int] = None,
    baseline_summary_path: Optional[Path] = None,
    evaluation_workers: Optional[int] = None,
    requirement_profiles_path: Optional[Path] = None,
    round_start_mode: Optional[str] = None,
    prompt_path: Optional[Path] = None,
    inject_strategy_guide: bool = True,
    warm_start_seed_path: Optional[Path] = None,
) -> Path:
    if agent_max_thinking_tokens is not None and (
        isinstance(agent_max_thinking_tokens, bool)
        or not isinstance(agent_max_thinking_tokens, int)
        or agent_max_thinking_tokens <= 0
    ):
        raise ValueError("agent_max_thinking_tokens must be a positive integer")
    if agent_max_output_tokens is not None and (
        isinstance(agent_max_output_tokens, bool)
        or not isinstance(agent_max_output_tokens, int)
        or agent_max_output_tokens <= 0
    ):
        raise ValueError("agent_max_output_tokens must be a positive integer")
    configured_rounds = int(config["discovery_rounds"])
    total_rounds = configured_rounds if round_limit is None else min(configured_rounds, round_limit)
    if total_rounds < 1:
        raise ValueError("round_limit must be positive")
    effective_start_mode = round_start_mode or str(
        config.get("round_start_mode", "autotts_reset")
    )
    if effective_start_mode not in ROUND_START_MODES:
        raise ValueError(
            f"round_start_mode must be one of {sorted(ROUND_START_MODES)}"
        )

    resolved_baseline_summary = (
        baseline_summary_path.resolve() if baseline_summary_path is not None else None
    )
    reference_baseline = (
        _read_json(resolved_baseline_summary)
        if resolved_baseline_summary is not None
        else None
    )
    resolved_profiles_path = (
        requirement_profiles_path.resolve()
        if requirement_profiles_path is not None
        else None
    )
    requirement_profiles: Optional[List[RequirementProfile]] = (
        load_requirement_profiles(resolved_profiles_path)
        if resolved_profiles_path is not None
        else None
    )
    if requirement_profiles is not None and beta_grid is not None:
        raise ValueError("personalized discovery cannot also specify a beta grid")
    allow_beta = requirement_profiles is None

    validate_candidate(CANDIDATE_TEMPLATE_PATH, allow_beta=allow_beta)
    template_sha256 = _sha256(CANDIDATE_TEMPLATE_PATH)
    artifacts_root = resolve_from_root(config, "artifacts/discovery")
    run_root = artifacts_root / run_id
    rounds_root = run_root / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    effective_seed_count = seed_count if seed_count is not None else config["shuffle_seeds"]
    effective_beta_grid = (
        None
        if requirement_profiles is not None
        else (beta_grid if beta_grid is not None else config["beta_grid"])
    )
    effective_evaluation_workers = validate_evaluation_workers(
        evaluation_workers
        if evaluation_workers is not None
        else config.get("evaluation_workers", 1)
    )
    effective_limits = {
        "max_turns": agent_max_turns,
        "max_budget_usd": agent_max_budget_usd,
        "timeout_seconds": agent_timeout_seconds,
        "max_retries": agent_max_retries,
        "max_thinking_tokens": agent_max_thinking_tokens,
        "max_output_tokens": agent_max_output_tokens,
    }
    baseline_sha256 = (
        _sha256(resolved_baseline_summary)
        if resolved_baseline_summary is not None
        else None
    )
    profiles_sha256 = (
        _sha256(resolved_profiles_path) if resolved_profiles_path is not None else None
    )
    discovery_mode = (
        "personalized_hard_requirements"
        if requirement_profiles is not None
        else "legacy_beta"
    )
    immutable_manifest = {
        "run_id": run_id,
        "seed_count": effective_seed_count,
        "beta_grid": effective_beta_grid,
        "discovery_mode": discovery_mode,
        "requirement_profiles_sha256": profiles_sha256,
        "discovery_datasets": config["discovery_datasets"],
        "heldout_visible": False,
        "publish_candidate": bool(publish_candidate),
        "agent_limits": effective_limits,
        "proposal_pipeline": "single_session_autotts_v1",
        "round_start_mode": effective_start_mode,
        "candidate_template_sha256": template_sha256,
        "warm_start_seed_sha256": (
            _sha256(warm_start_seed_path.resolve())
            if warm_start_seed_path is not None
            else None
        ),
    }
    if not manifest_path.exists():
        _atomic_json(
            manifest_path,
            {
                **immutable_manifest,
                "configured_rounds": configured_rounds,
                "requested_rounds": total_rounds,
                "agent_model": model,
                "reference_baseline_sha256": baseline_sha256,
                "evaluation_workers": effective_evaluation_workers,
            },
        )
    else:
        manifest = _read_json(manifest_path)
        mismatches = {
            key: {"recorded": manifest.get(key), "requested": value}
            for key, value in immutable_manifest.items()
            if manifest.get(key) != value
        }
        recorded_model = manifest.get("agent_model")
        if recorded_model is not None and recorded_model != model:
            mismatches["agent_model"] = {
                "recorded": recorded_model,
                "requested": model,
            }
        recorded_baseline = manifest.get("reference_baseline_sha256")
        if recorded_baseline is not None and recorded_baseline != baseline_sha256:
            mismatches["reference_baseline_sha256"] = {
                "recorded": recorded_baseline,
                "requested": baseline_sha256,
            }
        if mismatches:
            raise ValueError(f"Resume parameters differ from manifest: {mismatches}")
        manifest["requested_rounds"] = max(
            int(manifest.get("requested_rounds", 0)), total_rounds
        )
        manifest["agent_model"] = model
        manifest["reference_baseline_sha256"] = baseline_sha256
        manifest["evaluation_workers"] = effective_evaluation_workers
        _atomic_json(manifest_path, manifest)

    live_candidate = ROOT / "policies" / "candidate.py"

    warm_start_note: Optional[str] = None
    if warm_start_seed_path is not None:
        warm_start_seed_path = warm_start_seed_path.resolve()
        validate_candidate(warm_start_seed_path, allow_beta=allow_beta)
        initial_policy_path = warm_start_seed_path
        initial_policy_sha256 = _sha256(warm_start_seed_path)
        seed_archive = run_root / "warm_start_seed.py"
        if not seed_archive.is_file():
            _copy_atomic(warm_start_seed_path, seed_archive)
        elif _sha256(seed_archive) != initial_policy_sha256:
            raise ValueError("warm_start_seed.py hash mismatch on resume")
        warm_start_note = (
            "This is an initial policy found for a similar source profile, not a fixed answer.\n"
            "Adapt its structure and parameters to the target requirements and feedback.\n"
            "The source profile score is not the target profile score.\n"
        )
    else:
        initial_policy_path = CANDIDATE_TEMPLATE_PATH
        initial_policy_sha256 = template_sha256

    current_candidate = initial_policy_path
    incumbent_summary: Optional[Dict[str, Any]] = None
    personalized_incumbent_summary_path: Optional[Path] = None
    if requirement_profiles is not None:
        initial_root = run_root / "initial_incumbent"
        incumbent_summary = evaluate(
            config=config,
            policy_path=initial_policy_path,
            output_dir=initial_root / "evaluation",
            split="discovery",
            seed_count=seed_count,
            requirement_profiles=requirement_profiles,
            workers=effective_evaluation_workers,
        )
        _atomic_json(
            initial_root / "complete.json",
            {
                "status": "complete",
                "stage": "init",
                "warm_start": warm_start_seed_path is not None,
                "candidate_sha256": initial_policy_sha256,
                "summary": incumbent_summary,
            },
        )
        personalized_incumbent_summary_path = initial_root / "evaluation" / "summary.json"
    previous_feedback: Optional[Path] = (
        run_root / "initial_incumbent" / "evaluation" / "public_feedback.json"
        if requirement_profiles is not None
        else None
    )

    for round_index in range(total_rounds):
        round_root = rounds_root / f"round_{round_index:02d}"
        state_path = round_root / "state.json"
        complete_path = round_root / "complete.json"
        if complete_path.is_file():
            complete = _read_json(complete_path)
            current_candidate = Path(
                complete.get("selected_candidate", str(CANDIDATE_TEMPLATE_PATH))
            )
            validate_candidate(current_candidate, allow_beta=allow_beta)
            if requirement_profiles is not None:
                incumbent_summary = complete["incumbent_after_summary"]
            previous_feedback = round_root / "evaluation" / "public_feedback.json"
            continue

        if state_path.is_file():
            state = _read_json(state_path)
        else:
            candidate_source = _round_candidate_source(
                round_start_mode=effective_start_mode,
                current_candidate=current_candidate,
                reset_seed_path=(
                    run_root / "warm_start_seed.py"
                    if warm_start_seed_path is not None
                    else None
                ),
            )
            state = _prepare_round(
                round_index=round_index,
                round_root=round_root,
                rounds_root=rounds_root,
                candidate_source=candidate_source,
                round_start_mode=effective_start_mode,
                previous_feedback=previous_feedback,
                reference_baseline=reference_baseline,
                reference_baseline_path=(
                    personalized_incumbent_summary_path
                    if requirement_profiles is not None
                    else resolved_baseline_summary
                ),
                requirement_profiles_path=resolved_profiles_path,
                inject_strategy_guide=inject_strategy_guide,
                warm_start_note=warm_start_note,
            )

        two_stage_statuses = {
            "analysis_prepared", "analysis_running", "analysis_rejected",
            "analysis_failed", "analysis_complete", "implementation_prepared",
            "implementation_running", "implementation_rejected",
            "implementation_failed", "implementation_complete",
        }
        if state["status"] in two_stage_statuses:
            raise RuntimeError(
                "This round was created by the retired two-stage pipeline. "
                "Start single-session discovery with a new run_id."
            )

        workspace = Path(state["workspace"])
        paths = _agent_paths(round_root)
        if state["status"] in {"agent_running", "agent_rejected", "agent_failed"}:
            calls, recovered_budget = _recover_agent_calls(
                round_index=round_index,
                round_root=round_root,
                state_status=state["status"],
            )
            audit = _audit_agent_calls(
                config=config, calls=calls, workspace=workspace
            )
            _atomic_json(paths["audit"], audit)
            assert_clean_audit(audit)
            _assert_inputs_unchanged(workspace, state["immutable_hashes"])
            validate_candidate(workspace / "candidate.py", allow_beta=allow_beta)
            state["status"] = "agent_complete"
            state["agent_audit"] = audit
            state["agent_recovered_completed_output"] = True
            if recovered_budget:
                state["agent_recovered_after_budget_exit"] = True
            _atomic_json(state_path, state)

        if state["status"] == "prepared":
            require_claude_sdk()
            state["status"] = "agent_running"
            _atomic_json(state_path, state)
            try:
                calls = _run_agent_round(
                    config=config,
                    round_root=round_root,
                    workspace=workspace,
                    model=model,
                    max_turns=agent_max_turns,
                    max_budget_usd=agent_max_budget_usd,
                    timeout_seconds=agent_timeout_seconds,
                    max_retries=agent_max_retries,
                    max_thinking_tokens=agent_max_thinking_tokens,
                    max_output_tokens=agent_max_output_tokens,
                    allow_beta=allow_beta,
                    prompt_path=prompt_path,
                )
            except Exception as exc:
                failure = {
                    "status": "agent_failed",
                    **_agent_failure_context(paths["events"], exc),
                }
                _atomic_json(paths["failure"], failure)
                state["status"] = "agent_failed"
                state["agent_error"] = failure
                _atomic_json(state_path, state)
                raise
            _atomic_json(paths["tool_calls"], calls)
            audit = _audit_agent_calls(
                config=config, calls=calls, workspace=workspace
            )
            _atomic_json(paths["audit"], audit)
            try:
                assert_clean_audit(audit)
                _assert_inputs_unchanged(workspace, state["immutable_hashes"])
                validate_candidate(workspace / "candidate.py", allow_beta=allow_beta)
            except Exception:
                state["status"] = "agent_rejected"
                state["agent_audit"] = audit
                _atomic_json(state_path, state)
                raise
            state["status"] = "agent_complete"
            state["agent_audit"] = audit
            _atomic_json(state_path, state)

        if state["status"] == "agent_complete":
            report = validate_candidate(
                workspace / "candidate.py", allow_beta=allow_beta
            )
            _atomic_json(round_root / "validation.json", report.to_dict())
            _copy_atomic(workspace / "candidate.py", round_root / "candidate.py")
            state["status"] = "validated"
            state["candidate_sha256"] = _sha256(round_root / "candidate.py")
            _atomic_json(state_path, state)

        if state["status"] == "validated":
            evaluation_dir = round_root / "evaluation"
            proposal_candidate = round_root / "candidate.py"
            summary = evaluate(
                config=config,
                policy_path=proposal_candidate,
                output_dir=evaluation_dir,
                split="discovery",
                seed_count=seed_count,
                beta_grid=beta_grid if requirement_profiles is None else None,
                workers=effective_evaluation_workers,
                requirement_profiles=requirement_profiles,
            )
            accepted = True
            incumbent_before = incumbent_summary
            if requirement_profiles is not None:
                if incumbent_summary is None:
                    raise RuntimeError("personalized discovery is missing incumbent utility")
                accepted = _proposal_improves_utility(summary, incumbent_summary)
                if accepted:
                    current_candidate = proposal_candidate
                    incumbent_summary = summary
            else:
                current_candidate = proposal_candidate
            _candidate_input_source = (
                "warm_start_seed"
                if warm_start_seed_path is not None
                and effective_start_mode == "autotts_reset"
                else "candidate_template"
                if effective_start_mode == "autotts_reset"
                else "incumbent"
            )
            complete = {
                "round": round_index,
                "status": "complete",
                "round_start_mode": effective_start_mode,
                "candidate_input_sha256": state["candidate_input_sha256"],
                "candidate_input_source": _candidate_input_source,
                "candidate_sha256": state["candidate_sha256"],
                "summary": summary,
                "accepted": accepted,
                "acceptance_rule": (
                    "proposal.utility > incumbent.utility"
                    if requirement_profiles is not None
                    else "legacy_unconditional"
                ),
                "incumbent_before_summary": incumbent_before,
                "incumbent_after_summary": incumbent_summary,
                "selected_candidate": str(current_candidate),
                "selected_candidate_sha256": _sha256(current_candidate),
            }
            _atomic_json(complete_path, complete)
            state["status"] = "complete"
            _atomic_json(state_path, state)
            previous_feedback = evaluation_dir / "public_feedback.json"
        elif state["status"] != "complete":
            raise RuntimeError(f"Unknown round state: {state['status']!r}")

    _atomic_json(
        run_root / "complete.json",
        {
            "status": "complete",
            "rounds_completed": total_rounds,
            "round_start_mode": effective_start_mode,
            "final_candidate": str(current_candidate),
            "final_candidate_sha256": _sha256(current_candidate),
            "published_candidate": str(live_candidate) if publish_candidate else None,
        },
    )
    _publish_candidate(current_candidate, live_candidate, enabled=publish_candidate)
    return run_root
