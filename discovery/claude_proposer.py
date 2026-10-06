from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import inspect
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ClaudeSDKUnavailable(RuntimeError):
    pass


class ClaudeRoundTimeout(RuntimeError):
    pass


def preflight_claude_sdk() -> None:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
    except ImportError as exc:
        raise ClaudeSDKUnavailable(
            "the Claude Agent SDK dependency closure is unavailable in this Python "
            f"environment: {exc}"
        ) from exc
    required_symbols = {
        "ClaudeAgentOptions": ClaudeAgentOptions,
        "HookMatcher": HookMatcher,
        "query": query,
    }
    invalid = [name for name, value in required_symbols.items() if not callable(value)]
    if invalid:
        raise ClaudeSDKUnavailable(
            "the Claude Agent SDK is missing callable symbols: " + ", ".join(invalid)
        )


def require_claude_sdk() -> None:
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError as exc:
        raise ClaudeSDKUnavailable(
            "claude-agent-sdk is not installed. Install the pinned optional dependency "
            "or provide it in the Claude Code environment before discovery."
        ) from exc


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    try:
        attributes = vars(value)
    except TypeError:
        return str(value)
    return {
        str(key): _jsonable(item)
        for key, item in attributes.items()
        if not str(key).startswith("_") and not callable(item)
    }


def _block_payload(block: Any) -> Dict[str, Any]:
    serialized = _jsonable(block)
    payload = serialized if isinstance(serialized, dict) else {"value": serialized}
    sdk_type = payload.pop("type", None)
    if sdk_type is not None:
        payload["sdk_type"] = sdk_type
    return {"type": type(block).__name__, **payload}


def _request_seed(
    *,
    prompt: str,
    system_prompt: str,
    cwd: Path,
    allowed_tools: List[str],
    model: Optional[str],
    max_turns: Optional[int],
    max_budget_usd: Optional[float],
    timeout_seconds: Optional[float],
    max_retries: int,
    max_thinking_tokens: Optional[int],
    max_output_tokens: Optional[int],
    claude_env: Dict[str, str],
) -> Dict[str, Any]:
    safe_environment_names = (
        "API_TIMEOUT_MS",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
        "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "CLAUDE_CODE_MAX_RETRIES",
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS",
        "MAX_THINKING_TOKENS",
    )
    request = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "cwd": str(cwd.resolve()),
        "allowed_tools": list(allowed_tools),
        "model": model,
        "max_turns": max_turns,
        "max_budget_usd": max_budget_usd,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "max_thinking_tokens": max_thinking_tokens,
        "max_output_tokens": max_output_tokens,
        "claude_env": {
            name: claude_env[name]
            for name in safe_environment_names
            if name in claude_env
        },
    }
    canonical = json.dumps(
        request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "personaltts_agent_request_seed_v1",
        "written_at": _timestamp(),
        "request_sha256": hashlib.sha256(canonical).hexdigest(),
        "request": request,
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sanitize_stderr(line: str) -> str:
    sanitized = line
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(name)
        if value:
            sanitized = sanitized.replace(value, "<redacted>")
    sanitized = re.sub(
        r"(?i)(authorization|x-api-key)([\"'=:\s]+)([^\s,;\"]+)",
        r"\1\2<redacted>",
        sanitized,
    )
    return sanitized[:4000]


def _stream_metadata(message: Any) -> Dict[str, Any]:
    raw_event = getattr(message, "event", None)
    if not isinstance(raw_event, dict):
        return {}
    metadata: Dict[str, Any] = {"stream_event_type": raw_event.get("type")}
    delta = raw_event.get("delta")
    if isinstance(delta, dict):
        metadata["stream_delta_type"] = delta.get("type")
    content_block = raw_event.get("content_block")
    if isinstance(content_block, dict):
        metadata["stream_block_type"] = content_block.get("type")
    return metadata


def _claude_env(
    max_retries: int,
    max_thinking_tokens: Optional[int] = None,
    request_timeout_seconds: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, str]:
    forwarded = {
        name: os.environ[name]
        for name in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_CUSTOM_HEADERS",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
            "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
            "CLAUDE_CODE_MAX_RETRIES",
        )
        if os.environ.get(name)
    }
    # The configured upstream rejects Claude Code's server-side compaction field.
    forwarded.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    # Claude Agent SDK appends /v1 itself; normalize legacy endpoint values that included it.
    base_url = forwarded.get("ANTHROPIC_BASE_URL")
    if base_url:
        normalized_base_url = base_url.rstrip("/")
        if normalized_base_url.endswith("/v1"):
            normalized_base_url = normalized_base_url[:-3]
        forwarded["ANTHROPIC_BASE_URL"] = normalized_base_url
    # A streaming error must not silently replay the same costly turn.
    forwarded.setdefault("CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK", "1")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    forwarded["CLAUDE_CODE_MAX_RETRIES"] = str(max_retries)
    if max_thinking_tokens is not None:
        if isinstance(max_thinking_tokens, bool) or not isinstance(
            max_thinking_tokens, int
        ) or max_thinking_tokens <= 0:
            raise ValueError("max_thinking_tokens must be a positive integer")
        forwarded["MAX_THINKING_TOKENS"] = str(max_thinking_tokens)
    if max_output_tokens is not None:
        if isinstance(max_output_tokens, bool) or not isinstance(
            max_output_tokens, int
        ) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        forwarded["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    if request_timeout_seconds is not None:
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive")
        timeout_ms = int(float(request_timeout_seconds) * 1000)
        forwarded["API_TIMEOUT_MS"] = str(timeout_ms)
        forwarded["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] = str(timeout_ms)
    return forwarded


async def run_claude_round(
    *,
    prompt: str,
    system_prompt: str,
    cwd: Path,
    allowed_tools: List[str],
    event_log: Path,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    max_retries: int = 0,
    max_thinking_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    heartbeat_path: Optional[Path] = None,
    stderr_log: Optional[Path] = None,
    tool_guard: Optional[Callable[[str, Any], Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    require_claude_sdk()
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError as exc:
        raise ClaudeSDKUnavailable(
            "claude-agent-sdk is not installed. Install the pinned optional dependency "
            "or provide it in the Claude Code environment before discovery."
        ) from exc

    blocked_tool_calls: Dict[str, str] = {}

    async def guard_tool_use(
        hook_input: Dict[str, Any],
        tool_use_id: Optional[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        del context
        if tool_guard is None:
            return {}
        tool = str(hook_input.get("tool_name", ""))
        tool_input = hook_input.get("tool_input", {})
        reason = tool_guard(tool, tool_input)
        if reason is None:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                }
            }
        call_id = str(tool_use_id or hook_input.get("tool_use_id") or "")
        blocked_tool_calls[call_id] = reason
        _append_jsonl(
            event_log,
            {
                "timestamp": _timestamp(),
                "message_type": "PreToolUseDenied",
                "content": [],
                "tool_use_id": call_id,
                "tool": tool,
                "reason": reason,
            },
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    claude_env = _claude_env(
        max_retries,
        max_thinking_tokens,
        None if timeout_seconds is None else float(timeout_seconds) + 300.0,
        max_output_tokens,
    )
    option_values: Dict[str, Any] = {
        "tools": allowed_tools,
        "allowed_tools": allowed_tools,
        "cwd": str(cwd.resolve()),
        "env": claude_env,
        "permission_mode": "acceptEdits",
        "system_prompt": system_prompt,
        "include_partial_messages": True,
    }
    if tool_guard is not None:
        from claude_agent_sdk import HookMatcher

        option_values["hooks"] = {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[guard_tool_use])]
        }
    if model:
        option_values["model"] = model
    if max_turns is not None:
        option_values["max_turns"] = int(max_turns)
    if max_budget_usd is not None:
        option_values["max_budget_usd"] = float(max_budget_usd)
    stderr_path = stderr_log or event_log.with_name("agent_stderr.jsonl")
    _atomic_json(
        event_log.with_name("agent_request_seed.json"),
        _request_seed(
            prompt=prompt,
            system_prompt=system_prompt,
            cwd=cwd,
            allowed_tools=allowed_tools,
            model=model,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_thinking_tokens=max_thinking_tokens,
            max_output_tokens=max_output_tokens,
            claude_env=claude_env,
        ),
    )

    def record_stderr(line: str) -> None:
        _append_jsonl(
            stderr_path,
            {"timestamp": _timestamp(), "line": _sanitize_stderr(str(line))},
        )

    option_values["stderr"] = record_stderr
    parameters = inspect.signature(ClaudeAgentOptions).parameters
    options = ClaudeAgentOptions(
        **{key: value for key, value in option_values.items() if key in parameters}
    )

    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    calls: List[Dict[str, Any]] = []
    heartbeat = heartbeat_path or event_log.with_name("agent_heartbeat.json")
    started = time.monotonic()
    progress: Dict[str, Any] = {
        "status": "running",
        "started_at": _timestamp(),
        "last_event_at": None,
        "last_message_type": None,
        "event_count": 0,
        "tool_call_count": 0,
        "max_turns": max_turns,
        "max_budget_usd": max_budget_usd,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "max_thinking_tokens": max_thinking_tokens,
        "max_output_tokens": max_output_tokens,
        "pre_tool_guard_enabled": tool_guard is not None,
        "pre_tool_denied_count": 0,
    }

    async def write_heartbeat() -> None:
        while True:
            progress["heartbeat_at"] = _timestamp()
            progress["elapsed_seconds"] = round(time.monotonic() - started, 3)
            _atomic_json(heartbeat, progress)
            await asyncio.sleep(5.0)

    async def collect() -> None:
        async for message in query(prompt=prompt, options=options):
            blocks = [
                _block_payload(block)
                for block in (getattr(message, "content", None) or [])
            ]
            message_type = type(message).__name__
            event = {
                "timestamp": _timestamp(),
                "message_type": message_type,
                "content": blocks,
                **_stream_metadata(message),
            }
            for name in ("duration_ms", "duration_api_ms", "num_turns", "total_cost_usd"):
                if hasattr(message, name):
                    value = getattr(message, name)
                    event[name] = value
                    progress[name] = value
            _append_jsonl(event_log, event)
            progress["event_count"] += 1
            progress["last_event_at"] = event["timestamp"]
            progress["last_message_type"] = message_type
            for block in blocks:
                if "name" in block and "input" in block:
                    calls.append(
                        {
                            "tool": str(block["name"]),
                            "input": block["input"],
                            "id": block.get("id"),
                        }
                    )
                    progress["tool_call_count"] += 1

    heartbeat_task = asyncio.create_task(write_heartbeat())
    try:
        if timeout_seconds is None:
            await collect()
        else:
            await asyncio.wait_for(collect(), timeout=float(timeout_seconds))
        progress["status"] = "complete"
    except asyncio.TimeoutError as exc:
        progress["status"] = "timeout"
        progress["error"] = (
            f"Claude round exceeded wall timeout of {float(timeout_seconds):.1f} seconds"
        )
        _append_jsonl(
            event_log,
            {
                "timestamp": _timestamp(),
                "message_type": "ClaudeRoundTimeout",
                "content": [],
                "timeout_seconds": timeout_seconds,
            },
        )
        raise ClaudeRoundTimeout(progress["error"]) from exc
    except Exception as exc:
        progress["status"] = "failed"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        progress["heartbeat_at"] = _timestamp()
        progress["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _atomic_json(heartbeat, progress)
    for call in calls:
        call_id = str(call.get("id") or "")
        if call_id in blocked_tool_calls:
            call["blocked"] = True
            call["block_reason"] = blocked_tool_calls[call_id]
    progress["pre_tool_denied_count"] = len(blocked_tool_calls)
    _atomic_json(heartbeat, progress)
    return calls
