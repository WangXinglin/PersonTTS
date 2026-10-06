from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List


ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])/[^\s'\"`|;&<>()\[\]{}]+")
QUOTED_ROOT_PATH = re.compile(r"(['\"])/\1")


def _bash_root_paths(command: str) -> Iterable[str]:
    if QUOTED_ROOT_PATH.search(command):
        yield "/"
    try:
        lexer = shlex.shlex(
            command,
            posix=True,
            punctuation_chars="|&;()<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        if "/" in lexer:
            yield "/"
    except ValueError:
        # Malformed shell quoting cannot execute; non-root absolute paths are
        # still caught by ABSOLUTE_PATH.
        return


def _read_only_analysis_bash(command: str, scripts: set[str]) -> bool:
    if any(marker in command for marker in ("\n", "\r", ";", "`", "$(")):
        return False
    sanitized = command.replace("2>&1", "").replace("2>/dev/null", "")
    operators_removed = sanitized.replace("&&", "").replace("||", "")
    if ">" in operators_removed or "<" in operators_removed or "&" in operators_removed:
        return False
    logical_parts = [
        part.strip() for part in re.split(r"\s*(?:&&|\|\|)\s*", sanitized)
    ]
    if not logical_parts or any(not part for part in logical_parts):
        return False
    for logical_part in logical_parts:
        pipeline = [part.strip() for part in logical_part.split("|")]
        if len(pipeline) > 2 or any(not part for part in pipeline):
            return False
        try:
            first = shlex.split(pipeline[0])
            second = shlex.split(pipeline[1]) if len(pipeline) == 2 else []
        except ValueError:
            return False
        if not first:
            return False
        analyzer = (
            len(first) >= 3
            and first[0] in {"python", "python3"}
            and first[1] in scripts
            and first[2] in {"summary", "cases", "--source"}
        )
        listing = first[0] == "ls"
        message = first[0] == "echo"
        if not analyzer and not listing and not message:
            return False
        if second and not (
            second[0] == "head"
            and len(second) <= 2
            and (len(second) == 1 or re.fullmatch(r"-?\d+|-n\d+", second[1]))
        ):
            return False
    return True


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _path_strings(tool: str, tool_input: Any) -> Iterable[str]:
    if not isinstance(tool_input, dict):
        return
    if tool == "Bash":
        keys = ("command",)
    elif tool in {"Read", "Write", "Edit"}:
        keys = ("file_path",)
    elif tool in {"Glob", "Grep"}:
        keys = ("path",)
    else:
        keys = ()
    for key in keys:
        yield from _strings(tool_input.get(key))


def audit_tool_calls(
    calls: Iterable[Dict[str, Any]],
    *,
    allowed_tools: Iterable[str],
    forbidden_paths: Iterable[str],
    workspace: str | Path | None = None,
    allowed_write_paths: Iterable[str | Path] | None = None,
    read_only_bash_scripts: Iterable[str] | None = None,
) -> Dict[str, Any]:
    allowed = set(allowed_tools)
    forbidden = [value.lower().replace("\\", "/") for value in forbidden_paths]
    allowed_root = (
        str(Path(workspace).resolve()).lower().replace("\\", "/")
        if workspace is not None
        else None
    )
    permitted_writes = (
        {str(Path(value).resolve()).lower().replace("\\", "/") for value in allowed_write_paths}
        if allowed_write_paths is not None
        else None
    )
    bash_scripts = set(read_only_bash_scripts or [])
    violations: List[Dict[str, Any]] = []
    count = 0
    for index, call in enumerate(calls):
        count += 1
        tool = str(call.get("tool", ""))
        tool_input = call.get("input", {})
        if tool not in allowed:
            violations.append(
                {"call_index": index, "tool": tool, "reason": "tool_not_allowed"}
            )
        if tool == "Bash" and bash_scripts:
            command = tool_input.get("command") if isinstance(tool_input, dict) else None
            if not isinstance(command, str) or not _read_only_analysis_bash(
                command, bash_scripts
            ):
                violations.append(
                    {
                        "call_index": index,
                        "tool": tool,
                        "reason": "bash_not_read_only_analyzer",
                    }
                )
        if permitted_writes is not None and tool in {"Write", "Edit"}:
            raw_target = tool_input.get("file_path") if isinstance(tool_input, dict) else None
            if not isinstance(raw_target, str):
                normalized_target = "<missing>"
            else:
                target = Path(raw_target)
                if not target.is_absolute() and workspace is not None:
                    target = Path(workspace) / target
                normalized_target = str(target.resolve()).lower().replace("\\", "/")
            if normalized_target not in permitted_writes:
                violations.append(
                    {
                        "call_index": index,
                        "tool": tool,
                        "reason": "write_target_not_allowed",
                        "matches": [normalized_target],
                    }
                )
        for raw in _strings(tool_input):
            normalized = raw.lower().replace("\\", "/")
            matches = sorted({token for token in forbidden if token and token in normalized})
            if matches:
                violations.append(
                    {
                        "call_index": index,
                        "tool": tool,
                        "reason": "forbidden_path_reference",
                        "matches": matches,
                    }
                )
        if allowed_root is not None:
            absolute_paths = {
                match.group(0).lower().replace("\\", "/").rstrip(".,:")
                for raw in _path_strings(tool, tool_input)
                for match in ABSOLUTE_PATH.finditer(raw)
            }
            for raw in _path_strings(tool, tool_input):
                if tool == "Bash":
                    absolute_paths.update(_bash_root_paths(raw))
                elif raw.strip().replace("\\", "/") == "/":
                    absolute_paths.add("/")
            outside = sorted(
                path
                for path in absolute_paths
                if path != "/dev/null"
                and path != allowed_root
                and not path.startswith(allowed_root + "/")
            )
            if outside:
                violations.append(
                    {
                        "call_index": index,
                        "tool": tool,
                        "reason": "workspace_escape_path",
                        "matches": outside,
                    }
                )
    return {
        "status": "ok" if not violations else "rejected",
        "tool_call_count": count,
        "violations": violations,
    }


def assert_clean_audit(report: Dict[str, Any]) -> None:
    if report.get("violations"):
        raise RuntimeError(
            "Claude tool-call audit rejected this round: "
            + repr(report["violations"])
        )

