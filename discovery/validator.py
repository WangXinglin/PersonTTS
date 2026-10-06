from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


ALLOWED_IMPORTS = {
    "__future__",
    "collections",
    "math",
    "policies.base",
    "policies.seeds",
    "statistics",
    "typing",
}
FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
FORBIDDEN_TEXT = {
    "../",
    "aime_26",
    "gold_answer",
    "private_cache",
    "source_files",
    "../outputs",
}
PUBLIC_ENV_ATTRIBUTES = {
    "beta",
    "can_afford_complete_candidates",
    "continue_branch",
    "continue_many",
    "cost",
    "finish",
    "models",
    "observe",
    "prune",
    "requirements",
    "refine",
    "refine_many",
    "spawn",
    "spawn_many",
    "trace",
}
PUBLIC_MAPPING_FIELDS = {
    "answer_history",
    "branch_id",
    "branches",
    "can_continue",
    "can_refine",
    "checkpoint_index",
    "latest_answer",
    "pruned",
    "remaining_by_model",
    "stage_index",
    "stage_kind",
    "stage_terminal",
}


class CandidateValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    path: str
    bytes: int
    ast_nodes: int
    candidate_bases: List[str]

    def to_dict(self) -> dict:
        return {
            "status": "ok",
            "path": self.path,
            "bytes": self.bytes,
            "ast_nodes": self.ast_nodes,
            "candidate_bases": self.candidate_bases,
        }


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _fail(errors: Iterable[str]) -> None:
    values = list(errors)
    if values:
        raise CandidateValidationError("Candidate policy rejected:\n- " + "\n- ".join(values))


def validate_candidate_source(
    source: str,
    *,
    path: str = "candidate.py",
    max_bytes: int = 65536,
    allow_beta: bool = True,
) -> ValidationReport:
    errors: List[str] = []
    if len(source.encode("utf-8")) > max_bytes:
        errors.append(f"file exceeds {max_bytes} bytes")
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise CandidateValidationError(f"Candidate policy has invalid syntax: {exc}") from exc

    for token in FORBIDDEN_TEXT:
        if token in source.lower():
            errors.append(f"forbidden text reference: {token}")

    for statement in tree.body:
        if not isinstance(
            statement,
            (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign),
        ):
            errors.append(f"top-level {type(statement).__name__} is not allowed")

    candidate_classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CandidatePolicy"
    ]
    if len(candidate_classes) != 1:
        errors.append("exactly one CandidatePolicy class is required")
        candidate_class = None
    else:
        candidate_class = candidate_classes[0]

    solve_env_names = set()
    private_helper_names = set()
    if candidate_class is not None:
        private_helper_names = {
            node.name
            for node in candidate_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("_")
            and not node.name.startswith("__")
        }
        solve_methods = [
            node
            for node in candidate_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solve"
        ]
        if len(solve_methods) != 1:
            errors.append("CandidatePolicy must define exactly one solve(self, env) method")
        elif len(solve_methods[0].args.args) < 2:
            errors.append("solve must accept the replay environment argument")
        else:
            solve_env_names.add(solve_methods[0].args.args[1].arg)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    errors.append(f"import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module not in ALLOWED_IMPORTS:
                errors.append(f"import is not allowed: {'.' * node.level}{module}")
        elif isinstance(node, ast.Call):
            call_name = _name(node.func)
            if call_name.split(".")[-1] in FORBIDDEN_CALLS:
                errors.append(f"call is not allowed: {call_name}")
        elif isinstance(node, ast.Attribute):
            policy_local_helper = (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in private_helper_names
            )
            super_init = (
                node.attr == "__init__"
                and isinstance(node.value, ast.Call)
                and _name(node.value.func) == "super"
            )
            if node.attr.startswith("_") and not policy_local_helper and not super_init:
                errors.append(f"private attribute access is not allowed: {node.attr}")
            if not allow_beta and node.attr in {"beta", "budget_beta"}:
                errors.append("beta is not available in personalized discovery")
            if node.attr in PUBLIC_MAPPING_FIELDS:
                errors.append(
                    f"public observations are dictionaries; use bracket access for {node.attr}"
                )
            if isinstance(node.value, ast.Name) and node.value.id in solve_env_names:
                public_attributes = (
                    PUBLIC_ENV_ATTRIBUTES
                    if allow_beta
                    else PUBLIC_ENV_ATTRIBUTES - {"beta"}
                )
                if node.attr not in public_attributes:
                    errors.append(f"non-public replay attribute: {node.value.id}.{node.attr}")
        elif (
            not allow_beta
            and isinstance(node, ast.Constant)
            and node.value in {"beta", "budget_beta"}
        ):
            errors.append("beta is not available in personalized discovery")

    _fail(dict.fromkeys(errors))
    assert candidate_class is not None
    return ValidationReport(
        path=path,
        bytes=len(source.encode("utf-8")),
        ast_nodes=sum(1 for _ in ast.walk(tree)),
        candidate_bases=[_name(base) for base in candidate_class.bases],
    )


def validate_candidate(
    path: str | Path, *, max_bytes: int = 65536, allow_beta: bool = True
) -> ValidationReport:
    candidate = Path(path).resolve()
    return validate_candidate_source(
        candidate.read_text(encoding="utf-8"),
        path=str(candidate),
        max_bytes=max_bytes,
        allow_beta=allow_beta,
    )
