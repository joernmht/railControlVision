"""JSON Schema pass: turn jsonschema errors into ``SCHEMA_INVALID`` issues."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from rail_vision_bench.schema.export import to_json_schema
from rail_vision_bench.schema.issues import IssueCode, ValidationIssue


def load_schema(path: Path) -> dict[str, Any]:
    """Load a JSON Schema file.

    Args:
        path: The schema file (normally ``schema/v0.json``).

    Returns:
        The parsed schema.

    Raises:
        ValueError: If the file does not hold a JSON object.
    """
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"{path}: expected a JSON object"
        raise ValueError(msg)
    return loaded


def schema_issues(
    doc: Mapping[str, Any], schema: Mapping[str, Any] | None = None
) -> list[ValidationIssue]:
    """Validate a raw document against the JSON Schema.

    Args:
        doc: The raw document (parsed JSON).
        schema: A schema to use instead of the in-memory export.

    Returns:
        One ``SCHEMA_INVALID`` error per jsonschema error, sorted by (path, message).
    """
    validator = Draft202012Validator(dict(schema) if schema is not None else to_json_schema())
    issues = [
        ValidationIssue(
            code=IssueCode.SCHEMA_INVALID,
            severity="error",
            message=error.message,
            path="/".join(str(part) for part in error.absolute_path),
        )
        for error in validator.iter_errors(dict(doc))
    ]
    return sorted(issues, key=lambda issue: (issue.path or "", issue.message))
