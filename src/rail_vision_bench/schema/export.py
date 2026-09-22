"""Deterministic JSON Schema (draft 2020-12) export of schema v0.

The installed package never reads ``schema/v0.json``; the schema is always
built in memory from the models. The committed file is the published contract
and is kept in sync by ``bench schema export --check``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from rail_vision_bench.schema.models import SceneAnnotation

SCHEMA_ID: Final = "https://github.com/joernmht/railControlVision/schema/v0.json"
SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"


def to_json_schema() -> dict[str, Any]:
    """Build the JSON Schema of :class:`SceneAnnotation` with ``$schema`` and ``$id``.

    Returns:
        The schema as a plain dictionary.
    """
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": SCHEMA_ID,
        **SceneAnnotation.model_json_schema(mode="validation"),
    }


def dump_json_schema() -> str:
    """Serialise the schema exactly as it is committed (stable key order, trailing newline).

    Returns:
        The schema text.
    """
    return json.dumps(to_json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json_schema(path: Path) -> None:
    """Write the schema text to ``path``, creating parent directories.

    Args:
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json_schema(), encoding="utf-8")


def check_json_schema(path: Path) -> bool:
    """Report whether the file at ``path`` is byte-identical to the current export.

    Args:
        path: The committed schema file.

    Returns:
        True when the file matches the models.
    """
    return path.read_text(encoding="utf-8") == dump_json_schema()
