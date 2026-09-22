"""The agent tools: the real validate tool and the tool names."""

from __future__ import annotations

from typing import Any

from rail_vision_bench.tools.tools import TOOL_NAMES, ToolResult, validate_tool
from tests.conftest import FIXTURES_DIR, load_json


def test_tool_names():
    assert TOOL_NAMES == ("crop", "read", "validate", "render")
    assert len(set(TOOL_NAMES)) == len(TOOL_NAMES)


def test_validate_tool_accepts_minimal(minimal_doc: dict[str, Any]):
    result = validate_tool(minimal_doc)
    assert isinstance(result, ToolResult)
    assert result.ok
    assert result.errors == []
    assert result.payload["ok"] is True


def test_validate_tool_reports_bad_dkw_path():
    result = validate_tool(load_json(FIXTURES_DIR / "invalid" / "dkw_bad_path.json"))
    assert not result.ok
    codes = {issue["code"] for issue in result.payload["issues"]}
    assert "STATE_PATH_NOT_ALLOWED" in codes
    assert result.errors
    assert all(isinstance(message, str) and message for message in result.errors)


def test_validate_tool_never_raises_on_garbage():
    result = validate_tool({"schema_version": "v0"})
    assert not result.ok
    assert {issue["code"] for issue in result.payload["issues"]} == {"SCHEMA_INVALID"}
