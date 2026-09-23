"""The MCP server: tool registration and calls without running a transport."""

from __future__ import annotations

import base64
import io
import json
from typing import Any, ClassVar

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image

from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.tools import mcp_server
from rail_vision_bench.tools.mcp_server import build_mcp_server
from rail_vision_bench.tools.tools import TOOL_NAMES


class ReadingProvider:
    name: ClassVar[str] = "reading"

    def __init__(self) -> None:
        self.requests: list[VisionRequest] = []

    async def complete(self, request: VisionRequest) -> VisionResponse:
        self.requests.append(request)
        return VisionResponse(
            text=json.dumps({"text": "W12", "confidence": 0.75}),
            usage=Usage(input_tokens=7, output_tokens=2),
        )


def png_base64(width: int = 40, height: int = 20) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def structured(result: Any) -> dict[str, Any]:
    assert isinstance(result, tuple)
    payload = result[1]
    assert isinstance(payload, dict)
    return payload


async def test_registers_the_four_tools():
    server = build_mcp_server()
    assert isinstance(server, FastMCP)
    tools = await server.list_tools()
    assert [tool.name for tool in tools] == list(TOOL_NAMES)
    assert all(tool.description for tool in tools)
    crop = next(tool for tool in tools if tool.name == "crop")
    assert set(crop.inputSchema["required"]) == {"image_base64", "bbox"}


async def test_validate_tool(minimal_doc: dict[str, Any]):
    server = build_mcp_server()
    ok = structured(await server.call_tool("validate", {"document": minimal_doc}))
    assert ok["ok"] is True
    bad = structured(await server.call_tool("validate", {"document": {"schema_version": "v0"}}))
    assert bad["ok"] is False
    assert {issue["code"] for issue in bad["payload"]["issues"]} == {"SCHEMA_INVALID"}


async def test_crop_tool_returns_an_image():
    server = build_mcp_server()
    result = await server.call_tool("crop", {"image_base64": png_base64(), "bbox": [0, 0, 10, 10]})
    assert isinstance(result, list)
    (content,) = result
    assert isinstance(content, ImageContent)
    assert content.mimeType == "image/png"
    with Image.open(io.BytesIO(base64.b64decode(content.data))) as image:
        assert image.size == (20, 20)


async def test_render_tool_returns_an_image(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mcp_server, "render_tool", lambda document: base64.b64decode(png_base64()))
    server = build_mcp_server()
    result = await server.call_tool("render", {"document": {}})
    assert isinstance(result, list)
    assert isinstance(result[0], ImageContent)


async def test_read_without_provider_is_an_error_result():
    server = build_mcp_server()
    result = structured(
        await server.call_tool("read", {"image_base64": png_base64(), "question": "q"})
    )
    assert result["ok"] is False
    assert "no provider" in result["errors"][0]


async def test_read_without_model_is_an_error_result():
    server = build_mcp_server(provider=ReadingProvider())
    result = structured(
        await server.call_tool("read", {"image_base64": png_base64(), "question": "q"})
    )
    assert result["ok"] is False
    assert "no model" in result["errors"][0]


async def test_read_with_provider():
    provider = ReadingProvider()
    server = build_mcp_server(provider=provider, model="m-1")
    result = structured(
        await server.call_tool("read", {"image_base64": png_base64(), "question": "label?"})
    )
    assert result["ok"] is True
    assert result["payload"]["text"] == "W12"
    assert result["payload"]["confidence"] == 0.75
    assert provider.requests[0].model == "m-1"
    bad = structured(await server.call_tool("read", {"image_base64": "%%%", "question": "q"}))
    assert bad["ok"] is False
    assert "base64" in bad["errors"][0]


async def test_read_with_unknown_provider_name():
    server = build_mcp_server(provider_name="nope", model="m")
    result = structured(
        await server.call_tool("read", {"image_base64": png_base64(), "question": "q"})
    )
    assert result["ok"] is False
    assert "unknown provider" in result["errors"][0]


def test_serve_mcp_runs_the_built_server(monkeypatch: pytest.MonkeyPatch):
    ran: list[str] = []
    monkeypatch.setattr(FastMCP, "run", lambda self, transport="stdio": ran.append(transport))
    mcp_server.serve_mcp(transport="sse", port=9999, model="m")
    assert ran == ["sse"]
