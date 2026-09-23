"""Expose the four tools over the Model Context Protocol.

:func:`build_mcp_server` registers crop, read, validate and render on a
``FastMCP`` server from the ``mcp`` package (the one ``langchain-mcp-adapters``
depends on); :func:`serve_mcp` runs it over stdio (local agent hosts) or SSE
(n8n and other remote clients). Images travel as base64 strings in and as MCP
image content out; documents are plain JSON objects.

The read tool needs a model: pass a provider (or a registry name) and the SDK
model id. Without them the server still starts and ``read`` answers with an
error result, so crop, validate and render stay usable on their own.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, Final, Literal

from mcp.server.fastmcp import FastMCP, Image

from rail_vision_bench.providers.base import VisionProvider
from rail_vision_bench.providers.registry import get_provider
from rail_vision_bench.settings import Settings
from rail_vision_bench.tools.tools import (
    ToolResult,
    crop_tool,
    read_tool,
    render_tool,
    validate_tool,
)

SERVER_NAME: Final = "rail-vision-bench"
SERVER_INSTRUCTIONS: Final = (
    "Tools for transcribing railway control imagery into rail-vision-bench schema v0: crop and "
    "magnify a region, read a label or indication with a vision model, validate a document, "
    "render a document as a panel image. Images are base64-encoded PNG/JPEG/WebP bytes."
)


def _decode(image_base64: str) -> bytes:
    """Decode a base64 image argument (a ``data:`` URL prefix is tolerated).

    Raises:
        ValueError: If the argument is not valid base64.
    """
    payload = image_base64.split(",", 1)[1] if image_base64.startswith("data:") else image_base64
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = f"image_base64 is not valid base64: {exc}"
        raise ValueError(msg) from exc


def build_mcp_server(
    *,
    provider: VisionProvider | None = None,
    provider_name: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> FastMCP:
    """Build the MCP server with crop, read, validate and render registered.

    Args:
        provider: The backend of the read tool; takes precedence over ``provider_name``.
        provider_name: Registry name of the read tool's backend, instantiated
            on the first read call with ``settings``.
        model: SDK model id of the read tool's requests.
        settings: Credentials for ``provider_name``; ``get_settings()`` when omitted.
        host: Bind host for the SSE transport.
        port: Bind port for the SSE transport.

    Returns:
        The configured server (not running).
    """
    server = FastMCP(SERVER_NAME, instructions=SERVER_INSTRUCTIONS, host=host, port=port)
    resolved: dict[str, VisionProvider] = {} if provider is None else {"provider": provider}

    def read_backend() -> VisionProvider | str:
        """Return the read tool's provider, or why there is none."""
        if "provider" in resolved:
            return resolved["provider"]
        if provider_name is None:
            return "the read tool has no provider: start the server with provider_name"
        try:
            resolved["provider"] = get_provider(provider_name, settings)
        except KeyError as exc:
            return str(exc)
        return resolved["provider"]

    @server.tool(name="crop")
    def crop(image_base64: str, bbox: list[float], zoom: float = 2.0) -> Image:
        """Cut bbox = [x0, y0, x1, y1] (pixels) out of an image and upscale it by zoom.

        Returns the crop as a PNG image.
        """
        if len(bbox) != 4:
            msg = "bbox needs four numbers [x0, y0, x1, y1]"
            raise ValueError(msg)
        region = (bbox[0], bbox[1], bbox[2], bbox[3])
        return Image(data=crop_tool(_decode(image_base64), region, zoom=zoom), format="png")

    @server.tool(name="read")
    async def read(image_base64: str, question: str) -> dict[str, Any]:
        """Ask a focused question about an image crop (a label, a lamp, an indication).

        Returns a tool result whose payload holds the transcription (text) and confidence.
        """
        backend = read_backend()
        if isinstance(backend, str):
            return ToolResult(ok=False, errors=[backend]).model_dump(mode="json")
        if model is None:
            message = "the read tool has no model: start the server with model"
            return ToolResult(ok=False, errors=[message]).model_dump(mode="json")
        try:
            image = _decode(image_base64)
        except ValueError as exc:
            return ToolResult(ok=False, errors=[str(exc)]).model_dump(mode="json")
        result = await read_tool(image, question, provider=backend, model=model)
        return result.model_dump(mode="json")

    @server.tool(name="validate")
    def validate(document: dict[str, Any]) -> dict[str, Any]:
        """Validate a scene document against schema v0 and the semantic rules.

        Returns a tool result: ok, the validation report as payload and the error messages.
        """
        return validate_tool(document).model_dump(mode="json")

    @server.tool(name="render")
    def render(document: dict[str, Any]) -> Image:
        """Re-render a scene document as a Stelltisch-style panel image (PNG)."""
        return Image(data=render_tool(document), format="png")

    return server


def serve_mcp(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    transport: Literal["stdio", "sse"] = "stdio",
    provider_name: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Serve crop, read, validate and render as MCP tools; blocks until the server stops.

    Args:
        host: Bind host for the SSE transport.
        port: Bind port for the SSE transport.
        transport: ``"stdio"`` for local agent hosts or ``"sse"`` for remote clients.
        provider_name: Registry name of the read tool's backend.
        model: SDK model id of the read tool's requests.
        settings: Credentials for ``provider_name``; ``get_settings()`` when omitted.
    """
    server = build_mcp_server(
        provider_name=provider_name, model=model, settings=settings, host=host, port=port
    )
    server.run(transport=transport)
