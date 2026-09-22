"""Expose the four tools over the Model Context Protocol (stub)."""

from __future__ import annotations

from typing import Literal


def serve_mcp(
    *, host: str = "127.0.0.1", port: int = 8765, transport: Literal["stdio", "sse"] = "stdio"
) -> None:
    """Serve crop, read, validate and render as MCP tools.

    The implementation will build a ``FastMCP`` server from the ``mcp``
    package (the one ``langchain-mcp-adapters`` depends on), register the
    callables of :mod:`rail_vision_bench.tools.tools` with their docstrings as
    tool descriptions, and run it over stdio (for local agent hosts) or SSE
    on ``host:port`` (for n8n and other remote clients). It blocks until the
    server stops.

    Args:
        host: Bind host for the SSE transport.
        port: Bind port for the SSE transport.
        transport: ``"stdio"`` or ``"sse"``.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.tools.mcp_server.serve_mcp is not implemented in the skeleton: "
        "serve the four tools over MCP via FastMCP"
    )
