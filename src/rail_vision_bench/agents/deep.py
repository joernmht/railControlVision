"""A ``deepagents`` variant of the agentic approach: one planning agent with tools (stub)."""

from __future__ import annotations

from typing import Any

from rail_vision_bench.providers.base import VisionProvider


def build_deep_agent(provider: VisionProvider) -> Any:
    """Create a deep agent that owns the crop, read, validate and render tools.

    The implementation will call ``deepagents.create_deep_agent`` with the four
    tools from :mod:`rail_vision_bench.tools.tools` wrapped as LangChain tools
    and a model adapter over ``provider``; the agent plans its own reading
    strategy instead of following the fixed six-node loop.

    Args:
        provider: The model backend.

    Returns:
        The compiled deep agent.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.deep.build_deep_agent is not implemented in the skeleton: "
        "create a deepagents agent over the four tools"
    )
