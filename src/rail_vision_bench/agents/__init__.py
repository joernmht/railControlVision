"""Agents: the single-shot baseline, the six-node LangGraph loop and the versioned prompts."""

from __future__ import annotations

from rail_vision_bench.agents.graph import AgenticResult, build_agent_graph, run_agentic
from rail_vision_bench.agents.prompts import PROMPT_NAMES, load_prompt
from rail_vision_bench.agents.single_shot import (
    ParseError,
    SingleShotResult,
    run_single_shot,
    single_shot_attempt,
)
from rail_vision_bench.agents.state import NODE_NAMES, AgentState, NodeName

__all__ = [
    "NODE_NAMES",
    "PROMPT_NAMES",
    "AgentState",
    "AgenticResult",
    "NodeName",
    "ParseError",
    "SingleShotResult",
    "build_agent_graph",
    "load_prompt",
    "run_agentic",
    "run_single_shot",
    "single_shot_attempt",
]
