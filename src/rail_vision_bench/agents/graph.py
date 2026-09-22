"""Wiring of the agentic loop as a LangGraph state graph.

The pipeline is planner -> reader -> interpreter -> geometer -> builder ->
critic, and the critic's router either ends the run or sends it back to the
planner with the critique kept in the state. The wiring is real; the nodes
raise ``NotImplementedError`` until they are implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rail_vision_bench.agents.nodes import (
    builder,
    critic,
    critic_should_continue,
    geometer,
    interpreter,
    planner,
    reader,
)
from rail_vision_bench.agents.state import AgentState, NodeName
from rail_vision_bench.providers.base import VisionProvider

NodeFn = Callable[..., AgentState]


class BoundNode(Protocol):
    """A node with the provider already bound: what ``StateGraph.add_node`` accepts.

    LangGraph's node protocol names its parameter ``state``, which a plain
    ``Callable[[AgentState], AgentState]`` (positional-only) does not satisfy.
    """

    def __call__(self, state: AgentState) -> AgentState:
        """Run the node on ``state`` and return the updated keys."""
        ...


_NODE_FNS: tuple[tuple[NodeName, NodeFn], ...] = (
    ("planner", planner),
    ("reader", reader),
    ("interpreter", interpreter),
    ("geometer", geometer),
    ("builder", builder),
    ("critic", critic),
)


def build_agent_graph(
    provider: VisionProvider, *, max_rounds: int = 3
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the six-node loop with every node bound to ``provider``.

    Args:
        provider: The model backend the nodes call.
        max_rounds: Round budget used when the initial state does not set ``max_rounds``.

    Returns:
        The compiled graph; invoke it with a state holding at least ``scene_id`` and ``image``.
    """

    def _bind(fn: NodeFn) -> BoundNode:
        def node(state: AgentState) -> AgentState:
            # The budget is a graph-level setting: a caller who omits it from the
            # initial state gets the compiled default rather than the router's fallback.
            merged: AgentState = {**state}
            merged.setdefault("max_rounds", max_rounds)
            return fn(merged, provider=provider)

        return node

    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    for name, fn in _NODE_FNS:
        graph.add_node(name, _bind(fn))
    graph.add_edge(START, "planner")
    for (source, _), (target, _) in pairwise(_NODE_FNS):
        graph.add_edge(source, target)
    graph.add_conditional_edges(
        "critic", critic_should_continue, {"planner": "planner", "__end__": END}
    )
    return graph.compile()
