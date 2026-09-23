"""LangGraph wiring of the agentic loop and the critic's router."""

from __future__ import annotations

import inspect
import io
from itertools import pairwise
from typing import ClassVar

import pytest
from PIL import Image

from rail_vision_bench.agents import nodes
from rail_vision_bench.agents.graph import build_agent_graph
from rail_vision_bench.agents.nodes import critic_should_continue
from rail_vision_bench.agents.state import NODE_NAMES, AgentState
from rail_vision_bench.providers.base import ImageInput, VisionRequest, VisionResponse


class FakeProvider:
    """A provider that echoes the prompt; enough to bind the nodes."""

    name: ClassVar[str] = "fake"

    async def complete(self, request: VisionRequest) -> VisionResponse:
        return VisionResponse(text=request.prompt)


def test_graph_compiles_with_every_node():
    graph = build_agent_graph(FakeProvider(), model="fake-1")
    drawn = graph.get_graph()
    assert set(NODE_NAMES) <= set(drawn.nodes)


def test_pipeline_edges_and_critic_loop():
    drawn = build_agent_graph(FakeProvider(), model="fake-1").get_graph()
    edges = {(edge.source, edge.target): edge.conditional for edge in drawn.edges}
    for source, target in pairwise(NODE_NAMES):
        assert edges[(source, target)] is False
    assert edges[("critic", "planner")] is True
    assert any(source == "critic" and edge for (source, _), edge in edges.items())


def test_nodes_are_coroutines():
    for name in NODE_NAMES:
        assert inspect.iscoroutinefunction(getattr(nodes, name)), name


async def test_ainvoke_needs_an_image():
    graph = build_agent_graph(FakeProvider(), model="fake-1", max_rounds=1)
    with pytest.raises(ValueError, match="no 'image'"):
        await graph.ainvoke({"scene_id": "s", "round": 0, "max_rounds": 1})


async def test_ainvoke_runs_a_round_with_an_echoing_provider():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32)).save(buffer, format="PNG")
    graph = build_agent_graph(FakeProvider(), model="fake-1", max_rounds=1)
    final = await graph.ainvoke({"scene_id": "s", "image": ImageInput(data=buffer.getvalue())})
    # the echo is no JSON: every node falls back, the loop still ends after its one round
    assert final["round"] == 1
    assert final["model"] == "fake-1"
    assert final["candidate"]["scene_id"] == "s"
    assert final["candidate"]["topology"] == {"nodes": [], "edges": []}
    assert final["errors"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"done": True, "round": 0, "max_rounds": 3}, "__end__"),
        ({"done": False, "round": 3, "max_rounds": 3}, "__end__"),
        ({"round": 4, "max_rounds": 3}, "__end__"),
        ({"done": False, "round": 1, "max_rounds": 3}, "planner"),
        ({}, "planner"),
        ({"round": 1}, "__end__"),
    ],
)
def test_critic_should_continue(state: AgentState, expected: str):
    assert critic_should_continue(state) == expected
