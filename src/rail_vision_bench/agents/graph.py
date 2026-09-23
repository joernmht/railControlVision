"""Wiring of the agentic loop as a LangGraph state graph.

The pipeline is planner -> reader -> interpreter -> geometer -> builder ->
critic, and the critic's router either ends the run or sends it back to the
planner with the critique kept in the state. The nodes are coroutines, so the
compiled graph is driven with ``ainvoke``; :func:`run_agentic` does that for
one image and reports the outcome in the shape the runner records.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from itertools import pairwise
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.agents.documents import default_source, parse_scene, scene_id_for
from rail_vision_bench.agents.nodes import (
    builder,
    critic,
    critic_should_continue,
    geometer,
    interpreter,
    planner,
    reader,
)
from rail_vision_bench.agents.state import NODE_NAMES, AgentState, NodeName
from rail_vision_bench.providers.base import ImageInput, Usage, VisionProvider
from rail_vision_bench.schema.models import SceneAnnotation, Source

NodeFn = Callable[..., Awaitable[AgentState]]


class BoundNode(Protocol):
    """A node with the provider already bound: what ``StateGraph.add_node`` accepts.

    LangGraph's node protocol names its parameter ``state``, which a plain
    ``Callable[[AgentState], AgentState]`` (positional-only) does not satisfy.
    """

    def __call__(self, state: AgentState) -> Awaitable[AgentState]:
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
    provider: VisionProvider, *, model: str, max_rounds: int = 3
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the six-node loop with every node bound to ``provider``.

    Args:
        provider: The model backend the nodes call.
        model: SDK model id used when the initial state does not set ``model``.
        max_rounds: Round budget used when the initial state does not set ``max_rounds``.

    Returns:
        The compiled graph; drive it with ``ainvoke`` and a state holding at
        least ``scene_id`` and ``image``.
    """

    def _bind(fn: NodeFn) -> BoundNode:
        async def node(state: AgentState) -> AgentState:
            # Model and budget are graph-level settings: a caller who omits them from
            # the initial state gets the compiled defaults rather than the router's fallback.
            merged: AgentState = {**state}
            merged.setdefault("max_rounds", max_rounds)
            merged.setdefault("model", model)
            update = await fn(merged, provider=provider)
            # Persist the defaults so the router and later nodes see the same values.
            return {"max_rounds": merged["max_rounds"], "model": merged["model"], **update}

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


class AgenticResult(BaseModel):
    """Everything one run of the agentic loop produced, parsed or not."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str
    annotation: SceneAnnotation | None = Field(
        default=None, description="The final candidate parsed, None when it fails schema v0."
    )
    document: dict[str, Any] | None = Field(
        default=None, description="The final candidate as raw JSON, None when none was built."
    )
    raw_text: str = Field(description="The final candidate serialised as JSON ('' when none).")
    parse_error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = Field(description="Wall-clock time of the whole loop.")
    rounds: int = Field(description="Rounds the loop ran.")
    done: bool = Field(default=False, description="Whether the critic accepted the candidate.")
    critique: str | None = Field(default=None, description="The critic's last critique.")
    validation: dict[str, Any] | None = Field(
        default=None, description="The validate tool's report on the final candidate."
    )
    errors: list[str] = Field(
        default_factory=list, description="Recoverable problems the nodes recorded."
    )


def recursion_limit(max_rounds: int) -> int:
    """LangGraph step budget that lets ``max_rounds`` full rounds finish.

    Args:
        max_rounds: The round budget.

    Returns:
        Six steps per round plus headroom.
    """
    return len(NODE_NAMES) * max(1, max_rounds) + 10


async def run_agentic(
    provider: VisionProvider,
    image: ImageInput,
    *,
    model: str,
    scene_id: str | None = None,
    max_rounds: int = 3,
    source: Source | None = None,
    on_round: Callable[[AgentState], Awaitable[None]] | None = None,
) -> AgenticResult:
    """Run the six-node loop over one image and collect the outcome.

    Args:
        provider: The model backend.
        image: The panel or screen image.
        model: The SDK model id every node sends.
        scene_id: The scene id; derived from the image bytes when omitted.
        max_rounds: The round budget.
        source: The source block (manifest data); derived from the image when omitted.
        on_round: Awaited with the state after every critic step (candidate,
            validation, critique, round), e.g. to stream partial results.

    Returns:
        The final candidate (parsed when it matches schema v0), its JSON text,
        the usage summed over every model call, the latency and the rounds run.

    Raises:
        ValueError: If ``source`` is omitted and the image cannot be decoded.
    """
    scene = scene_id if scene_id is not None else scene_id_for(image)
    src = source if source is not None else default_source(image, scene_id=scene)
    graph = build_agent_graph(provider, model=model, max_rounds=max_rounds)
    initial: AgentState = {
        "scene_id": scene,
        "image": image,
        "model": model,
        "source": src.model_dump(mode="json"),
        "round": 0,
        "max_rounds": max_rounds,
        "usage": Usage(),
        "errors": [],
        "critique": None,
    }
    final: AgentState = {**initial}
    started = time.perf_counter()
    # Every key is last-write-wins (no reducers), so merging the per-node updates
    # reproduces the graph's state while letting a caller watch each round.
    async for chunk in graph.astream(
        initial,
        config={"recursion_limit": recursion_limit(max_rounds)},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            if update:
                final.update(update)
            if node_name == "critic" and on_round is not None:
                await on_round(final)
    latency_ms = (time.perf_counter() - started) * 1000.0
    candidate = final.get("candidate")
    annotation: SceneAnnotation | None = None
    if candidate is None:
        parse_error: str | None = "the agentic loop assembled no candidate document"
    else:
        annotation, parse_error = parse_scene(candidate)
    return AgenticResult(
        scene_id=scene,
        annotation=annotation,
        document=candidate,
        raw_text=json.dumps(candidate, ensure_ascii=False) if candidate is not None else "",
        parse_error=parse_error,
        usage=final.get("usage") or Usage(),
        latency_ms=latency_ms,
        rounds=final.get("round", 0),
        done=bool(final.get("done")),
        critique=final.get("critique"),
        validation=final.get("validation"),
        errors=list(final.get("errors", [])),
    )
