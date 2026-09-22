"""The six node functions of the agentic loop (stubs) and the critic's router (real).

Every node has the same shape: it takes the current :class:`AgentState`, uses
the keyword-only ``provider`` for model calls, and returns the keys it updates.
The prompt in ``prompts/agentic/<node>.md`` is the node's specification.
"""

from __future__ import annotations

from typing import Literal

from rail_vision_bench.agents.state import AgentState
from rail_vision_bench.providers.base import VisionProvider


def planner(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Plan the round: which regions to read and what the previous critique asks for.

    Reads ``scene_id``, ``image``, ``critique``, ``round`` and ``max_rounds``;
    writes ``plan`` (ordered work items) and increments ``round``.

    Args:
        state: The loop state.
        provider: The model used to inspect the overview image.

    Returns:
        The updated keys.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.nodes.planner is not implemented in the skeleton: "
        "turn the image and the last critique into an ordered plan of regions to read"
    )


def reader(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Transcribe labels, legends and indications from crops of the planned regions.

    Reads ``image`` and ``plan``; writes ``crops`` and ``readings``.

    Args:
        state: The loop state.
        provider: The model used to read each crop.

    Returns:
        The updated keys.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.nodes.reader is not implemented in the skeleton: "
        "crop the planned regions and transcribe their labels and indications"
    )


def interpreter(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Map transcribed symbols and literals to element kinds and states.

    Reads ``readings`` and ``crops``; writes ``interpretation``.

    Args:
        state: The loop state.
        provider: The model used to classify symbols.

    Returns:
        The updated keys.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.nodes.interpreter is not implemented in the skeleton: "
        "classify the readings into node kinds, signal kinds, positions and aspects"
    )


def geometer(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Connect the interpreted elements into edges with node ports and attachments.

    Reads ``image``, ``interpretation`` and ``readings``; writes ``geometry``.

    Args:
        state: The loop state.
        provider: The model used to trace track lines.

    Returns:
        The updated keys.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.nodes.geometer is not implemented in the skeleton: "
        "trace the track lines into edges between node ports with pixel geometry"
    )


def builder(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Assemble one schema v0 document from the interpretation and the geometry.

    Reads ``scene_id``, ``image``, ``interpretation`` and ``geometry``; writes
    ``candidate``.

    Args:
        state: The loop state.
        provider: The model used to emit the JSON document.

    Returns:
        The updated keys.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.nodes.builder is not implemented in the skeleton: "
        "assemble the SceneAnnotation JSON candidate"
    )


def critic(state: AgentState, *, provider: VisionProvider) -> AgentState:
    """Validate the candidate, compare its re-rendering with the image and judge.

    Reads ``candidate``, ``image``, ``round`` and ``max_rounds``; writes
    ``validation`` (the validate tool's report), ``critique`` and ``done``.

    Args:
        state: The loop state.
        provider: The model used to compare the re-rendering with the image.

    Returns:
        The updated keys.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.agents.nodes.critic is not implemented in the skeleton: "
        "validate the candidate, compare the re-rendering with the image and decide"
    )


def critic_should_continue(state: AgentState) -> Literal["planner", "__end__"]:
    """Route after the critic: stop when it is satisfied or the round budget is spent.

    Args:
        state: The loop state after the critic ran.

    Returns:
        ``"__end__"`` when ``done`` is set or ``round`` reached ``max_rounds``,
        ``"planner"`` for another round.
    """
    if state.get("done") or state.get("round", 0) >= state.get("max_rounds", 1):
        return "__end__"
    return "planner"
