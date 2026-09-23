"""State shared by the nodes of the agentic loop.

The state is a ``total=False`` TypedDict: a caller seeds ``scene_id``,
``image``, ``model`` and the round budget (optionally ``source``), every node
adds the keys it owns, and the critic decides whether another round runs. The
prompt of each node documents which keys it reads and writes; ``usage`` and
``errors`` are bookkeeping every node appends to.
"""

from __future__ import annotations

from typing import Any, Final, Literal, TypedDict

from rail_vision_bench.providers.base import ImageInput, Usage

NodeName = Literal["planner", "reader", "interpreter", "geometer", "builder", "critic"]

NODE_NAMES: Final[tuple[NodeName, ...]] = (
    "planner",
    "reader",
    "interpreter",
    "geometer",
    "builder",
    "critic",
)
"""The six nodes in pipeline order; the critic may loop back to the planner."""


class AgentState(TypedDict, total=False):
    """Everything the loop knows about one scene between node calls."""

    scene_id: str
    image: ImageInput
    model: str
    source: dict[str, Any]
    plan: list[str]
    regions: list[dict[str, Any]]
    crops: list[dict[str, Any]]
    readings: list[dict[str, Any]]
    interpretation: dict[str, Any]
    geometry: dict[str, Any]
    candidate: dict[str, Any] | None
    validation: dict[str, Any] | None
    critique: str | None
    round: int
    max_rounds: int
    done: bool
    usage: Usage
    errors: list[str]
