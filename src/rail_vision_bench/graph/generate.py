"""Deterministic, valid-by-construction topology and scene generator.

The generator uses the same rules tables as the validator, so every result
passes ``validate_document(strict=True)``. Hypothesis strategies wrap it and
the synthetic-data pipeline will reuse it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import pairwise
from typing import Final

from rail_vision_bench.graph.rules import SLIP_KINDS, SWITCHING_KINDS, TERMINAL_KINDS
from rail_vision_bench.schema.models import (
    Direction,
    EdgeAttachment,
    Node,
    NodeKind,
    Occupancy,
    Port,
    PortRef,
    Provenance,
    ProvenanceKind,
    Route,
    RouteState,
    RouteStatus,
    SceneAnnotation,
    Signal,
    SignalAspect,
    SignalKind,
    SignalState,
    Source,
    SourceKind,
    State,
    SwitchPosition,
    SwitchState,
    Topology,
    Track,
    TrackKind,
    TrackState,
)

DEFAULT_KINDS: Final[tuple[NodeKind, ...]] = (
    NodeKind.JOINT,
    NodeKind.SWITCH,
    NodeKind.DKW,
    NodeKind.EKW,
    NodeKind.CROSSING,
)
# The ports through which the main line enters (west) and leaves (east) a node.
_WEST_EAST: Final[dict[NodeKind, tuple[Port, Port]]] = {
    NodeKind.BOUNDARY: (Port.A, Port.A),
    NodeKind.BUFFER_STOP: (Port.A, Port.A),
    NodeKind.JOINT: (Port.A, Port.B),
    NodeKind.SWITCH: (Port.TOE, Port.STRAIGHT),
    NodeKind.CROSSING: (Port.A, Port.C),
    NodeKind.EKW: (Port.A, Port.C),
    NodeKind.DKW: (Port.A, Port.C),
}
_FOUR_PORT_KINDS: Final[frozenset[NodeKind]] = frozenset(
    {NodeKind.CROSSING, NodeKind.EKW, NodeKind.DKW}
)
_STRAIGHT_PATH: Final[tuple[Port, Port]] = (Port.A, Port.C)


def _main_edges(topology: Topology) -> list[Track]:
    return [edge for edge in topology.edges if edge.kind == TrackKind.MAIN]


def generate_topology(
    seed: int, *, n_inner: int = 3, kinds: Sequence[NodeKind] = DEFAULT_KINDS
) -> Topology:
    """Generate a main line ``n_w`` - ``x0`` .. ``x{n-1}`` - ``n_e`` with sidings.

    Inner node kinds are drawn with ``random.Random(seed)``. Every switch gets a
    siding from its diverging port to a buffer stop; every crossing, ekw and dkw
    gets one siding from port B and one from port D, so every port carries
    exactly one edge.

    Args:
        seed: Seed of the pseudo-random choice of inner node kinds.
        n_inner: Number of inner nodes (0 gives ``n_w`` - ``m0`` - ``n_e``).
        kinds: The non-terminal kinds to draw inner nodes from.

    Returns:
        The topology.

    Raises:
        ValueError: If ``kinds`` is empty, contains a terminal kind, or ``n_inner`` < 0.
    """
    if n_inner < 0:
        msg = f"n_inner must be >= 0, got {n_inner}"
        raise ValueError(msg)
    if not kinds or any(kind in TERMINAL_KINDS for kind in kinds):
        msg = f"kinds must be non-empty and non-terminal, got {list(kinds)}"
        raise ValueError(msg)
    rng = random.Random(seed)
    nodes = [Node(id="n_w", kind=NodeKind.BOUNDARY)]
    edges: list[Track] = []
    for i in range(n_inner):
        kind = rng.choice(list(kinds))
        node = Node(
            id=f"x{i}",
            kind=kind,
            label=f"W{i}" if kind in SWITCHING_KINDS else None,
            half_labels=[f"W{i}a", f"W{i}b"] if kind in SLIP_KINDS else [],
        )
        nodes.append(node)
        if kind == NodeKind.SWITCH:
            nodes.append(Node(id=f"b{i}", kind=NodeKind.BUFFER_STOP))
            edges.append(_siding(f"s{i}", node.id, Port.DIVERGING, f"b{i}"))
        elif kind in _FOUR_PORT_KINDS:
            nodes.append(Node(id=f"b{i}a", kind=NodeKind.BUFFER_STOP))
            nodes.append(Node(id=f"b{i}b", kind=NodeKind.BUFFER_STOP))
            edges.append(_siding(f"s{i}a", node.id, Port.B, f"b{i}a"))
            edges.append(_siding(f"s{i}b", node.id, Port.D, f"b{i}b"))
    nodes.append(Node(id="n_e", kind=NodeKind.BOUNDARY))

    # The main line runs through the boundaries and the inner nodes, never the buffer stops.
    line = [node for node in nodes if not node.id.startswith("b")]
    main = [
        Track(
            id=f"m{i}",
            kind=TrackKind.MAIN,
            a=PortRef(node=west.id, port=_WEST_EAST[west.kind][1]),
            b=PortRef(node=east.id, port=_WEST_EAST[east.kind][0]),
        )
        for i, (west, east) in enumerate(pairwise(line))
    ]
    return Topology(nodes=nodes, edges=main + edges)


def _siding(edge_id: str, node_id: str, port: Port, buffer_id: str) -> Track:
    return Track(
        id=edge_id,
        kind=TrackKind.SIDING,
        a=PortRef(node=node_id, port=port),
        b=PortRef(node=buffer_id, port=Port.A),
    )


def generate_scene(
    seed: int, *, n_inner: int = 3, kinds: Sequence[NodeKind] = DEFAULT_KINDS
) -> SceneAnnotation:
    """Generate a strict-valid scene: topology, two main signals, one route and full state.

    ``sig_w`` protects the main line eastwards from ``m0``; ``sig_e`` protects it
    westwards from the last main edge; route ``rt_w`` runs from ``sig_w`` over
    every main edge to ``n_e`` with all switches set straight (A-C on slips).

    Args:
        seed: Seed of the pseudo-random choice of inner node kinds.
        n_inner: Number of inner nodes.
        kinds: The non-terminal kinds to draw inner nodes from.

    Returns:
        A ground-truth scene without geometry or confidences.
    """
    topology = generate_topology(seed, n_inner=n_inner, kinds=kinds)
    scene_id = f"synthetic-{seed}-{n_inner}"
    main = _main_edges(topology)
    signals = [
        Signal(
            id="sig_w",
            kind=SignalKind.MAIN,
            system="H/V",
            at=EdgeAttachment(edge=main[0].id, offset=0.1, direction=Direction.A_TO_B),
        ),
        Signal(
            id="sig_e",
            kind=SignalKind.MAIN,
            system="H/V",
            at=EdgeAttachment(edge=main[-1].id, offset=0.9, direction=Direction.B_TO_A),
        ),
    ]
    switch_positions: dict[str, SwitchPosition | tuple[Port, Port]] = {}
    switches: dict[str, SwitchState] = {}
    for node in topology.nodes:
        if node.kind == NodeKind.SWITCH:
            switch_positions[node.id] = SwitchPosition.STRAIGHT
            switches[node.id] = SwitchState(position=SwitchPosition.STRAIGHT, raw="+")
        elif node.kind in SLIP_KINDS:
            switch_positions[node.id] = _STRAIGHT_PATH
            switches[node.id] = SwitchState(active_paths=[_STRAIGHT_PATH], raw="a:+ b:+")
    route = Route(
        id="rt_w",
        label="sig_w -> n_e",
        start="sig_w",
        end="n_e",
        path=[edge.id for edge in main],
        switch_positions=switch_positions,
    )
    state = State(
        switches=switches,
        signals={
            "sig_w": SignalState(aspect=SignalAspect.PROCEED, raw="Hp1"),
            "sig_e": SignalState(aspect=SignalAspect.STOP, raw="Hp0"),
        },
        tracks={
            edge.id: TrackState(occupancy=Occupancy.FREE, route_set=edge.kind == TrackKind.MAIN)
            for edge in topology.edges
        },
        routes={"rt_w": RouteState(status=RouteStatus.SET)},
    )
    return SceneAnnotation(
        schema_version="v0",
        scene_id=scene_id,
        source=Source(
            kind=SourceKind.SYNTHETIC, image=f"synthetic/{scene_id}.png", width=800, height=400
        ),
        provenance=Provenance(
            kind=ProvenanceKind.GROUND_TRUTH, tool="rail_vision_bench.graph.generate"
        ),
        topology=topology,
        signals=signals,
        routes=[route],
        state=state,
    )
