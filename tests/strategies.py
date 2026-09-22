"""Hypothesis strategies around the valid-by-construction generator, plus mutators.

The mutators break exactly one rule each on a deep copy of the scene, so that
property tests can assert a precise issue code.
"""

from __future__ import annotations

import random

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from rail_vision_bench.graph.generate import generate_scene, generate_topology
from rail_vision_bench.graph.rules import SLIP_KINDS, TERMINAL_KINDS
from rail_vision_bench.schema.models import (
    Node,
    NodeKind,
    Port,
    PortRef,
    SceneAnnotation,
    Topology,
    Track,
    TrackKind,
    TrackState,
)


def topologies(*, max_inner: int = 6) -> SearchStrategy[Topology]:
    """Topologies from the generator over seeds and inner-node counts."""
    return st.builds(
        generate_topology, seed=st.integers(0, 2**31 - 1), n_inner=st.integers(0, max_inner)
    )


def scenes(*, max_inner: int = 6) -> SearchStrategy[SceneAnnotation]:
    """Strict-valid scenes from the generator over seeds and inner-node counts."""
    return st.builds(
        generate_scene, seed=st.integers(0, 2**31 - 1), n_inner=st.integers(0, max_inner)
    )


def with_extra_edge(scene: SceneAnnotation, rng: random.Random) -> tuple[SceneAnnotation, str]:
    """Plug one more edge into an already used port of an inner node.

    Falls back to a boundary node when the scene has no inner node; every port of
    every generated node is used, so the target always gains one endpoint.

    Returns:
        The mutated deep copy and the id of the node whose degree is now wrong.
    """
    mutated = scene.model_copy(deep=True)
    inner = [node for node in mutated.topology.nodes if node.kind not in TERMINAL_KINDS]
    target = rng.choice(inner or mutated.topology.nodes)
    used_ports = [
        ref.port
        for edge in mutated.topology.edges
        for ref in (edge.a, edge.b)
        if ref.node == target.id
    ]
    mutated.topology.nodes.append(Node(id="x_extra", kind=NodeKind.BUFFER_STOP))
    mutated.topology.edges.append(
        Track(
            id="e_extra",
            kind=TrackKind.SIDING,
            a=PortRef(node="x_extra", port=Port.A),
            b=PortRef(node=target.id, port=rng.choice(used_ports)),
        )
    )
    mutated.state.tracks["e_extra"] = TrackState()
    return mutated, target.id


def with_dropped_edge(
    scene: SceneAnnotation, rng: random.Random
) -> tuple[SceneAnnotation, str, str]:
    """Remove one edge together with everything that references it.

    The edge's track state, the signals and derailers attached to it, and the
    routes using it (with their state) are removed too, so that only the two
    degree violations remain.

    Returns:
        The mutated deep copy and the ids of the edge's two end nodes.
    """
    mutated = scene.model_copy(deep=True)
    edge = rng.choice(mutated.topology.edges)
    mutated.topology.edges = [e for e in mutated.topology.edges if e.id != edge.id]
    mutated.state.tracks.pop(edge.id, None)
    dropped_signals = {s.id for s in mutated.signals if s.at.edge == edge.id}
    mutated.signals = [s for s in mutated.signals if s.id not in dropped_signals]
    for signal_id in dropped_signals:
        mutated.state.signals.pop(signal_id, None)
    dropped_derailers = {d.id for d in mutated.derailers if d.at.edge == edge.id}
    mutated.derailers = [d for d in mutated.derailers if d.id not in dropped_derailers]
    for derailer_id in dropped_derailers:
        mutated.state.derailers.pop(derailer_id, None)
    dropped_routes = {
        r.id
        for r in mutated.routes
        if edge.id in r.path or r.start in dropped_signals or r.end in dropped_signals
    }
    mutated.routes = [r for r in mutated.routes if r.id not in dropped_routes]
    for route_id in dropped_routes:
        mutated.state.routes.pop(route_id, None)
    return mutated, edge.a.node, edge.b.node


def with_bad_active_path(scene: SceneAnnotation) -> tuple[SceneAnnotation, str] | None:
    """Give the first dkw/ekw node the impossible active path A-B.

    Returns:
        The mutated deep copy and the node id, or None when the scene has no slip node.
    """
    slip = next((node for node in scene.topology.nodes if node.kind in SLIP_KINDS), None)
    if slip is None:
        return None
    mutated = scene.model_copy(deep=True)
    mutated.state.switches[slip.id].active_paths = [(Port.A, Port.B)]
    return mutated, slip.id
