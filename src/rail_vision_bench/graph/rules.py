"""The single definition of a valid track graph: ports, degrees and path sets per node kind.

The validator, the topology generator and the hypothesis strategies all import
these tables, so there is exactly one place where "valid" is defined.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

from rail_vision_bench.schema.models import NodeKind, Port, SwitchPosition

_AC: Final = frozenset({Port.A, Port.C})
_BD: Final = frozenset({Port.B, Port.D})
_AD: Final = frozenset({Port.A, Port.D})
_BC: Final = frozenset({Port.B, Port.C})

PORTS_BY_KIND: Final[dict[NodeKind, frozenset[Port]]] = {
    NodeKind.BUFFER_STOP: frozenset({Port.A}),
    NodeKind.BOUNDARY: frozenset({Port.A}),
    NodeKind.JOINT: frozenset({Port.A, Port.B}),
    NodeKind.SWITCH: frozenset({Port.TOE, Port.STRAIGHT, Port.DIVERGING}),
    NodeKind.CROSSING: frozenset({Port.A, Port.B, Port.C, Port.D}),
    NodeKind.EKW: frozenset({Port.A, Port.B, Port.C, Port.D}),
    NodeKind.DKW: frozenset({Port.A, Port.B, Port.C, Port.D}),
}
# Every port of a node must carry exactly one edge endpoint.
DEGREE_BY_KIND: Final[dict[NodeKind, int]] = {
    kind: len(ports) for kind, ports in PORTS_BY_KIND.items()
}
# Port pairs a vehicle can traverse through a node of the given kind.
ALLOWED_PATHS_BY_KIND: Final[dict[NodeKind, frozenset[frozenset[Port]]]] = {
    NodeKind.BUFFER_STOP: frozenset(),
    NodeKind.BOUNDARY: frozenset(),
    NodeKind.JOINT: frozenset({frozenset({Port.A, Port.B})}),
    NodeKind.SWITCH: frozenset(
        {frozenset({Port.TOE, Port.STRAIGHT}), frozenset({Port.TOE, Port.DIVERGING})}
    ),
    NodeKind.CROSSING: frozenset({_AC, _BD}),
    NodeKind.EKW: frozenset({_AC, _BD, _AD}),
    NodeKind.DKW: frozenset({_AC, _BD, _AD, _BC}),
}
# Sets of paths that can be connected simultaneously; an active set must lie inside one family.
PATH_FAMILIES_BY_KIND: Final[dict[NodeKind, tuple[frozenset[frozenset[Port]], ...]]] = {
    NodeKind.CROSSING: (frozenset({_AC, _BD}),),
    NodeKind.EKW: (frozenset({_AC, _BD}), frozenset({_AD})),
    NodeKind.DKW: (frozenset({_AC, _BD}), frozenset({_AD, _BC})),
}
SWITCHING_KINDS: Final[frozenset[NodeKind]] = frozenset(
    {NodeKind.SWITCH, NodeKind.EKW, NodeKind.DKW}
)
SLIP_KINDS: Final[frozenset[NodeKind]] = frozenset({NodeKind.EKW, NodeKind.DKW})
TERMINAL_KINDS: Final[frozenset[NodeKind]] = frozenset({NodeKind.BUFFER_STOP, NodeKind.BOUNDARY})


def as_path(ports: Iterable[Port]) -> frozenset[Port]:
    """Normalise a port pair to the unordered form used by the tables.

    Args:
        ports: The ports of the path (normally two).

    Returns:
        The ports as a frozenset.
    """
    return frozenset(ports)


def is_path_allowed(kind: NodeKind, path: frozenset[Port]) -> bool:
    """Report whether a node of ``kind`` can be traversed between the ports of ``path``.

    Args:
        kind: The node kind.
        path: An unordered port pair.

    Returns:
        True when the path is in ``ALLOWED_PATHS_BY_KIND[kind]``.
    """
    return path in ALLOWED_PATHS_BY_KIND[kind]


def is_path_set_allowed(kind: NodeKind, paths: Sequence[frozenset[Port]]) -> bool:
    """Report whether ``paths`` can be connected simultaneously at a node of ``kind``.

    Every path must be allowed, the paths must be pairwise port-disjoint, and
    for kinds with path families the whole set must lie inside one family. An
    empty set is always allowed; a plain switch allows no path set at all
    (its state is a position, not a path list).

    Args:
        kind: The node kind.
        paths: The unordered port pairs that are connected at once.

    Returns:
        True when the set is physically possible.
    """
    if not paths:
        return True
    if kind == NodeKind.SWITCH or not all(is_path_allowed(kind, path) for path in paths):
        return False
    used: set[Port] = set()
    for path in paths:
        if used & path:
            return False
        used |= path
    families = PATH_FAMILIES_BY_KIND.get(kind)
    if families is None:
        return True
    return any(set(paths) <= family for family in families)


def switch_position_for(path: frozenset[Port]) -> SwitchPosition | None:
    """Map a traversal path through a plain switch to the position it requires.

    Args:
        path: An unordered port pair.

    Returns:
        ``STRAIGHT`` for toe-straight, ``DIVERGING`` for toe-diverging, else None.
    """
    if path == frozenset({Port.TOE, Port.STRAIGHT}):
        return SwitchPosition.STRAIGHT
    if path == frozenset({Port.TOE, Port.DIVERGING}):
        return SwitchPosition.DIVERGING
    return None


def expected_setting(
    kind: NodeKind, path: frozenset[Port]
) -> SwitchPosition | frozenset[Port] | None:
    """Return the switch setting a route must declare to traverse ``path`` at a node.

    Args:
        kind: The node kind.
        path: The unordered port pair the route uses at the node.

    Returns:
        A position for a switch, the path itself for a dkw/ekw, None otherwise.
    """
    if kind == NodeKind.SWITCH:
        return switch_position_for(path)
    if kind in SLIP_KINDS:
        return path
    return None
