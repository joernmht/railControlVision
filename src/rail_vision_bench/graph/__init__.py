"""Track graph: rules tables, networkx conversion, validator and generator."""

from __future__ import annotations

from rail_vision_bench.graph.build import endpoint_ports, to_networkx
from rail_vision_bench.graph.generate import DEFAULT_KINDS, generate_scene, generate_topology
from rail_vision_bench.graph.rules import (
    ALLOWED_PATHS_BY_KIND,
    DEGREE_BY_KIND,
    PATH_FAMILIES_BY_KIND,
    PORTS_BY_KIND,
    SLIP_KINDS,
    SWITCHING_KINDS,
    TERMINAL_KINDS,
)
from rail_vision_bench.graph.validator import (
    validate_document,
    validate_routes,
    validate_scene,
    validate_state,
    validate_topology,
)

__all__ = [
    "ALLOWED_PATHS_BY_KIND",
    "DEFAULT_KINDS",
    "DEGREE_BY_KIND",
    "PATH_FAMILIES_BY_KIND",
    "PORTS_BY_KIND",
    "SLIP_KINDS",
    "SWITCHING_KINDS",
    "TERMINAL_KINDS",
    "endpoint_ports",
    "generate_scene",
    "generate_topology",
    "to_networkx",
    "validate_document",
    "validate_routes",
    "validate_scene",
    "validate_state",
    "validate_topology",
]
