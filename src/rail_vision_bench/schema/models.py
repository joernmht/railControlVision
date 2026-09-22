"""Pydantic models of schema v0: one document type for ground truth and predictions.

The models are purely structural (shape, enums, ranges, id pattern, no extra
keys). Every semantic rule (uniqueness, references, ports, degrees, path sets,
route walks) lives in :mod:`rail_vision_bench.graph.validator`, so that any
document accepted here can still be reported on with a precise issue code.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ElementId = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$",
        description="Identifier of a node, edge, signal, derailer, route or scene.",
    ),
]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class NodeKind(StrEnum):
    """Kinds of topology nodes (track ends, joints, switches and crossings)."""

    BUFFER_STOP = "buffer_stop"
    BOUNDARY = "boundary"
    JOINT = "joint"
    SWITCH = "switch"
    CROSSING = "crossing"
    EKW = "ekw"
    DKW = "dkw"


class Port(StrEnum):
    """Ports of a node; which ones exist depends on the node kind.

    buffer_stop and boundary expose A; joint exposes A and B; switch exposes toe,
    straight and diverging; crossing, ekw and dkw expose A, B, C and D where A and
    B lie on one side and C and D on the other (straight paths A-C and B-D, slip
    paths A-D for ekw and dkw, B-C for dkw only).
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    TOE = "toe"
    STRAIGHT = "straight"
    DIVERGING = "diverging"


PortPath = tuple[Port, Port]


class TrackKind(StrEnum):
    """Kinds of track sections (edges)."""

    MAIN = "main"
    SIDING = "siding"
    PLATFORM = "platform"
    YARD = "yard"
    UNKNOWN = "unknown"


class SignalKind(StrEnum):
    """Kinds of signals."""

    MAIN = "main"
    DISTANT = "distant"
    SHUNT = "shunt"
    COMBINED = "combined"
    REPEATER = "repeater"
    BLOCK_MARKER = "block_marker"
    UNKNOWN = "unknown"


class SignalAspect(StrEnum):
    """Signal aspects; the literal indication (Hp0, Vr2, Ks1, ...) goes into ``raw``."""

    STOP = "stop"
    PROCEED = "proceed"
    PROCEED_REDUCED = "proceed_reduced"
    EXPECT_STOP = "expect_stop"
    EXPECT_PROCEED = "expect_proceed"
    EXPECT_PROCEED_REDUCED = "expect_proceed_reduced"
    SHUNT_PROCEED = "shunt_proceed"
    DARK = "dark"
    UNKNOWN = "unknown"


class SwitchPosition(StrEnum):
    """Geometric position of a plain switch; the panel's +/- indication goes into ``raw``."""

    STRAIGHT = "straight"
    DIVERGING = "diverging"
    MOVING = "moving"
    UNKNOWN = "unknown"


class Occupancy(StrEnum):
    """Track occupancy."""

    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class DerailerPosition(StrEnum):
    """Derailer position."""

    APPLIED = "applied"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class RouteStatus(StrEnum):
    """Route (Fahrstrasse) status."""

    NOT_SET = "not_set"
    SET = "set"
    LOCKED = "locked"
    RELEASING = "releasing"
    UNKNOWN = "unknown"


class Direction(StrEnum):
    """Travel direction along an edge, relative to its ``a`` -> ``b`` orientation."""

    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"


class SourceKind(StrEnum):
    """Where the source image comes from."""

    SYNTHETIC = "synthetic"
    PANEL_PHOTO = "panel_photo"
    ESTW_SCREEN = "estw_screen"
    CTC_SCREEN = "ctc_screen"
    STREAM_FRAME = "stream_frame"


class ProvenanceKind(StrEnum):
    """Whether a document is ground truth or a model prediction."""

    GROUND_TRUTH = "ground_truth"
    PREDICTION = "prediction"


class _Model(BaseModel):
    """Base for every schema model: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class Geometry(_Model):
    """Pixel geometry of an element in the coordinate system of ``source.image``."""

    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="[x0, y0, x1, y1] in pixels of source.image."
    )
    polyline: list[tuple[float, float]] | None = Field(
        default=None, description="[[x, y], ...] in pixels of source.image."
    )
    point: tuple[float, float] | None = Field(
        default=None, description="[x, y] in pixels of source.image."
    )


class PortRef(_Model):
    """One end of an edge: a node and the port it plugs into."""

    node: ElementId
    port: Port


class Node(_Model):
    """A topology node."""

    id: ElementId
    kind: NodeKind
    label: str | None = None
    half_labels: list[str] = Field(
        default_factory=list,
        description="Labels of the two switch halves of a dkw/ekw, e.g. ['12a', '12b'].",
    )
    geometry: Geometry | None = None
    confidence: Confidence | None = None


class Track(_Model):
    """A track section (edge) between two node ports."""

    id: ElementId
    kind: TrackKind = TrackKind.UNKNOWN
    label: str | None = None
    a: PortRef
    b: PortRef
    geometry: Geometry | None = None
    confidence: Confidence | None = None


class EdgeAttachment(_Model):
    """Position of a trackside element on an edge."""

    edge: ElementId
    offset: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Position along the edge from a (0.0) to b (1.0)."
    )
    direction: Direction = Field(description="The travel direction the element governs.")


class Signal(_Model):
    """A signal attached to an edge."""

    id: ElementId
    kind: SignalKind
    label: str | None = None
    system: str | None = Field(default=None, description="Signalling system, e.g. H/V, Ks, Hl.")
    at: EdgeAttachment
    geometry: Geometry | None = None
    confidence: Confidence | None = None


class Derailer(_Model):
    """A derailer (Gleissperre) attached to an edge."""

    id: ElementId
    label: str | None = None
    at: EdgeAttachment
    geometry: Geometry | None = None
    confidence: Confidence | None = None


class Route(_Model):
    """A route (Fahrstrasse) from a start signal over an ordered list of edges."""

    id: ElementId
    label: str | None = None
    start: ElementId = Field(description="Id of the start signal.")
    end: ElementId = Field(description="Id of the end signal or end node.")
    path: list[ElementId] = Field(min_length=1, description="Edge ids ordered from start to end.")
    switch_positions: dict[ElementId, SwitchPosition | PortPath] = Field(
        default_factory=dict,
        description="Required setting per switching node: a position for a switch, "
        "a [Port, Port] path for a dkw/ekw.",
    )
    confidence: Confidence | None = None


class Topology(_Model):
    """Nodes and edges of the track graph."""

    nodes: list[Node] = Field(default_factory=list)
    edges: list[Track] = Field(default_factory=list)


class SwitchState(_Model):
    """State of a switching node; ``position`` for a switch, ``active_paths`` for a dkw/ekw."""

    position: SwitchPosition | None = None
    active_paths: list[PortPath] | None = Field(
        default=None, description="Zero, one or two simultaneously connected port pairs."
    )
    raw: str | None = Field(
        default=None, description="Transcribed indication, e.g. '+', '-', '12a:+ 12b:-'."
    )
    confidence: Confidence | None = None


class SignalState(_Model):
    """State of a signal."""

    aspect: SignalAspect = SignalAspect.UNKNOWN
    raw: str | None = Field(default=None, description="Transcribed aspect, e.g. 'Hp0', 'Vr2'.")
    confidence: Confidence | None = None


class TrackState(_Model):
    """State of a track section: occupancy and route illumination are independent."""

    occupancy: Occupancy = Occupancy.UNKNOWN
    route_set: bool | None = Field(
        default=None, description="Fahrstrassenausleuchtung, independent of occupancy."
    )
    confidence: Confidence | None = None


class DerailerState(_Model):
    """State of a derailer."""

    position: DerailerPosition = DerailerPosition.UNKNOWN
    confidence: Confidence | None = None


class RouteState(_Model):
    """State of a route."""

    status: RouteStatus = RouteStatus.UNKNOWN
    confidence: Confidence | None = None


class State(_Model):
    """Observed state keyed by element id."""

    switches: dict[ElementId, SwitchState] = Field(default_factory=dict)
    signals: dict[ElementId, SignalState] = Field(default_factory=dict)
    tracks: dict[ElementId, TrackState] = Field(default_factory=dict)
    derailers: dict[ElementId, DerailerState] = Field(default_factory=dict)
    routes: dict[ElementId, RouteState] = Field(default_factory=dict)
    observed_at_ms: int | None = None


class Source(_Model):
    """The source image the document describes."""

    kind: SourceKind
    image: str = Field(description="Image path relative to the dataset root.")
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]
    frame_index: int | None = None
    timestamp_ms: int | None = None


class Provenance(_Model):
    """Who or what produced the document."""

    kind: ProvenanceKind
    run_id: str | None = None
    model_id: str | None = None
    annotator: str | None = None
    tool: str | None = None
    created_at: datetime | None = None


class SceneAnnotation(_Model):
    """A complete annotation of one scene: topology, trackside elements, routes and state."""

    schema_version: Literal["v0"]
    scene_id: ElementId
    source: Source
    provenance: Provenance
    topology: Topology
    signals: list[Signal] = Field(default_factory=list)
    derailers: list[Derailer] = Field(default_factory=list)
    routes: list[Route] = Field(default_factory=list)
    state: State
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Free-form notes (station name, difficulty tags, ...)."
    )

    def node_map(self) -> dict[str, Node]:
        """Return nodes keyed by id (last one wins on duplicates).

        Returns:
            Mapping from node id to node.
        """
        return {node.id: node for node in self.topology.nodes}

    def edge_map(self) -> dict[str, Track]:
        """Return edges keyed by id (last one wins on duplicates).

        Returns:
            Mapping from edge id to edge.
        """
        return {edge.id: edge for edge in self.topology.edges}

    def signal_map(self) -> dict[str, Signal]:
        """Return signals keyed by id (last one wins on duplicates).

        Returns:
            Mapping from signal id to signal.
        """
        return {signal.id: signal for signal in self.signals}

    def derailer_map(self) -> dict[str, Derailer]:
        """Return derailers keyed by id (last one wins on duplicates).

        Returns:
            Mapping from derailer id to derailer.
        """
        return {derailer.id: derailer for derailer in self.derailers}

    def route_map(self) -> dict[str, Route]:
        """Return routes keyed by id (last one wins on duplicates).

        Returns:
            Mapping from route id to route.
        """
        return {route.id: route for route in self.routes}
