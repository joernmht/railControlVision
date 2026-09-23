"""Rendering of scene annotations as panel-style images.

Two steps keep the ground truth of synthetic data exact by construction:

1. :func:`layout_scene` places the topology on a tile grid (the pitch of a
   German mosaic panel, Stelltisch) and writes the resulting pixel geometry
   (node points and tiles, track polylines, signal and derailer glyph boxes)
   into the document;
2. :func:`render_svg` draws a document *from its geometry* with ``svgwrite``,
   using the same glyph helpers the layout used for the signal and derailer
   boxes, and :func:`svg_to_png` / :func:`render_png` rasterise it with
   ``cairosvg`` at the source size.

State is drawn as a panel shows it: occupancy (red) and route illumination
(yellow on a Stelltisch, green on an ESTW screen) as independent lamp strips,
the set leg of a switch lit and the other leg interrupted, the active paths of
slip switches lit, signal lamps in their aspect colours and derailers applied
across or removed beside the track.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from rail_vision_bench.schema.models import (
    Derailer,
    DerailerPosition,
    Direction,
    EdgeAttachment,
    Geometry,
    Node,
    NodeKind,
    Occupancy,
    Port,
    SceneAnnotation,
    Signal,
    SignalAspect,
    SignalKind,
    SwitchPosition,
    Track,
)

STYLES: Final[tuple[str, ...]] = ("stelltisch", "estw")
"""Supported visual styles."""

DEFAULT_TILE_PX: Final[float] = 32.0
"""Tile pitch used by the synthetic dataset generator."""

Point = tuple[float, float]
GridPoint = tuple[int, int]

# Port directions of a node whose main line runs west -> east, as (dx, dy) grid steps with y
# pointing down. A node "facing" west mirrors the x component. Terminal kinds point their only
# port east (the node is the west end of its track).
_PORT_VECTORS: Final[dict[NodeKind, dict[Port, GridPoint]]] = {
    NodeKind.BOUNDARY: {Port.A: (1, 0)},
    NodeKind.BUFFER_STOP: {Port.A: (1, 0)},
    NodeKind.JOINT: {Port.A: (-1, 0), Port.B: (1, 0)},
    NodeKind.SWITCH: {Port.TOE: (-1, 0), Port.STRAIGHT: (1, 0), Port.DIVERGING: (1, 1)},
    NodeKind.CROSSING: {Port.A: (-1, 0), Port.C: (1, 0), Port.B: (-1, 1), Port.D: (1, -1)},
    NodeKind.EKW: {Port.A: (-1, 0), Port.C: (1, 0), Port.B: (-1, 1), Port.D: (1, -1)},
    NodeKind.DKW: {Port.A: (-1, 0), Port.C: (1, 0), Port.B: (-1, 1), Port.D: (1, -1)},
}
_MAIN_GAP: Final[int] = 3
"""Straight grid steps between the port stubs of a horizontal edge (edge length 5 tiles)."""
_COMPONENT_ROW_GAP: Final[int] = 3


@dataclass(frozen=True)
class _Palette:
    background: str
    tile: str | None
    seam: str | None
    track: str
    lamp_off: str | None
    route: str
    occupied: str
    indicator: str
    label: str
    signal_body: str
    lamp_dark: str


_PALETTES: Final[dict[str, _Palette]] = {
    "stelltisch": _Palette(
        background="#6d716b",
        tile="#bfc3b8",
        seam="#80847c",
        track="#1c1c1c",
        lamp_off="#5a5a5a",
        route="#ffd23f",
        occupied="#e3202a",
        indicator="#ffffff",
        label="#111111",
        signal_body="#1c1c1c",
        lamp_dark="#4a4a4a",
    ),
    "estw": _Palette(
        background="#000000",
        tile=None,
        seam=None,
        track="#8c8c8c",
        lamp_off=None,
        route="#22c83c",
        occupied="#ff2a2a",
        indicator="#ffff66",
        label="#f0f0f0",
        signal_body="#d0d0d0",
        lamp_dark="#303030",
    ),
}

_RED: Final = "#e3202a"
_GREEN: Final = "#1fc23a"
_YELLOW: Final = "#ffc21a"
_WHITE: Final = "#ffffff"
_GREY: Final = "#8a8a8a"
_ASPECT_LAMPS: Final[dict[SignalAspect, tuple[str | None, ...]]] = {
    SignalAspect.STOP: (_RED,),
    SignalAspect.PROCEED: (_GREEN,),
    SignalAspect.PROCEED_REDUCED: (_GREEN, _YELLOW),
    SignalAspect.EXPECT_STOP: (_YELLOW,),
    SignalAspect.EXPECT_PROCEED: (_GREEN,),
    SignalAspect.EXPECT_PROCEED_REDUCED: (_GREEN, _YELLOW),
    SignalAspect.SHUNT_PROCEED: (_WHITE, _WHITE),
    SignalAspect.DARK: (None,),
    SignalAspect.UNKNOWN: (_GREY,),
}
_FONT: Final = "DejaVu Sans, Arial, Helvetica, sans-serif"


# --------------------------------------------------------------------------- geometry helpers


def _add(p: Point, q: Point) -> Point:
    return (p[0] + q[0], p[1] + q[1])


def _scale(p: Point, k: float) -> Point:
    return (p[0] * k, p[1] * k)


def _unit(p: Point) -> Point:
    length = math.hypot(p[0], p[1])
    return (1.0, 0.0) if length == 0 else (p[0] / length, p[1] / length)


def _dedupe(points: Iterable[Point]) -> list[Point]:
    """Drop consecutive duplicates and interior points on a straight run."""
    out: list[Point] = []
    for point in points:
        if out and math.isclose(out[-1][0], point[0]) and math.isclose(out[-1][1], point[1]):
            continue
        out.append(point)
    simplified: list[Point] = out[:1]
    for i in range(1, len(out) - 1):
        a, b, c = simplified[-1], out[i], out[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(cross) > 1e-9 or dot < 0:
            simplified.append(b)
    if len(out) > 1:
        simplified.append(out[-1])
    return simplified


def _length(points: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in pairwise(points))


def _point_at(points: Sequence[Point], distance: float) -> tuple[Point, Point]:
    """Return the point at arc length ``distance`` and the unit tangent there."""
    if len(points) == 1:
        return points[0], (1.0, 0.0)
    remaining = max(0.0, distance)
    for a, b in pairwise(points):
        seg = math.dist(a, b)
        if seg == 0:
            continue
        if remaining <= seg:
            k = remaining / seg
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k), _unit(
                (b[0] - a[0], b[1] - a[1])
            )
        remaining -= seg
    a, b = points[-2], points[-1]
    return points[-1], _unit((b[0] - a[0], b[1] - a[1]))


def _trim(points: Sequence[Point], start: float, end: float) -> list[Point]:
    """Cut ``start`` pixels of arc length off the front and ``end`` off the back."""
    total = _length(points)
    if total <= start + end:
        return []
    out: list[Point] = [_point_at(points, start)[0]]
    walked = 0.0
    for a, b in pairwise(points):
        walked += math.dist(a, b)
        if start < walked < total - end:
            out.append(b)
    out.append(_point_at(points, total - end)[0])
    return _dedupe(out)


def _bbox(points: Iterable[Point], pad: float = 0.0) -> tuple[float, float, float, float]:
    xs, ys = zip(*points, strict=True)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _clamp_point(p: Point, width: int, height: int) -> Point:
    return (min(max(p[0], 0.0), float(width)), min(max(p[1], 0.0), float(height)))


def _clamp_bbox(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x0, y0 = _clamp_point((box[0], box[1]), width, height)
    x1, y1 = _clamp_point((box[2], box[3]), width, height)
    return (x0, y0, x1, y1)


def _octilinear(p: GridPoint, q: GridPoint) -> list[GridPoint]:
    """Connect two grid points with horizontal/vertical and 45 degree segments."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    if dx == 0 or dy == 0 or abs(dx) == abs(dy):
        return [p, q]
    corner = (q[0] - sx * abs(dy), p[1]) if abs(dx) > abs(dy) else (q[0], p[1] + sy * abs(dx))
    return [p, corner, q]


# --------------------------------------------------------------------------- glyphs


@dataclass(frozen=True)
class _Glyph:
    """Placement of a trackside element: its anchor on the track and its symbol box."""

    anchor: Point
    tangent: Point
    normal: Point
    center: Point
    bbox: tuple[float, float, float, float]


def _attachment_frame(polyline: Sequence[Point], at: EdgeAttachment) -> tuple[Point, Point, Point]:
    """Track point, travel tangent and left-hand normal of an attachment."""
    point, tangent = _point_at(polyline, at.offset * _length(polyline))
    if at.direction == Direction.B_TO_A:
        tangent = (-tangent[0], -tangent[1])
    # y points down, so the left of travel direction (tx, ty) is (ty, -tx).
    return point, tangent, (tangent[1], -tangent[0])


def _signal_parts(center: Point, tangent: Point, tile: float) -> tuple[Point, Point, float]:
    """Mast foot, second-lamp position and lamp radius of a signal glyph around its lamp."""
    radius = 0.15 * tile
    foot = _add(center, _scale(tangent, -0.42 * tile))
    second = _add(center, _scale(tangent, -0.3 * tile))
    return foot, second, radius


def _signal_glyph(polyline: Sequence[Point], signal: Signal, tile: float) -> _Glyph:
    """Signal symbol beside the track, on the left of the governed direction."""
    anchor, tangent, normal = _attachment_frame(polyline, signal.at)
    base = _add(anchor, _scale(normal, 0.45 * tile))
    center = _add(base, _scale(tangent, 0.2 * tile))
    foot, _, radius = _signal_parts(center, tangent, tile)
    ends = [_add(foot, _scale(normal, 0.12 * tile)), _add(foot, _scale(normal, -0.12 * tile))]
    box = _bbox([*ends, _add(center, (radius, radius)), _add(center, (-radius, -radius))])
    return _Glyph(anchor=anchor, tangent=tangent, normal=normal, center=center, bbox=box)


def _derailer_glyph(polyline: Sequence[Point], derailer: Derailer, tile: float) -> _Glyph:
    """Derailer symbol on the track at its attachment point."""
    anchor, tangent, normal = _attachment_frame(polyline, derailer.at)
    half = 0.3 * tile
    corners = [
        _add(anchor, _add(_scale(normal, sn * half), _scale(tangent, st * 0.15 * tile)))
        for sn in (-1, 1)
        for st in (-1, 1)
    ]
    return _Glyph(anchor=anchor, tangent=tangent, normal=normal, center=anchor, bbox=_bbox(corners))


def _glyph_at(glyph: _Glyph, geometry: Geometry | None) -> _Glyph:
    """Move a computed glyph so its center matches stored geometry, if there is any."""
    if geometry is None:
        return glyph
    if geometry.point is not None:
        target = geometry.point
    elif geometry.bbox is not None:
        x0, y0, x1, y1 = geometry.bbox
        target = ((x0 + x1) / 2, (y0 + y1) / 2)
    else:
        return glyph
    shift = (target[0] - glyph.center[0], target[1] - glyph.center[1])
    x0, y0, x1, y1 = glyph.bbox
    return _Glyph(
        anchor=glyph.anchor,
        tangent=glyph.tangent,
        normal=glyph.normal,
        center=target,
        bbox=(x0 + shift[0], y0 + shift[1], x1 + shift[0], y1 + shift[1]),
    )


# --------------------------------------------------------------------------- layout


def _vector(kind: NodeKind, port: Port, facing: int) -> GridPoint:
    dx, dy = _PORT_VECTORS.get(kind, {}).get(port, (1, 0))
    return (dx * facing, dy)


def _place_nodes(doc: SceneAnnotation) -> tuple[dict[str, GridPoint], dict[str, int]]:
    """Place every node on the tile grid by a breadth-first walk over the ports.

    Each component starts at a boundary (else a buffer stop, else any node)
    facing east; a neighbour is put where the port stubs of both ends line up,
    horizontal edges five tiles long and diagonal sidings two tiles. A node
    whose cell is taken moves further along the edge direction.
    """
    nodes = doc.node_map()
    incidence: dict[str, list[tuple[Port, str, Port]]] = {node_id: [] for node_id in nodes}
    for edge in doc.topology.edges:
        if edge.a.node in nodes and edge.b.node in nodes and edge.a.node != edge.b.node:
            incidence[edge.a.node].append((edge.a.port, edge.b.node, edge.b.port))
            incidence[edge.b.node].append((edge.b.port, edge.a.node, edge.a.port))
    rank = {NodeKind.BOUNDARY: 0, NodeKind.BUFFER_STOP: 1}
    seeds = sorted(nodes, key=lambda node_id: rank.get(nodes[node_id].kind, 2))
    pos: dict[str, GridPoint] = {}
    facing: dict[str, int] = {}
    taken: set[GridPoint] = set()
    for seed in seeds:
        if seed in pos:
            continue
        row = max((y for _, y in pos.values()), default=-_COMPONENT_ROW_GAP) + _COMPONENT_ROW_GAP
        col = min((x for x, _ in pos.values()), default=0)
        pos[seed], facing[seed] = (col, row), 1
        taken.add((col, row))
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            kind = nodes[current].kind
            links = sorted(incidence[current], key=lambda link: _vector(kind, link[0], 1)[1] != 0)
            for port, other, other_port in links:
                if other in pos:
                    continue
                d = _vector(kind, port, facing[current])
                other_kind = nodes[other].kind
                east = _vector(other_kind, other_port, 1)
                other_facing = 1 if east[0] == -d[0] else -1
                mv = _vector(other_kind, other_port, other_facing)
                q1 = (pos[current][0] + d[0], pos[current][1] + d[1])
                gap = _MAIN_GAP if d[1] == 0 and mv[1] == 0 else 0
                target = (q1[0] + d[0] * gap - mv[0], q1[1] - mv[1])
                while target in taken:
                    target = (target[0] + d[0], target[1])
                pos[other], facing[other] = target, other_facing
                taken.add(target)
                queue.append(other)
    return pos, facing


def _grid_polyline(
    edge: Track, nodes: dict[str, Node], pos: dict[str, GridPoint], facing: dict[str, int]
) -> list[GridPoint]:
    pa, pb = pos[edge.a.node], pos[edge.b.node]
    da = _vector(nodes[edge.a.node].kind, edge.a.port, facing[edge.a.node])
    db = _vector(nodes[edge.b.node].kind, edge.b.port, facing[edge.b.node])
    q1 = (pa[0] + da[0], pa[1] + da[1])
    q2 = (pb[0] + db[0], pb[1] + db[1])
    if q1 == pb or q2 == pa:
        return [pa, pb]
    return [pa, *_octilinear(q1, q2), pb]


def layout_scene(
    doc: SceneAnnotation, *, tile_px: float | None = None, margin_tiles: float = 1.0
) -> SceneAnnotation:
    """Lay the scene out on a Stelltisch tile grid and store the pixel geometry.

    Nodes sit at tile centres; tracks run horizontally or at 45 degrees
    between them. Every node gets ``point`` (its tile centre) and ``bbox`` (its
    tile), every edge ``polyline`` and ``bbox``, every signal and derailer the
    ``point`` and ``bbox`` of the glyph :func:`render_svg` draws for it. All
    coordinates are clamped to ``[0, width] x [0, height]``. The tile pitch
    and grid origin are recorded in ``meta["render"]`` so the renderer draws
    the tile seams where the layout put them. Existing geometry is replaced.

    Args:
        doc: The scene; its topology may be any graph (dangling references and
            self-loops are left without geometry).
        tile_px: Tile pitch in pixels. When given, ``source.width`` and
            ``source.height`` are set to fit the layout; when ``None`` the
            layout is scaled and centred into the existing source size.
        margin_tiles: Empty border around the outermost tile centres, in tiles.

    Returns:
        A copy of ``doc`` with geometry filled in (and possibly a new source size).

    Raises:
        ValueError: If ``tile_px`` or ``margin_tiles`` is not positive.
    """
    if tile_px is not None and tile_px <= 0:
        msg = f"tile_px must be positive, got {tile_px}"
        raise ValueError(msg)
    if margin_tiles <= 0:
        msg = f"margin_tiles must be positive, got {margin_tiles}"
        raise ValueError(msg)
    nodes = doc.node_map()
    pos, facing = _place_nodes(doc)
    grid_lines: dict[str, list[GridPoint]] = {
        edge.id: _grid_polyline(edge, nodes, pos, facing)
        for edge in doc.topology.edges
        if edge.a.node in pos and edge.b.node in pos and edge.a.node != edge.b.node
    }
    everything = list(pos.values()) + [p for line in grid_lines.values() for p in line]
    if not everything:
        everything = [(0, 0)]
    min_x = min(x for x, _ in everything)
    max_x = max(x for x, _ in everything)
    min_y = min(y for _, y in everything)
    max_y = max(y for _, y in everything)
    cols = max_x - min_x + 2 * margin_tiles
    rows = max_y - min_y + 2 * margin_tiles
    source = doc.source
    if tile_px is not None:
        tile = float(tile_px)
        width, height = max(1, math.ceil(cols * tile)), max(1, math.ceil(rows * tile))
        source = source.model_copy(update={"width": width, "height": height})
    else:
        width, height = source.width, source.height
        tile = min(width / cols, height / rows)
    ox = (width - cols * tile) / 2 + (margin_tiles - min_x) * tile
    oy = (height - rows * tile) / 2 + (margin_tiles - min_y) * tile

    def px(p: GridPoint) -> Point:
        return _clamp_point((ox + p[0] * tile, oy + p[1] * tile), width, height)

    half = tile / 2
    new_nodes = [
        node.model_copy(
            update={
                "geometry": Geometry(
                    point=px(pos[node.id]),
                    bbox=_clamp_bbox(_bbox([px(pos[node.id])], half), width, height),
                )
            }
            if node.id in pos
            else {}
        )
        for node in doc.topology.nodes
    ]
    polylines = {edge_id: [px(p) for p in line] for edge_id, line in grid_lines.items()}
    track_pad = 0.09 * tile
    new_edges = [
        edge.model_copy(
            update={
                "geometry": Geometry(
                    polyline=polylines[edge.id],
                    bbox=_clamp_bbox(_bbox(polylines[edge.id], track_pad), width, height),
                )
            }
            if edge.id in polylines
            else {}
        )
        for edge in doc.topology.edges
    ]

    def glyph_geometry(glyph: _Glyph) -> Geometry:
        return Geometry(
            point=_clamp_point(glyph.center, width, height),
            bbox=_clamp_bbox(glyph.bbox, width, height),
        )

    new_signals = [
        signal.model_copy(
            update={
                "geometry": glyph_geometry(_signal_glyph(polylines[signal.at.edge], signal, tile))
            }
            if signal.at.edge in polylines
            else {}
        )
        for signal in doc.signals
    ]
    new_derailers = [
        derailer.model_copy(
            update={
                "geometry": glyph_geometry(
                    _derailer_glyph(polylines[derailer.at.edge], derailer, tile)
                )
            }
            if derailer.at.edge in polylines
            else {}
        )
        for derailer in doc.derailers
    ]
    meta = dict(doc.meta)
    meta["render"] = {
        "tile_px": tile,
        "grid_origin": [round(ox - half, 4), round(oy - half, 4)],
    }
    return doc.model_copy(
        update={
            "source": source,
            "topology": doc.topology.model_copy(update={"nodes": new_nodes, "edges": new_edges}),
            "signals": new_signals,
            "derailers": new_derailers,
            "meta": meta,
        }
    )


def has_complete_geometry(doc: SceneAnnotation) -> bool:
    """Tell whether every node has a position and every edge a polyline.

    Args:
        doc: The scene.

    Returns:
        True when the document can be drawn from its own geometry.
    """
    nodes_ok = all(
        node.geometry is not None
        and (node.geometry.point is not None or node.geometry.bbox is not None)
        for node in doc.topology.nodes
    )
    edges_ok = all(
        edge.geometry is not None
        and edge.geometry.polyline is not None
        and len(edge.geometry.polyline) >= 2
        for edge in doc.topology.edges
    )
    return nodes_ok and edges_ok


# --------------------------------------------------------------------------- rendering


def _node_point(node: Node) -> Point | None:
    if node.geometry is None:
        return None
    if node.geometry.point is not None:
        return node.geometry.point
    if node.geometry.bbox is not None:
        x0, y0, x1, y1 = node.geometry.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    return None


def _render_params(doc: SceneAnnotation) -> tuple[float, Point]:
    """Tile pitch and grid origin from ``meta["render"]``, else a size-based default."""
    info = doc.meta.get("render")
    if isinstance(info, dict):
        tile = info.get("tile_px")
        origin = info.get("grid_origin")
        if isinstance(tile, int | float) and tile > 0:
            if isinstance(origin, list | tuple) and len(origin) == 2:
                return float(tile), (float(origin[0]), float(origin[1]))
            return float(tile), (0.0, 0.0)
    return max(10.0, min(doc.source.width, doc.source.height) / 20), (0.0, 0.0)


class _Painter:
    """Draws one document onto an ``svgwrite`` drawing."""

    def __init__(self, doc: SceneAnnotation, style: str, tile: float, origin: Point) -> None:
        import svgwrite  # type: ignore[import-untyped]

        self.doc = doc
        self.palette = _PALETTES[style]
        self.style = style
        self.tile = tile
        self.origin = origin
        self.width = doc.source.width
        self.height = doc.source.height
        self.dwg: Any = svgwrite.Drawing(
            size=(self.width, self.height), profile="full", debug=False
        )
        self.dwg.viewbox(0, 0, self.width, self.height)
        self.nodes = doc.node_map()
        self.polylines: dict[str, list[Point]] = {
            edge.id: [(float(x), float(y)) for x, y in edge.geometry.polyline]
            for edge in doc.topology.edges
            if edge.geometry is not None
            and edge.geometry.polyline is not None
            and len(edge.geometry.polyline) >= 2
        }
        # (edge id, end) of every edge endpoint at a node, keyed by node and port.
        self.ends: dict[str, dict[Port, tuple[str, str]]] = {}
        for edge in doc.topology.edges:
            if edge.id not in self.polylines:
                continue
            self.ends.setdefault(edge.a.node, {})[edge.a.port] = (edge.id, "a")
            self.ends.setdefault(edge.b.node, {})[edge.b.port] = (edge.id, "b")

    # -- primitives
    def _line(self, points: Sequence[Point], color: str, width: float, **extra: Any) -> None:
        if len(points) < 2:
            return
        self.dwg.add(
            self.dwg.polyline(
                points=[(round(x, 2), round(y, 2)) for x, y in points],
                fill="none",
                stroke=color,
                stroke_width=round(width, 3),
                stroke_linecap=extra.pop("linecap", "butt"),
                stroke_linejoin="round",
                **extra,
            )
        )

    def _text(self, text: str, at: Point, size: float, color: str | None = None) -> None:
        self.dwg.add(
            self.dwg.text(
                text,
                insert=(round(at[0], 2), round(at[1] + size * 0.35, 2)),
                font_size=round(size, 2),
                font_family=_FONT,
                font_weight="bold",
                fill=color or self.palette.label,
                text_anchor="middle",
            )
        )

    def _circle(self, center: Point, radius: float, fill: str, stroke: str | None) -> None:
        self.dwg.add(
            self.dwg.circle(
                center=(round(center[0], 2), round(center[1], 2)),
                r=round(radius, 2),
                fill=fill,
                stroke=stroke or "none",
                stroke_width=round(0.04 * self.tile, 3),
            )
        )

    def _polygon(self, points: Sequence[Point], fill: str, stroke: str | None = None) -> None:
        self.dwg.add(
            self.dwg.polygon(
                points=[(round(x, 2), round(y, 2)) for x, y in points],
                fill=fill,
                stroke=stroke or "none",
                stroke_width=round(0.04 * self.tile, 3),
            )
        )

    def _end_frame(self, node_id: str, port: Port, distance: float) -> tuple[Point, Point] | None:
        """Point at ``distance`` from a node along the edge on ``port``, and the direction."""
        end = self.ends.get(node_id, {}).get(port)
        if end is None:
            return None
        line = self.polylines[end[0]]
        if end[1] == "b":
            line = line[::-1]
        distance = min(distance, _length(line))
        return _point_at(line, distance)

    # -- layers
    def background(self) -> None:
        p = self.palette
        self.dwg.add(
            self.dwg.rect(insert=(0, 0), size=(self.width, self.height), fill=p.background)
        )
        if p.tile is None or p.seam is None:
            return
        t = self.tile
        inset = 0.03 * t
        ox = self.origin[0] % t - t
        oy = self.origin[1] % t - t
        x = ox
        while x < self.width:
            y = oy
            while y < self.height:
                self.dwg.add(
                    self.dwg.rect(
                        insert=(round(x + inset, 2), round(y + inset, 2)),
                        size=(round(t - 2 * inset, 2), round(t - 2 * inset, 2)),
                        fill=p.tile,
                    )
                )
                y += t
            x += t
        seam_width = max(1.0, 0.05 * t)
        x = ox
        while x <= self.width + t:
            self._line([(x, 0), (x, self.height)], p.seam, seam_width)
            x += t
        y = oy
        while y <= self.height + t:
            self._line([(0, y), (self.width, y)], p.seam, seam_width)
            y += t

    def _switch_trims(self) -> dict[tuple[str, str], float]:
        """Gap (in pixels) at edge ends that are the unset leg of a plain switch."""
        gap = 0.4 * self.tile
        trims: dict[tuple[str, str], float] = {}
        for node_id, node in self.nodes.items():
            if node.kind != NodeKind.SWITCH:
                continue
            state = self.doc.state.switches.get(node_id)
            position = state.position if state is not None else None
            unset: tuple[Port, ...] = ()
            if position == SwitchPosition.STRAIGHT:
                unset = (Port.DIVERGING,)
            elif position == SwitchPosition.DIVERGING:
                unset = (Port.STRAIGHT,)
            elif position == SwitchPosition.MOVING:
                unset = (Port.STRAIGHT, Port.DIVERGING)
            for port in unset:
                end = self.ends.get(node_id, {}).get(port)
                if end is not None:
                    trims[end] = gap
        return trims

    def tracks(self) -> None:
        p = self.palette
        t = self.tile
        trims = self._switch_trims()
        for edge in self.doc.topology.edges:
            line = self.polylines.get(edge.id)
            if line is None:
                continue
            drawn = _trim(line, trims.get((edge.id, "a"), 0.0), trims.get((edge.id, "b"), 0.0))
            if len(drawn) < 2:
                continue
            state = self.doc.state.tracks.get(edge.id)
            occupied = state is not None and state.occupancy == Occupancy.OCCUPIED
            routed = state is not None and state.route_set is True
            if self.style == "estw":
                self._line(drawn, p.route if routed else p.track, 0.14 * t)
                if occupied:
                    pattern = f"{0.3 * t:.2f},{0.12 * t:.2f}" if routed else None
                    extra = {"stroke_dasharray": pattern} if pattern else {}
                    self._line(drawn, p.occupied, 0.14 * t, **extra)
                continue
            self._line(drawn, p.track, 0.2 * t)
            dash, gap = 0.22 * t, 0.14 * t
            lamp = 0.08 * t
            if occupied and routed:
                # Both axes stay visible: red and yellow lamps alternate along the section.
                pattern = f"{dash:.2f},{dash + 2 * gap:.2f}"
                self._line(drawn, p.occupied, lamp, stroke_dasharray=pattern)
                self._line(
                    drawn,
                    p.route,
                    lamp,
                    stroke_dasharray=pattern,
                    stroke_dashoffset=f"{-(dash + gap):.2f}",
                )
            elif occupied or routed:
                color = p.occupied if occupied else p.route
                self._line(drawn, color, lamp, stroke_dasharray=f"{dash:.2f},{gap:.2f}")
            elif p.lamp_off is not None:
                self._line(drawn, p.lamp_off, lamp * 0.6, stroke_dasharray=f"{dash:.2f},{gap:.2f}")

    def nodes_layer(self) -> None:
        for node_id, node in self.nodes.items():
            center = _node_point(node)
            if center is None:
                continue
            if node.kind in (NodeKind.BUFFER_STOP, NodeKind.BOUNDARY):
                self._terminal(node, center)
            elif node.kind == NodeKind.JOINT:
                self._joint(node_id, center)
            elif node.kind == NodeKind.SWITCH:
                self._switch(node_id, center)
            else:
                self._four_port(node, center)
            self._node_labels(node, center)

    def _terminal(self, node: Node, center: Point) -> None:
        frame = self._end_frame(node.id, Port.A, 0.3 * self.tile)
        direction = frame[1] if frame is not None else (1.0, 0.0)
        normal = (direction[1], -direction[0])
        t = self.tile
        color = self.palette.track
        if node.kind == NodeKind.BUFFER_STOP:
            half = 0.32 * t
            self._line(
                [_add(center, _scale(normal, half)), _add(center, _scale(normal, -half))],
                color,
                0.16 * t,
            )
            return
        # A boundary is an open track end: an arrow pointing out of the panel.
        back = (-direction[0], -direction[1])
        tip = _add(center, _scale(back, 0.3 * t))
        wing = 0.22 * t
        self._polygon(
            [
                tip,
                _add(_add(center, _scale(normal, wing)), _scale(back, -0.05 * t)),
                _add(_add(center, _scale(normal, -wing)), _scale(back, -0.05 * t)),
            ],
            color,
        )

    def _joint(self, node_id: str, center: Point) -> None:
        frame = self._end_frame(node_id, Port.A, 0.3 * self.tile) or self._end_frame(
            node_id, Port.B, 0.3 * self.tile
        )
        direction = frame[1] if frame is not None else (1.0, 0.0)
        normal = (direction[1], -direction[0])
        t = self.tile
        gap_color = self.palette.tile or self.palette.background
        self._line(
            [_add(center, _scale(normal, 0.13 * t)), _add(center, _scale(normal, -0.13 * t))],
            gap_color,
            0.07 * t,
        )
        self._line(
            [_add(center, _scale(normal, 0.28 * t)), _add(center, _scale(normal, -0.28 * t))],
            self.palette.track,
            0.04 * t,
        )

    def _switch(self, node_id: str, center: Point) -> None:
        state = self.doc.state.switches.get(node_id)
        position = state.position if state is not None else None
        leg = {SwitchPosition.STRAIGHT: Port.STRAIGHT, SwitchPosition.DIVERGING: Port.DIVERGING}
        port = leg.get(position) if position is not None else None
        self._circle(center, 0.1 * self.tile, self.palette.track, None)
        if port is None:
            return
        start = self._end_frame(node_id, port, 0.12 * self.tile)
        stop = self._end_frame(node_id, port, 0.55 * self.tile)
        if start is not None and stop is not None:
            self._line([start[0], stop[0]], self.palette.indicator, 0.09 * self.tile)

    def _four_port(self, node: Node, center: Point) -> None:
        t = self.tile
        stubs: dict[Port, Point] = {}
        for port in (Port.A, Port.B, Port.C, Port.D):
            frame = self._end_frame(node.id, port, 0.5 * t)
            if frame is not None:
                stubs[port] = frame[0]
        slips: list[tuple[Port, Port]] = []
        if node.kind == NodeKind.EKW:
            slips = [(Port.A, Port.D)]
        elif node.kind == NodeKind.DKW:
            slips = [(Port.A, Port.D), (Port.B, Port.C)]
        for a, b in slips:
            if a in stubs and b in stubs:
                self._curve(stubs[a], center, stubs[b], self.palette.track, 0.12 * t)
        self._circle(center, 0.08 * t, self.palette.track, None)
        state = self.doc.state.switches.get(node.id)
        if node.kind == NodeKind.CROSSING or state is None or not state.active_paths:
            return
        for a, b in state.active_paths:
            if a in stubs and b in stubs:
                self._curve(stubs[a], center, stubs[b], self.palette.indicator, 0.07 * t)

    def _curve(self, a: Point, control: Point, b: Point, color: str, width: float) -> None:
        self.dwg.add(
            self.dwg.path(
                d=(
                    f"M {a[0]:.2f} {a[1]:.2f} "
                    f"Q {control[0]:.2f} {control[1]:.2f} {b[0]:.2f} {b[1]:.2f}"
                ),
                fill="none",
                stroke=color,
                stroke_width=round(width, 3),
                stroke_linecap="round",
            )
        )

    def _node_labels(self, node: Node, center: Point) -> None:
        t = self.tile
        size = 0.34 * t
        if node.label:
            self._text(node.label, _add(center, (0.0, -0.62 * t)), size)
        if node.half_labels:
            for label, dx in zip(node.half_labels[:2], (-0.55 * t, 0.55 * t), strict=False):
                self._text(label, _add(center, (dx, 0.62 * t)), size * 0.75)

    def signals(self) -> None:
        t = self.tile
        for signal in self.doc.signals:
            line = self.polylines.get(signal.at.edge)
            if line is None:
                continue
            glyph = _glyph_at(_signal_glyph(line, signal, t), signal.geometry)
            state = self.doc.state.signals.get(signal.id)
            aspect = state.aspect if state is not None else SignalAspect.UNKNOWN
            foot, second, radius = _signal_parts(glyph.center, glyph.tangent, t)
            body = self.palette.signal_body
            self._line(
                [
                    _add(foot, _scale(glyph.normal, 0.12 * t)),
                    _add(foot, _scale(glyph.normal, -0.12 * t)),
                ],
                body,
                0.06 * t,
            )
            self._line([foot, glyph.center], body, 0.06 * t)
            lamps = _ASPECT_LAMPS[aspect]
            if len(lamps) > 1:
                self._circle(second, radius * 0.8, lamps[1] or self.palette.lamp_dark, body)
            main_color = lamps[0] or self.palette.lamp_dark
            if signal.kind == SignalKind.SHUNT:
                r = radius
                cx, cy = glyph.center
                self.dwg.add(
                    self.dwg.rect(
                        insert=(round(cx - r, 2), round(cy - r, 2)),
                        size=(round(2 * r, 2), round(2 * r, 2)),
                        fill=main_color,
                        stroke=body,
                        stroke_width=round(0.04 * t, 3),
                    )
                )
            elif signal.kind == SignalKind.DISTANT:
                self._circle(glyph.center, radius, main_color, body)
                self._circle(glyph.center, radius * 0.45, body, None)
            else:
                self._circle(glyph.center, radius, main_color, body)
            if signal.label:
                label_at = _add(glyph.center, _scale(glyph.normal, 0.42 * t))
                self._text(signal.label, label_at, 0.3 * t)

    def derailers(self) -> None:
        t = self.tile
        for derailer in self.doc.derailers:
            line = self.polylines.get(derailer.at.edge)
            if line is None:
                continue
            glyph = _glyph_at(_derailer_glyph(line, derailer, t), derailer.geometry)
            state = self.doc.state.derailers.get(derailer.id)
            position = state.position if state is not None else DerailerPosition.UNKNOWN
            c, n, tg = glyph.center, glyph.normal, glyph.tangent
            if position == DerailerPosition.APPLIED:
                # Applied: a yellow wedge lying across the rail.
                points = [
                    _add(c, _scale(n, 0.3 * t)),
                    _add(_add(c, _scale(n, -0.3 * t)), _scale(tg, 0.15 * t)),
                    _add(_add(c, _scale(n, -0.3 * t)), _scale(tg, -0.15 * t)),
                ]
                self._polygon(points, _YELLOW, self.palette.track)
            else:
                # Removed: the wedge swung off the rail, beside the track.
                base = _add(c, _scale(n, 0.22 * t))
                points = [
                    _add(base, _scale(n, 0.08 * t)),
                    _add(_add(base, _scale(n, -0.08 * t)), _scale(tg, 0.15 * t)),
                    _add(_add(base, _scale(n, -0.08 * t)), _scale(tg, -0.15 * t)),
                ]
                fill = self.palette.lamp_dark if position == DerailerPosition.REMOVED else _GREY
                self._polygon(points, fill, self.palette.track)
            if derailer.label:
                self._text(derailer.label, _add(c, _scale(n, -0.55 * t)), 0.28 * t)

    def track_labels(self) -> None:
        t = self.tile
        for edge in self.doc.topology.edges:
            line = self.polylines.get(edge.id)
            if line is None or not edge.label:
                continue
            mid, tangent = _point_at(line, _length(line) / 2)
            normal = (tangent[1], -tangent[0]) if tangent[0] >= 0 else (-tangent[1], tangent[0])
            self._text(edge.label, _add(mid, _scale(normal, 0.42 * t)), 0.3 * t)


def render_svg(doc: SceneAnnotation, *, style: str = "stelltisch") -> str:
    """Draw a scene as an SVG document.

    The drawing uses the document's own pixel geometry (node points, track
    polylines; signal and derailer glyphs are placed at their stored geometry
    or, without it, beside the track at their attachment) in an SVG exactly
    ``source.width`` x ``source.height`` pixels large. A document without a
    point for every node and a polyline for every edge (typically a model
    prediction) is first laid out with :func:`layout_scene` at its source size.
    ``stelltisch`` draws the grey-green tile mosaic of a German
    Drucktastenstellwerk (tile pitch from ``meta["render"]`` when present),
    black tracks with lamp strips (red occupancy, yellow route illumination),
    white switch-position and slip-path indicators; ``estw`` draws the dark
    screen of an electronic interlocking with grey, green (route set) and red
    (occupied) tracks. Signals show their aspect colours, derailers whether
    they are applied.

    Args:
        doc: The scene to render.
        style: Visual style preset, one of :data:`STYLES`.

    Returns:
        The SVG document as a string.

    Raises:
        ValueError: If ``style`` is unknown.
    """
    if style not in _PALETTES:
        msg = f"unknown style {style!r}; expected one of {', '.join(STYLES)}"
        raise ValueError(msg)
    if not has_complete_geometry(doc):
        doc = layout_scene(doc)
    tile, origin = _render_params(doc)
    painter = _Painter(doc, style, tile, origin)
    painter.background()
    painter.tracks()
    painter.nodes_layer()
    painter.derailers()
    painter.signals()
    painter.track_labels()
    return str(painter.dwg.tostring())


def svg_to_bytes(svg: str, *, scale: float = 1.0) -> bytes:
    """Rasterise an SVG document to PNG bytes with ``cairosvg``.

    Args:
        svg: The SVG document.
        scale: Rasterisation scale factor.

    Returns:
        The encoded PNG.

    Raises:
        ValueError: If ``scale`` is not positive.
    """
    if scale <= 0:
        msg = f"scale must be positive, got {scale}"
        raise ValueError(msg)
    import cairosvg  # type: ignore[import-untyped]

    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), scale=scale)
    if not isinstance(png, bytes):  # pragma: no cover - cairosvg returns bytes without write_to
        msg = "cairosvg did not return PNG bytes"
        raise TypeError(msg)
    return png


def svg_to_png(svg: str, out: Path, *, scale: float = 1.0) -> Path:
    """Rasterise an SVG document to PNG.

    Uses ``cairosvg.svg2png`` with ``scale`` as the output scale factor; the
    parent directory of ``out`` is created when missing.

    Args:
        svg: The SVG document.
        out: Destination PNG path.
        scale: Rasterisation scale factor.

    Returns:
        The written PNG path.

    Raises:
        ValueError: If ``scale`` is not positive.
    """
    png = svg_to_bytes(svg, scale=scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    return out


def render_png(doc: SceneAnnotation, *, style: str = "stelltisch", scale: float = 1.0) -> bytes:
    """Render a scene straight to PNG bytes (:func:`render_svg` then :func:`svg_to_bytes`).

    Args:
        doc: The scene to render.
        style: Visual style preset, one of :data:`STYLES`.
        scale: Rasterisation scale factor.

    Returns:
        The encoded PNG, ``source.width * scale`` x ``source.height * scale`` pixels.

    Raises:
        ValueError: If ``style`` is unknown or ``scale`` is not positive.
    """
    return svg_to_bytes(render_svg(doc, style=style), scale=scale)
