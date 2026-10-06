"""Assignment of predicted elements to ground-truth elements.

The same assignment pairs one model's elements with another's in the ground-truth pre-fill
consensus. The cost of pairing two elements is ``w_label * label_distance + w_kind * kind_mismatch +
w_geom * geometry_distance``; the Hungarian method finds the cheapest one-to-one assignment
and pairs above ``threshold`` are dropped. Geometry distances are normalised by
:data:`GEOM_SCALE` times the image diagonal, so two points that far apart are as different
as two elements of different kinds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz.distance import Levenshtein

from rail_vision_bench.schema.models import (
    Derailer,
    Geometry,
    Node,
    Route,
    SceneAnnotation,
    Signal,
    Track,
)

Family = Literal["nodes", "edges", "signals", "derailers", "routes"]
Element = Node | Track | Signal | Derailer | Route
Point = tuple[float, float]

GEOM_SCALE: Final = 0.05
"""Distance (as a fraction of the image diagonal) at which geometry counts as fully different."""

_NO_GEOMETRY: Final = 1.0
UNMATCHABLE: Final = 1e6
"""Cost of a pair whose geometries are both known and too far apart to be one element."""
_POLYLINE_SAMPLES: Final = 16


class Match(BaseModel):
    """One accepted ground-truth/prediction pair and its assignment cost."""

    model_config = ConfigDict(extra="forbid")

    gt_id: str
    pred_id: str
    cost: Annotated[float, Field(ge=0)]


def elements(doc: SceneAnnotation, family: Family) -> Sequence[Element]:
    """Return one element family of a document."""
    if family == "nodes":
        return doc.topology.nodes
    if family == "edges":
        return doc.topology.edges
    if family == "signals":
        return doc.signals
    if family == "derailers":
        return doc.derailers
    return doc.routes


def anchor(geometry: Geometry | None) -> Point | None:
    """Return a representative point: the point, the bbox centre or the polyline midpoint."""
    if geometry is None:
        return None
    if geometry.point is not None:
        return geometry.point
    if geometry.bbox is not None:
        x0, y0, x1, y1 = geometry.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    if geometry.polyline:
        samples = sample_polyline(geometry.polyline, 3)
        return samples[1]
    return None


def sample_polyline(polyline: Sequence[Point], n: int = _POLYLINE_SAMPLES) -> list[Point]:
    """Return ``n`` points evenly spaced by arc length along a polyline (n >= 2)."""
    if len(polyline) == 1:
        return [polyline[0]] * n
    lengths = [math.dist(a, b) for a, b in pairwise(polyline)]
    total = sum(lengths)
    if total == 0:
        return [polyline[0]] * n
    out: list[Point] = []
    for i in range(n):
        target = total * i / (n - 1)
        walked = 0.0
        for (a, b), length in zip(pairwise(polyline), lengths, strict=True):
            if walked + length >= target or (a, b) == (polyline[-2], polyline[-1]):
                t = 0.0 if length == 0 else min(1.0, (target - walked) / length)
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                break
            walked += length
    return out


def _point_segment(p: Point, a: Point, b: Point) -> float:
    """Distance from ``p`` to the segment ``a``-``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == dy == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def polyline_distance(first: Sequence[Point], second: Sequence[Point]) -> float:
    """Symmetric mean distance between two polylines, in pixels."""

    def one_way(src: Sequence[Point], dst: Sequence[Point]) -> float:
        if len(dst) == 1:
            return sum(math.dist(p, dst[0]) for p in sample_polyline(src)) / _POLYLINE_SAMPLES
        return (
            sum(
                min(_point_segment(p, a, b) for a, b in pairwise(dst)) for p in sample_polyline(src)
            )
            / _POLYLINE_SAMPLES
        )

    return (one_way(first, second) + one_way(second, first)) / 2


def geometry_distance(first: Geometry | None, second: Geometry | None, diagonal: float) -> float:
    """Normalised geometry distance in ``[0, 1]``; 1 when either side has no geometry."""
    if first is not None and second is not None and first.polyline and second.polyline:
        pixels = polyline_distance(first.polyline, second.polyline)
    else:
        a, b = anchor(first), anchor(second)
        if a is None or b is None:
            return _NO_GEOMETRY
        pixels = math.dist(a, b)
    return min(1.0, pixels / (GEOM_SCALE * diagonal))


def label_distance(first: str | None, second: str | None) -> float:
    """Normalised edit distance of two labels; 0 when both are missing, 1 when one is."""
    if first is None and second is None:
        return 0.0
    if first is None or second is None:
        return 1.0
    return float(Levenshtein.normalized_distance(first.strip(), second.strip()))


def _kind(element: Element) -> str | None:
    """The element's kind, if its family has one."""
    kind = getattr(element, "kind", None)
    return None if kind is None else str(kind)


def pair_cost(
    first: Element,
    second: Element,
    diagonal: float,
    *,
    w_label: float = 0.5,
    w_kind: float = 0.3,
    w_geom: float = 0.2,
) -> float:
    """The assignment cost of two elements of one family.

    When both elements carry geometry and it is at least ``GEOM_SCALE`` of the diagonal
    apart, the pair is :data:`UNMATCHABLE`: labels and kinds alone (often missing or shared
    by dozens of elements on a panel) never pair two distant elements.
    """
    kind_mismatch = 0.0 if _kind(first) == _kind(second) else 1.0
    if isinstance(first, Route) or isinstance(second, Route):
        geometry = _NO_GEOMETRY
    else:
        geometry = geometry_distance(first.geometry, second.geometry, diagonal)
        both = anchor(first.geometry) is not None and anchor(second.geometry) is not None
        if both and geometry >= 1.0:
            return UNMATCHABLE
    return (
        w_label * label_distance(first.label, second.label)
        + w_kind * kind_mismatch
        + w_geom * geometry
    )


def assign(cost: Sequence[Sequence[float]], threshold: float) -> list[tuple[int, int, float]]:
    """Solve the assignment problem and keep the pairs whose cost is at most ``threshold``.

    Args:
        cost: A rows x columns cost matrix (rows and columns may differ in number).
        threshold: Maximum cost of an accepted pair.

    Returns:
        ``(row, column, cost)`` triples.
    """
    if not cost or not cost[0]:
        return []
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

    rows, cols = linear_sum_assignment(cost)
    return [
        (int(r), int(c), float(cost[r][c]))
        for r, c in zip(rows, cols, strict=True)
        if cost[r][c] <= threshold
    ]


def match_elements(
    gt: SceneAnnotation,
    pred: SceneAnnotation,
    *,
    family: Family,
    w_label: float = 0.5,
    w_kind: float = 0.3,
    w_geom: float = 0.2,
    threshold: float = 0.6,
) -> list[Match]:
    """Match one element family of a prediction against the ground truth.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        family: Which element list to match.
        w_label: Weight of the label distance.
        w_kind: Weight of the kind mismatch.
        w_geom: Weight of the geometry distance.
        threshold: Maximum cost of an accepted pair.

    Returns:
        The accepted pairs, ordered by ground-truth position.
    """
    truth, guess = elements(gt, family), elements(pred, family)
    diagonal = math.hypot(gt.source.width, gt.source.height)
    cost = [
        [pair_cost(t, g, diagonal, w_label=w_label, w_kind=w_kind, w_geom=w_geom) for g in guess]
        for t in truth
    ]
    return sorted(
        (
            Match(gt_id=truth[r].id, pred_id=guess[c].id, cost=value)
            for r, c, value in assign(cost, threshold)
        ),
        key=lambda match: [e.id for e in truth].index(match.gt_id),
    )
