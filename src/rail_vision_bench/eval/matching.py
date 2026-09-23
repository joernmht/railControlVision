"""Assignment of predicted elements to ground-truth elements.

Every family is matched with one cost model: for a ground-truth element ``g``
and a predicted element ``p`` the cost is the weighted mean

``(w_label * label + w_kind * kind + w_geom * geom) / (sum of the weights used)``

over the terms that are *available* for the pair; a term that cannot be
computed (both labels missing, no kind for the family, no geometry on either
side) is dropped and the remaining weights are renormalised, so a perfect
prediction always costs 0 and every cost lies in ``[0, 1]``. The three terms
per family:

========== ============================ ======================= ===============================
family     label term                   kind term               geometry term
========== ============================ ======================= ===============================
nodes      ``label`` (else half labels) ``NodeKind`` equality   pixel geometry
edges      ``label``                    ``TrackKind`` equality  endpoints under the node
                                                                matching (mean with pixel
                                                                geometry when both have it)
signals    ``label``                    ``SignalKind`` equality attachment under the edge
                                                                matching (mean with pixel
                                                                geometry when both have it)
derailers  ``label``                    (none)                  as for signals
routes     ``label``                    start/end mismatch      ``1 - Jaccard`` of the path edge
                                        under the signal/node   sets under the edge matching
                                        matchings
========== ============================ ======================= ===============================

* **label**: ``rapidfuzz`` normalised Levenshtein distance of the stripped,
  case-sensitive labels (``a`` and ``A`` are different signals on German
  panels); 1 when exactly one side has a label, unavailable when neither has.
* **pixel geometry**: coordinates are divided by the document's own
  ``source.width``/``source.height``; the box of an element is its ``bbox``,
  else the bounding box of its ``polyline``; ``1 - IoU`` when both boxes have
  a positive area, otherwise the distance of the centres (``point``, else box
  centre) divided by ``sqrt(2)``.
* **endpoints** (edges): per endpoint 1 when the predicted node is matched to
  the ground-truth node and the port agrees, 0.5 when only the node agrees,
  0 otherwise; the distance is ``1 -`` the better mean over both orientations.
* **attachment** (signals, derailers): 1 when the predicted edge is not
  matched to the ground-truth edge, else the offset difference (with the
  predicted offset mirrored when its edge runs the other way).

Families depend on each other, so :func:`match_all` matches them in the order
nodes, edges, signals, derailers, routes. The Hungarian method
(``scipy.optimize.linear_sum_assignment``) solves every assignment; pairs
costing more than ``threshold`` are never accepted, and a tiny tie-breaker
prefers pairs with identical ids among otherwise equal candidates (needed for
unlabelled, geometry-free scenes such as the synthetic generator's).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz.distance import Levenshtein

from rail_vision_bench.schema.models import (
    Derailer,
    EdgeAttachment,
    Geometry,
    Node,
    Route,
    SceneAnnotation,
    Signal,
    Source,
    Track,
)

Family = Literal["nodes", "edges", "signals", "derailers", "routes"]
FAMILIES: Final[tuple[Family, ...]] = get_args(Family)
"""Element families in dependency order (later families use earlier matchings)."""

_TIE_BREAK: Final[float] = 1e-6
_Box = tuple[float, float, float, float]
_Point = tuple[float, float]


class Match(BaseModel):
    """One accepted ground-truth/prediction pair and its assignment cost."""

    model_config = ConfigDict(extra="forbid")

    gt_id: str
    pred_id: str
    cost: Annotated[float, Field(ge=0)]
    family: Family | None = Field(
        default=None, description="The element family; None when built by hand."
    )


@dataclass(frozen=True)
class _Weights:
    label: float
    kind: float
    geom: float
    threshold: float


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

    The cost model per family is described in the module docstring. Families
    that depend on other matchings (edges on nodes, signals and derailers on
    edges, routes on all of them) compute those first with the same weights.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        family: Which element list to match.
        w_label: Weight of the label distance.
        w_kind: Weight of the kind mismatch.
        w_geom: Weight of the geometry distance.
        threshold: Maximum cost of an accepted pair.

    Returns:
        The accepted pairs, sorted by ground-truth id.

    Raises:
        ValueError: If ``family`` is unknown or a weight is negative.
    """
    if family not in FAMILIES:
        msg = f"unknown family {family!r}; expected one of {FAMILIES}"
        raise ValueError(msg)
    upto = FAMILIES[: FAMILIES.index(family) + 1]
    return _match_families(gt, pred, _weights(w_label, w_kind, w_geom, threshold), upto)[family]


def match_all(
    gt: SceneAnnotation,
    pred: SceneAnnotation,
    *,
    w_label: float = 0.5,
    w_kind: float = 0.3,
    w_geom: float = 0.2,
    threshold: float = 0.6,
) -> dict[Family, list[Match]]:
    """Match every element family, in dependency order.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        w_label: Weight of the label distance.
        w_kind: Weight of the kind mismatch.
        w_geom: Weight of the geometry distance.
        threshold: Maximum cost of an accepted pair.

    Returns:
        The accepted pairs per family.

    Raises:
        ValueError: If a weight is negative.
    """
    return _match_families(gt, pred, _weights(w_label, w_kind, w_geom, threshold), FAMILIES)


def _weights(w_label: float, w_kind: float, w_geom: float, threshold: float) -> _Weights:
    if min(w_label, w_kind, w_geom) < 0:
        msg = "matching weights must be non-negative"
        raise ValueError(msg)
    return _Weights(w_label, w_kind, w_geom, threshold)


def _match_families(
    gt: SceneAnnotation, pred: SceneAnnotation, weights: _Weights, families: Sequence[Family]
) -> dict[Family, list[Match]]:
    result: dict[Family, list[Match]] = {}
    for family in families:
        maps = {name: _pred_to_gt(matches) for name, matches in result.items()}
        costs = _cost_matrix(gt, pred, family, weights, maps)
        gt_ids = [element.id for element in _elements(gt, family)]
        pred_ids = [element.id for element in _elements(pred, family)]
        result[family] = _assign(gt_ids, pred_ids, costs, weights.threshold, family)
    return result


def _pred_to_gt(matches: Sequence[Match]) -> dict[str, str]:
    return {match.pred_id: match.gt_id for match in matches}


def _elements(
    doc: SceneAnnotation, family: Family
) -> Sequence[Node | Track | Signal | Derailer | Route]:
    if family == "nodes":
        return doc.topology.nodes
    if family == "edges":
        return doc.topology.edges
    if family == "signals":
        return doc.signals
    if family == "derailers":
        return doc.derailers
    return doc.routes


def _assign(
    gt_ids: Sequence[str],
    pred_ids: Sequence[str],
    costs: list[list[float]],
    threshold: float,
    family: Family,
) -> list[Match]:
    """Solve the assignment and keep the pairs within the threshold."""
    if not gt_ids or not pred_ids:
        return []
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

    blocked = threshold + 1.0
    matrix = [
        [
            (cost if cost <= threshold else blocked)
            + (0.0 if gt_ids[i] == pred_ids[j] else _TIE_BREAK)
            for j, cost in enumerate(row)
        ]
        for i, row in enumerate(costs)
    ]
    rows, cols = linear_sum_assignment(matrix)
    matches = [
        Match(gt_id=gt_ids[i], pred_id=pred_ids[j], cost=costs[i][j], family=family)
        for i, j in zip(
            cast("list[int]", rows.tolist()), cast("list[int]", cols.tolist()), strict=True
        )
        if costs[i][j] <= threshold
    ]
    return sorted(matches, key=lambda match: match.gt_id)


def _cost_matrix(
    gt: SceneAnnotation,
    pred: SceneAnnotation,
    family: Family,
    weights: _Weights,
    maps: Mapping[Family, Mapping[str, str]],
) -> list[list[float]]:
    context = _Context(gt=gt, pred=pred, maps=maps)
    return [
        [_combine(_terms(context, family, g, p), weights) for p in _elements(pred, family)]
        for g in _elements(gt, family)
    ]


@dataclass(frozen=True)
class _Context:
    gt: SceneAnnotation
    pred: SceneAnnotation
    maps: Mapping[Family, Mapping[str, str]]

    def mapped(self, family: Family, pred_id: str) -> str | None:
        return self.maps.get(family, {}).get(pred_id)


_Terms = tuple[float | None, float | None, float | None]


def _combine(terms: _Terms, weights: _Weights) -> float:
    total = 0.0
    weight_sum = 0.0
    for value, weight in zip(terms, (weights.label, weights.kind, weights.geom), strict=True):
        if value is not None and weight > 0:
            total += weight * value
            weight_sum += weight
    return total / weight_sum if weight_sum > 0 else 0.0


def _terms(
    context: _Context,
    family: Family,
    g: Node | Track | Signal | Derailer | Route,
    p: Node | Track | Signal | Derailer | Route,
) -> _Terms:
    if isinstance(g, Node) and isinstance(p, Node):
        return (
            label_distance(_node_label(g), _node_label(p)),
            float(g.kind != p.kind),
            geometry_distance(g.geometry, context.gt.source, p.geometry, context.pred.source),
        )
    if isinstance(g, Track) and isinstance(p, Track):
        return (
            label_distance(g.label, p.label),
            float(g.kind != p.kind),
            _mean(
                _endpoint_distance(context, g, p),
                geometry_distance(g.geometry, context.gt.source, p.geometry, context.pred.source),
            ),
        )
    if isinstance(g, Signal) and isinstance(p, Signal):
        return (
            label_distance(g.label, p.label),
            float(g.kind != p.kind),
            _attached_distance(context, g, p),
        )
    if isinstance(g, Derailer) and isinstance(p, Derailer):
        return (label_distance(g.label, p.label), None, _attached_distance(context, g, p))
    if isinstance(g, Route) and isinstance(p, Route):
        return (
            label_distance(g.label, p.label),
            _route_anchor_distance(context, g, p),
            _route_path_distance(context, g, p),
        )
    msg = f"elements of family {family!r} have mismatched types"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def _node_label(node: Node) -> str | None:
    if node.label:
        return node.label
    return " ".join(node.half_labels) or None


def _mean(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def label_distance(gt_label: str | None, pred_label: str | None) -> float | None:
    """Normalised edit distance of two labels.

    Args:
        gt_label: The ground-truth label.
        pred_label: The predicted label.

    Returns:
        ``None`` when neither side has a (non-blank) label, 1.0 when only one
        side has one, else the ``rapidfuzz`` normalised Levenshtein distance
        of the stripped, case-sensitive strings.
    """
    gt_text = (gt_label or "").strip()
    pred_text = (pred_label or "").strip()
    if not gt_text and not pred_text:
        return None
    if not gt_text or not pred_text:
        return 1.0
    return float(Levenshtein.normalized_distance(gt_text, pred_text))


def geometry_distance(
    gt_geom: Geometry | None, gt_source: Source, pred_geom: Geometry | None, pred_source: Source
) -> float | None:
    """Distance of two pixel geometries in normalised image coordinates.

    Args:
        gt_geom: The ground-truth geometry.
        gt_source: Source of the ground truth (for its width and height).
        pred_geom: The predicted geometry.
        pred_source: Source of the prediction (for its width and height).

    Returns:
        ``1 - IoU`` when both boxes have a positive area, else the centre
        distance divided by ``sqrt(2)`` (so in ``[0, 1]`` for in-bounds
        coordinates, clipped to 1), or ``None`` when either side has no
        usable geometry.
    """
    gt_box = _box(gt_geom, gt_source)
    pred_box = _box(pred_geom, pred_source)
    if gt_box is not None and pred_box is not None and _area(gt_box) > 0 and _area(pred_box) > 0:
        return 1.0 - _iou(gt_box, pred_box)
    gt_centre = _centre(gt_geom, gt_source, gt_box)
    pred_centre = _centre(pred_geom, pred_source, pred_box)
    if gt_centre is None or pred_centre is None:
        return None
    distance = math.dist(gt_centre, pred_centre) / math.sqrt(2.0)
    return min(1.0, distance)


def _box(geom: Geometry | None, source: Source) -> _Box | None:
    if geom is None:
        return None
    if geom.bbox is not None:
        x0, y0, x1, y1 = geom.bbox
    elif geom.polyline:
        xs = [x for x, _ in geom.polyline]
        ys = [y for _, y in geom.polyline]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    return (
        min(x0, x1) / source.width,
        min(y0, y1) / source.height,
        max(x0, x1) / source.width,
        max(y0, y1) / source.height,
    )


def _centre(geom: Geometry | None, source: Source, box: _Box | None) -> _Point | None:
    if geom is not None and geom.point is not None:
        return (geom.point[0] / source.width, geom.point[1] / source.height)
    if box is not None:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    return None


def _area(box: _Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: _Box, b: _Box) -> float:
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _endpoint_distance(context: _Context, g: Track, p: Track) -> float:
    def score(gt_node: str, gt_port: str, pred_node: str, pred_port: str) -> float:
        if context.mapped("nodes", pred_node) != gt_node:
            return 0.0
        return 1.0 if gt_port == pred_port else 0.5

    same = score(g.a.node, g.a.port, p.a.node, p.a.port) + score(
        g.b.node, g.b.port, p.b.node, p.b.port
    )
    flipped = score(g.a.node, g.a.port, p.b.node, p.b.port) + score(
        g.b.node, g.b.port, p.a.node, p.a.port
    )
    return 1.0 - max(same, flipped) / 2.0


def _attached_distance(
    context: _Context, g: Signal | Derailer, p: Signal | Derailer
) -> float | None:
    return _mean(
        _attachment_distance(context, g.at, p.at),
        geometry_distance(g.geometry, context.gt.source, p.geometry, context.pred.source),
    )


def _attachment_distance(context: _Context, g: EdgeAttachment, p: EdgeAttachment) -> float:
    if context.mapped("edges", p.edge) != g.edge:
        return 1.0
    offset = p.offset
    if _edge_reversed(context, g.edge, p.edge):
        offset = 1.0 - offset
    return min(1.0, abs(g.offset - offset))


def edge_reversed(
    gt: SceneAnnotation,
    pred: SceneAnnotation,
    gt_edge: str,
    pred_edge: str,
    node_map: Mapping[str, str],
) -> bool:
    """Tell whether a matched predicted edge runs opposite to its ground-truth edge.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        gt_edge: The ground-truth edge id.
        pred_edge: The matched predicted edge id.
        node_map: Predicted node id to ground-truth node id.

    Returns:
        True when the predicted ``a`` end maps onto the ground-truth ``b`` node
        (or the ``b`` end onto ``a``) and not in the same orientation.
    """
    g = gt.edge_map().get(gt_edge)
    p = pred.edge_map().get(pred_edge)
    if g is None or p is None:
        return False
    pa = node_map.get(p.a.node)
    pb = node_map.get(p.b.node)
    same = int(pa == g.a.node) + int(pb == g.b.node)
    flipped = int(pa == g.b.node) + int(pb == g.a.node)
    return flipped > same


def _edge_reversed(context: _Context, gt_edge: str, pred_edge: str) -> bool:
    return edge_reversed(
        context.gt, context.pred, gt_edge, pred_edge, context.maps.get("nodes", {})
    )


def translate_anchor(maps: Mapping[Family, Mapping[str, str]], pred_id: str) -> str | None:
    """Translate a predicted signal or node id into the ground-truth id space.

    Args:
        maps: Predicted id to ground-truth id per family.
        pred_id: A predicted signal or node id (a route start or end).

    Returns:
        The matched ground-truth id, or None when the element is unmatched.
    """
    return maps.get("signals", {}).get(pred_id) or maps.get("nodes", {}).get(pred_id)


def _route_anchor_distance(context: _Context, g: Route, p: Route) -> float:
    start = translate_anchor(context.maps, p.start) == g.start
    end = translate_anchor(context.maps, p.end) == g.end
    return 1.0 - (int(start) + int(end)) / 2.0


def _route_path_distance(context: _Context, g: Route, p: Route) -> float:
    edges = context.maps.get("edges", {})
    gt_path = set(g.path)
    pred_path = {edges.get(edge, f"\0unmatched:{edge}") for edge in p.path}
    union = gt_path | pred_path
    return 1.0 - len(gt_path & pred_path) / len(union) if union else 0.0
