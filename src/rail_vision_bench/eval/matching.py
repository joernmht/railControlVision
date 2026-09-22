"""Assignment of predicted elements to ground-truth elements."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from rail_vision_bench.schema.models import SceneAnnotation


class Match(BaseModel):
    """One accepted ground-truth/prediction pair and its assignment cost."""

    model_config = ConfigDict(extra="forbid")

    gt_id: str
    pred_id: str
    cost: Annotated[float, Field(ge=0)]


def match_elements(
    gt: SceneAnnotation,
    pred: SceneAnnotation,
    *,
    family: Literal["nodes", "edges", "signals", "derailers", "routes"],
    w_label: float = 0.5,
    w_kind: float = 0.3,
    w_geom: float = 0.2,
    threshold: float = 0.6,
) -> list[Match]:
    """Match one element family of a prediction against the ground truth.

    Intended implementation: build the cost matrix
    ``w_label * label_distance + w_kind * kind_mismatch + w_geom * geometry_distance``
    where ``label_distance`` is a normalised edit distance (``rapidfuzz``),
    ``kind_mismatch`` is 0 or 1, and ``geometry_distance`` is ``1 - IoU`` of
    the bounding boxes or the normalised centre distance when only points are
    available; solve the assignment with the Hungarian method
    (``scipy.optimize.linear_sum_assignment``) and keep pairs whose cost is at
    most ``threshold``.

    Args:
        gt: The ground-truth document.
        pred: The predicted document.
        family: Which element list to match.
        w_label: Weight of the label distance.
        w_kind: Weight of the kind mismatch.
        w_geom: Weight of the geometry distance.
        threshold: Maximum cost of an accepted pair.

    Returns:
        The accepted pairs.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.eval.matching.match_elements is not implemented in the skeleton: "
        "Hungarian assignment over a label/kind/geometry cost matrix"
    )
