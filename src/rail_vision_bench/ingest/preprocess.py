"""Geometric preprocessing of panel photos and screenshots (stubs)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


def perspective_correct(image: Image, quad: Sequence[tuple[float, float]]) -> Image:
    """Rectify a panel photographed at an angle.

    Intended implementation: ``cv2.getPerspectiveTransform`` from the four
    ``quad`` corners (top-left, top-right, bottom-right, bottom-left) to an
    axis-aligned rectangle followed by ``cv2.warpPerspective``.

    Args:
        image: The source image.
        quad: Four pixel corners of the panel in the source image.

    Returns:
        The rectified image.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.ingest.preprocess.perspective_correct is not implemented in the "
        "skeleton: warp the panel quadrilateral to a rectangle with OpenCV"
    )


def crop_zoom(
    image: Image, bbox: tuple[float, float, float, float], *, scale: float = 2.0
) -> Image:
    """Crop a region and upscale it for label reading.

    Intended implementation: crop ``bbox`` (``x0, y0, x1, y1`` in pixels) and
    resize by ``scale`` with ``cv2.INTER_CUBIC``; this is what the agentic
    reader node calls on every element it needs to transcribe.

    Args:
        image: The source image.
        bbox: The region to crop, in pixels.
        scale: Upscaling factor applied to the crop.

    Returns:
        The enlarged crop.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.ingest.preprocess.crop_zoom is not implemented in the skeleton: "
        "crop a bounding box and upscale it with OpenCV"
    )


def detect_tile_seams(image: Image) -> list[tuple[float, float, float, float]]:
    """Find the tile grid of a Stelltisch (Drucktastenstellwerk mosaic panel).

    Intended implementation: Canny edges plus a probabilistic Hough transform
    (``cv2.HoughLinesP``) to recover the horizontal and vertical seams
    between the panel tiles, returned as one bounding box per tile.

    Args:
        image: A rectified panel image.

    Returns:
        Tile bounding boxes ``(x0, y0, x1, y1)`` in pixels.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.ingest.preprocess.detect_tile_seams is not implemented in the "
        "skeleton: recover the mosaic tile grid with OpenCV edge and line detection"
    )
