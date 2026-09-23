"""Geometric preprocessing of panel photos and screenshots with OpenCV."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


def perspective_correct(image: Image, quad: Sequence[tuple[float, float]]) -> Image:
    """Rectify a panel photographed at an angle.

    ``cv2.getPerspectiveTransform`` maps the four ``quad`` corners (top-left,
    top-right, bottom-right, bottom-left) onto an axis-aligned rectangle whose
    width and height are the longer of the two opposite quad sides, and
    ``cv2.warpPerspective`` (bicubic) resamples the image into it.

    Args:
        image: The source image.
        quad: Four pixel corners of the panel in the source image.

    Returns:
        The rectified RGB image.

    Raises:
        ValueError: If ``quad`` does not have exactly four corners or is degenerate.
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    if len(quad) != 4:
        msg = f"quad must have four corners, got {len(quad)}"
        raise ValueError(msg)
    tl, tr, br, bl = (tuple(map(float, corner)) for corner in quad)
    width = round(max(math.dist(tl, tr), math.dist(bl, br)))
    height = round(max(math.dist(tl, bl), math.dist(tr, br)))
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    area = 0.5 * abs(
        sum(src[i, 0] * src[(i + 1) % 4, 1] - src[(i + 1) % 4, 0] * src[i, 1] for i in range(4))
    )
    if width < 1 or height < 1 or area < 1.0:
        msg = f"quad is degenerate: {list(quad)}"
        raise ValueError(msg)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    rgb = np.asarray(image.convert("RGB"))
    warped = cv2.warpPerspective(
        rgb, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return PILImage.fromarray(warped)


def crop_zoom(
    image: Image, bbox: tuple[float, float, float, float], *, scale: float = 2.0
) -> Image:
    """Crop a region and upscale it for label reading.

    ``bbox`` (``x0, y0, x1, y1`` in pixels) is clipped to the image, rounded
    outwards to whole pixels, cropped and resized by ``scale`` with
    ``cv2.INTER_CUBIC``; this is what the agentic reader node calls on every
    element it needs to transcribe.

    Args:
        image: The source image.
        bbox: The region to crop, in pixels.
        scale: Upscaling factor applied to the crop.

    Returns:
        The enlarged RGB crop.

    Raises:
        ValueError: If ``scale`` is not positive or the clipped ``bbox`` is empty.
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    if scale <= 0:
        msg = f"scale must be positive, got {scale}"
        raise ValueError(msg)
    x0, y0, x1, y1 = clip_bbox(bbox, image.width, image.height)
    rgb = np.asarray(image.convert("RGB"))[y0:y1, x0:x1]
    size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
    return PILImage.fromarray(cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC))


def clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clip a pixel box to an image and round it outwards to whole pixels.

    Args:
        bbox: ``x0, y0, x1, y1`` in pixels; the corners may come in any order.
        width: Image width.
        height: Image height.

    Returns:
        The integer box ``x0, y0, x1, y1`` with ``x0 < x1`` and ``y0 < y1``.

    Raises:
        ValueError: If the box has no area inside the image.
    """
    xa, ya, xb, yb = bbox
    x0 = max(0, math.floor(min(xa, xb)))
    y0 = max(0, math.floor(min(ya, yb)))
    x1 = min(width, math.ceil(max(xa, xb)))
    y1 = min(height, math.ceil(max(ya, yb)))
    if x1 <= x0 or y1 <= y0:
        msg = f"bbox {tuple(bbox)} has no area inside the {width}x{height} image"
        raise ValueError(msg)
    return x0, y0, x1, y1


def _cluster(positions: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Merge (position, weight) pairs closer than ``tol`` into weighted means."""
    merged: list[tuple[float, float]] = []
    for pos, weight in sorted(positions):
        if merged and pos - merged[-1][0] <= tol:
            last_pos, last_weight = merged[-1]
            total = last_weight + weight
            merged[-1] = ((last_pos * last_weight + pos * weight) / total, total)
        else:
            merged.append((pos, weight))
    return merged


def _nearest(sorted_positions: list[float], value: float) -> float:
    """Return the element of a non-empty sorted list closest to ``value``."""
    index = bisect.bisect_left(sorted_positions, value)
    neighbours = sorted_positions[max(0, index - 1) : index + 1]
    return min(neighbours, key=lambda pos: abs(pos - value))


def _periodic_grid(
    candidates: list[tuple[float, float]], extent: int, *, min_period: float, tol: float
) -> list[float]:
    """Fit the regular grid that best explains the candidate seam positions.

    The gaps between each candidate and its next few neighbours are tried as
    the period and every candidate as the phase; a grid scores one point per
    grid line with a candidate within ``tol`` and loses one per line without,
    over the span the candidates cover. Track edges and labels add isolated
    candidates that no regular grid explains, so the seams win; ties go to the
    longer period.
    """
    if len(candidates) < 2:
        return []
    positions = [pos for pos, _ in candidates]
    lo, hi = positions[0], positions[-1]
    periods = {
        round(b - a, 1)
        for i, a in enumerate(positions)
        for b in positions[i + 1 : i + 4]
        if b - a >= min_period
    }
    best: tuple[int, float, list[float]] | None = None
    for period in sorted(periods):
        for phase in positions:
            line = phase - math.floor((phase - lo + tol) / period) * period
            lines: list[float] = []
            hits = 0
            misses = 0
            while line <= hi + tol:
                near = _nearest(positions, line)
                if abs(near - line) <= tol:
                    hits += 1
                    lines.append(near)
                else:
                    misses += 1
                    lines.append(line)
                line += period
            score = hits - misses
            if hits >= 2 and (best is None or (score, period) > (best[0], best[1])):
                best = (score, period, lines)
    if best is None:
        return []
    return [line for line in best[2] if 0 <= line <= extent]


def detect_tile_seams(image: Image) -> list[tuple[float, float, float, float]]:
    """Find the tile grid of a Stelltisch (Drucktastenstellwerk mosaic panel).

    Canny edges and a probabilistic Hough transform (``cv2.HoughLinesP``)
    give long horizontal and vertical line segments; their positions are
    clustered per axis and a regular grid (period and phase) is fitted to them,
    which discards track edges and label strokes that do not repeat with the
    tile pitch. Every cell between two consecutive grid lines on both axes is a
    tile.

    Args:
        image: A rectified panel image.

    Returns:
        Tile bounding boxes ``(x0, y0, x1, y1)`` in pixels, row by row from the
        top left; empty when no regular grid is found on either axis.
    """
    import cv2
    import numpy as np

    grey = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = grey.shape
    edges = cv2.Canny(grey, 30, 90)
    min_length = max(10, min(width, height) // 4)
    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(10, min_length // 2),
        minLineLength=min_length,
        maxLineGap=max(3, min(width, height) // 50),
    )
    vertical: list[tuple[float, float]] = []
    horizontal: list[tuple[float, float]] = []
    for x0, y0, x1, y1 in [] if segments is None else segments.reshape(-1, 4).tolist():
        if abs(x1 - x0) <= 2 and abs(y1 - y0) > 0:
            vertical.append(((x0 + x1) / 2.0, float(abs(y1 - y0))))
        elif abs(y1 - y0) <= 2 and abs(x1 - x0) > 0:
            horizontal.append(((y0 + y1) / 2.0, float(abs(x1 - x0))))
    tol = 3.0
    min_period = 8.0
    xs = _periodic_grid(_cluster(vertical, tol), width, min_period=min_period, tol=tol)
    ys = _periodic_grid(_cluster(horizontal, tol), height, min_period=min_period, tol=tol)
    if len(xs) < 2 or len(ys) < 2:
        return []
    return [
        (float(x0), float(y0), float(x1), float(y1))
        for y0, y1 in pairwise(ys)
        for x0, x1 in pairwise(xs)
    ]
