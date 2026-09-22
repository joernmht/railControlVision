"""Frame extraction from video files and RTSP streams (stub)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class Frame:
    """One decoded frame of a video source."""

    index: int
    """Zero-based index of the frame in the decoded (possibly subsampled) sequence."""
    timestamp_ms: int
    """Presentation time of the frame in milliseconds since the start of the source."""
    image: Image
    """The decoded frame as an RGB PIL image."""


def iter_frames(source: str, *, fps: float | None = None) -> Iterator[Frame]:
    """Decode a video file or RTSP URL into frames.

    Intended implementation: open ``source`` with PyAV (``av.open``), decode
    the first video stream, optionally subsample to ``fps`` frames per second,
    and fall back to ``imageio`` (ffmpeg plugin) for containers PyAV cannot
    open. Returns an iterator rather than being a generator so the skeleton
    raises immediately instead of on first ``next()``.

    Args:
        source: A file path or an ``rtsp://`` URL.
        fps: Target frame rate; ``None`` keeps every decoded frame.

    Returns:
        An iterator over the decoded frames.

    Raises:
        NotImplementedError: Always, in the skeleton.
    """
    raise NotImplementedError(
        "rail_vision_bench.ingest.video.iter_frames is not implemented in the skeleton: "
        "decode a file or RTSP stream with PyAV (imageio fallback) into Frame objects"
    )
