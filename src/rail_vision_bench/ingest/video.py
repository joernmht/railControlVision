"""Frame extraction from video files and RTSP streams with PyAV (imageio fallback)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

_RTSP_OPTIONS = {"rtsp_transport": "tcp", "timeout": "5000000"}


@dataclass(frozen=True)
class Frame:
    """One decoded frame of a video source."""

    index: int
    """Zero-based index of the frame in the decoded (possibly subsampled) sequence."""
    timestamp_ms: int
    """Presentation time of the frame in milliseconds since the start of the source."""
    image: Image
    """The decoded frame as an RGB PIL image."""


class _Subsampler:
    """Keeps a frame whenever its timestamp reaches the next slot of the target rate."""

    def __init__(self, fps: float | None) -> None:
        self._period_ms = None if fps is None else 1000.0 / fps
        self._next_ms: float | None = None

    def keep(self, timestamp_ms: float) -> bool:
        if self._period_ms is None:
            return True
        if self._next_ms is None:
            self._next_ms = timestamp_ms
        # Half a millisecond of slack absorbs the rounding of container time bases.
        if timestamp_ms + 0.5 < self._next_ms:
            return False
        while self._next_ms <= timestamp_ms + 0.5:
            self._next_ms += self._period_ms
        return True


def _iter_pyav(container: Any, fps: float | None) -> Iterator[Frame]:
    """Decode the first video stream of an open PyAV container, closing it at the end."""
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate) if stream.average_rate else None
        start_s = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        sampler = _Subsampler(fps)
        index = 0
        for decoded_index, frame in enumerate(container.decode(stream)):
            if frame.time is not None:
                seconds = float(frame.time) - start_s
            elif rate:
                seconds = decoded_index / rate
            else:
                seconds = 0.0
            timestamp_ms = max(0.0, seconds * 1000.0)
            if not sampler.keep(timestamp_ms):
                continue
            yield Frame(index=index, timestamp_ms=round(timestamp_ms), image=frame.to_image())
            index += 1
    finally:
        container.close()


def _iter_imageio(resource: Any, meta: dict[str, Any], fps: float | None) -> Iterator[Frame]:
    """Decode frames through an open imageio resource, closing it at the end."""
    from PIL import Image as PILImage

    try:
        rate = float(meta.get("fps") or 0.0) or None
        sampler = _Subsampler(fps)
        index = 0
        for decoded_index, array in enumerate(resource.iter()):
            timestamp_ms = decoded_index * 1000.0 / rate if rate else 0.0
            if not sampler.keep(timestamp_ms):
                continue
            image = PILImage.fromarray(array).convert("RGB")
            yield Frame(index=index, timestamp_ms=round(timestamp_ms), image=image)
            index += 1
    finally:
        resource.close()


def iter_frames(source: str, *, fps: float | None = None) -> Iterator[Frame]:
    """Decode a video file or RTSP URL into frames.

    ``source`` is opened eagerly with PyAV (``av.open``; RTSP over TCP), so a
    missing file or an unreachable stream fails on the call rather than on the
    first ``next()``; containers PyAV cannot open are retried with ``imageio``.
    Frames of the first video stream are decoded lazily. With ``fps`` set,
    frames are subsampled by presentation time: a frame is kept whenever its
    timestamp reaches the next ``1 / fps`` slot, so the output rate never
    exceeds ``fps`` and the source rate is kept when it is lower. ``index``
    counts the frames actually yielded.

    Args:
        source: A file path or an ``rtsp://`` URL.
        fps: Target frame rate; ``None`` keeps every decoded frame.

    Returns:
        An iterator over the decoded frames; exhausting or closing it closes
        the source.

    Raises:
        ValueError: If ``fps`` is not positive, or neither PyAV nor imageio can
            open ``source`` (the PyAV error is chained).
        FileNotFoundError: If ``source`` is a local path that does not exist.
    """
    if fps is not None and fps <= 0:
        msg = f"fps must be positive, got {fps}"
        raise ValueError(msg)
    import av

    is_stream = "://" in source
    if not is_stream:
        from pathlib import Path

        if not Path(source).exists():
            msg = f"video source not found: {source}"
            raise FileNotFoundError(msg)
    options = _RTSP_OPTIONS if source.startswith(("rtsp://", "rtsps://")) else {}
    try:
        container = av.open(source, options=options)
        if not container.streams.video:
            container.close()
            msg = f"{source} has no video stream"
            raise ValueError(msg)
    except av.FFmpegError as pyav_error:
        import imageio.v3 as iio

        resource: Any = None
        try:
            resource = iio.imopen(source, "r")
            meta = dict(resource.metadata())
        except Exception as fallback_error:
            if resource is not None:
                resource.close()
            msg = f"cannot open video source {source}: {pyav_error}"
            raise ValueError(msg) from fallback_error
        return _iter_imageio(resource, meta, fps)
    return _iter_pyav(container, fps)
