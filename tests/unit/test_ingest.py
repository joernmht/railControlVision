"""Ingest: still images, video frames, screen capture, preprocessing and quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from PIL import Image

from rail_vision_bench.ingest.image import load_image
from rail_vision_bench.ingest.preprocess import (
    clip_bbox,
    crop_zoom,
    detect_tile_seams,
    perspective_correct,
)
from rail_vision_bench.ingest.quality import QualityMetrics, measure
from rail_vision_bench.ingest.screen import capture_screen
from rail_vision_bench.ingest.video import Frame, iter_frames

# --------------------------------------------------------------------------- still images


def test_load_image_png_rgba_becomes_rgb(tmp_path: Path):
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (12, 7), (10, 20, 30, 128)).save(path)
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size == (12, 7)


def test_load_image_applies_exif_orientation(tmp_path: Path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise to display
    Image.new("RGB", (40, 20), (200, 0, 0)).save(path, exif=exif)
    image = load_image(path)
    assert image.size == (20, 40)


def test_load_image_heic(tmp_path: Path):
    pillow_heif = pytest.importorskip("pillow_heif")
    if not pillow_heif.libheif_info().get("encoders", {}).get("x265"):
        pytest.skip("libheif has no HEVC encoder")
    pillow_heif.register_heif_opener()
    path = tmp_path / "panel.heic"
    Image.new("RGB", (64, 48), (0, 120, 0)).save(path, format="HEIF")
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size == (64, 48)


def test_load_image_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


# --------------------------------------------------------------------------- video


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2 s, 10 fps, 32x32 MPEG-4 file whose frame i has grey level 10 * i."""
    import av

    path = tmp_path_factory.mktemp("video") / "tiny.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(20):
            array = np.full((32, 32, 3), 10 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_iter_frames_all(tiny_video: Path):
    frames = list(iter_frames(str(tiny_video)))
    assert len(frames) == 20
    assert all(isinstance(frame, Frame) for frame in frames)
    assert [frame.index for frame in frames] == list(range(20))
    assert [frame.timestamp_ms for frame in frames] == [100 * i for i in range(20)]
    assert frames[0].image.mode == "RGB"
    assert frames[0].image.size == (32, 32)
    # Frames come out in order: the grey level rises with the index.
    levels = [float(np.asarray(frame.image).mean()) for frame in frames]
    assert levels == sorted(levels)


def test_iter_frames_subsamples(tiny_video: Path):
    frames = list(iter_frames(str(tiny_video), fps=5))
    assert [frame.index for frame in frames] == list(range(10))
    assert [frame.timestamp_ms for frame in frames] == [200 * i for i in range(10)]


def test_iter_frames_higher_fps_keeps_source_rate(tiny_video: Path):
    assert len(list(iter_frames(str(tiny_video), fps=50))) == 20


def test_iter_frames_rejects_bad_fps(tiny_video: Path):
    with pytest.raises(ValueError, match="fps"):
        iter_frames(str(tiny_video), fps=0)


def test_iter_frames_missing_file_fails_eagerly(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        iter_frames(str(tmp_path / "missing.mp4"))


def test_iter_frames_not_a_video(tmp_path: Path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot open"):
        iter_frames(str(path))


# --------------------------------------------------------------------------- screen capture


class _FakeShot:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        pixel = bytes([30, 20, 10, 255])  # BGRA
        self.bgra = pixel * (width * height)


class _FakeMSS:
    grabbed: ClassVar[list[dict[str, int]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.monitors = [
            {"left": 0, "top": 0, "width": 300, "height": 100},
            {"left": 0, "top": 0, "width": 200, "height": 100},
            {"left": 200, "top": 0, "width": 100, "height": 80},
        ]

    def __enter__(self) -> _FakeMSS:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def grab(self, area: dict[str, int]) -> _FakeShot:
        type(self).grabbed.append(area)
        return _FakeShot(area["width"], area["height"])


@pytest.fixture
def fake_mss(monkeypatch: pytest.MonkeyPatch) -> type[_FakeMSS]:
    import mss

    _FakeMSS.grabbed = []
    monkeypatch.setattr(mss, "mss", _FakeMSS)
    return _FakeMSS


def test_capture_screen_monitor(fake_mss: type[_FakeMSS]):
    image = capture_screen(2)
    assert image.mode == "RGB"
    assert image.size == (100, 80)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    assert fake_mss.grabbed == [{"left": 200, "top": 0, "width": 100, "height": 80}]


def test_capture_screen_region_is_monitor_relative(fake_mss: type[_FakeMSS]):
    image = capture_screen(2, region=(10, 5, 50, 40))
    assert image.size == (50, 40)
    assert fake_mss.grabbed == [{"left": 210, "top": 5, "width": 50, "height": 40}]


@pytest.mark.usefixtures("fake_mss")
@pytest.mark.parametrize(
    ("monitor", "region"), [(3, None), (-1, None), (1, (0, 0, 0, 10)), (1, (150, 0, 100, 10))]
)
def test_capture_screen_rejects_bad_arguments(
    monitor: int, region: tuple[int, int, int, int] | None
):
    with pytest.raises(ValueError, match=r"monitor|region"):
        capture_screen(monitor, region=region)


@pytest.mark.requires_display
def test_capture_screen_real_display():
    image = capture_screen(1)
    assert image.mode == "RGB"
    assert image.width > 0
    assert image.height > 0


# --------------------------------------------------------------------------- preprocessing


def test_perspective_correct_rectifies_quad():
    source = Image.new("RGB", (200, 150), (0, 0, 0))
    quad = [(40.0, 30.0), (160.0, 20.0), (170.0, 130.0), (30.0, 120.0)]
    from PIL import ImageDraw

    ImageDraw.Draw(source).polygon(quad, fill=(255, 255, 255))
    out = perspective_correct(source, quad)
    width = round(max(np.hypot(120, 10), np.hypot(140, 10)))
    height = round(max(np.hypot(10, 90), np.hypot(10, 110)))
    assert out.size == (width, height)
    inner = np.asarray(out)[3:-3, 3:-3]
    assert inner.mean() > 240


@pytest.mark.parametrize("quad", [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], [(5.0, 5.0)] * 4])
def test_perspective_correct_rejects_bad_quad(quad: list[tuple[float, float]]):
    with pytest.raises(ValueError, match="quad"):
        perspective_correct(Image.new("RGB", (10, 10)), quad)


def test_crop_zoom_scales_and_clips():
    image = Image.new("RGB", (100, 50), (1, 2, 3))
    assert crop_zoom(image, (10, 10, 30, 20), scale=3).size == (60, 30)
    # Parts outside the image are clipped away before zooming.
    assert crop_zoom(image, (90, 40, 130, 70), scale=2).size == (20, 20)


def test_crop_zoom_rejects_empty_and_bad_scale():
    image = Image.new("RGB", (100, 50))
    with pytest.raises(ValueError, match="no area"):
        crop_zoom(image, (200, 200, 300, 300))
    with pytest.raises(ValueError, match="scale"):
        crop_zoom(image, (0, 0, 10, 10), scale=0)


def test_clip_bbox_orders_and_rounds_outwards():
    assert clip_bbox((10.6, 20.2, 2.3, 4.9), 100, 100) == (2, 4, 11, 21)


def test_detect_tile_seams_on_rendered_panel():
    from io import BytesIO

    from rail_vision_bench.graph.generate import generate_scene
    from rail_vision_bench.synth.render import layout_scene, render_png

    doc = layout_scene(generate_scene(4, n_inner=3), tile_px=32)
    image = Image.open(BytesIO(render_png(doc))).convert("RGB")
    tiles = detect_tile_seams(image)
    assert len(tiles) >= 10
    widths = [x1 - x0 for x0, _, x1, _ in tiles]
    heights = [y1 - y0 for _, y0, _, y1 in tiles]
    assert np.median(widths) == pytest.approx(32, abs=1.5)
    assert np.median(heights) == pytest.approx(32, abs=1.5)
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= image.width
        assert 0 <= y0 < y1 <= image.height


def test_detect_tile_seams_blank_image():
    assert detect_tile_seams(Image.new("RGB", (120, 80), (128, 128, 128))) == []


# --------------------------------------------------------------------------- quality


def test_measure_flat_and_white():
    flat = measure(Image.new("RGB", (50, 50), (100, 100, 100)))
    assert isinstance(flat, QualityMetrics)
    assert flat.blur_var == pytest.approx(0.0)
    assert flat.glare_fraction == 0.0
    white = measure(Image.new("RGB", (50, 50), (255, 255, 255)))
    assert white.glare_fraction == 1.0


def test_measure_blur_lowers_laplacian_variance():
    from PIL import ImageFilter

    checker = (np.indices((64, 64)).sum(axis=0) // 4 % 2 * 200).astype(np.uint8)
    sharp = Image.fromarray(checker).convert("RGB")
    blurred = sharp.filter(ImageFilter.GaussianBlur(3))
    assert measure(sharp).blur_var > 10 * measure(blurred).blur_var
    assert measure(sharp).blur_var > 300  # an "easy" image by the default thresholds


def test_measure_glare_fraction():
    array = np.zeros((10, 10, 3), dtype=np.uint8)
    array[:3] = 255
    assert measure(Image.fromarray(array)).glare_fraction == pytest.approx(0.3)


def test_iter_frames_imageio_fallback(tiny_video: Path, monkeypatch: pytest.MonkeyPatch):
    import av

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise av.error.InvalidDataError(1094995529, "Invalid data found when processing input")

    monkeypatch.setattr(av, "open", refuse)
    frames = list(iter_frames(str(tiny_video), fps=5))
    assert len(frames) == 10
    assert [frame.timestamp_ms for frame in frames] == [200 * i for i in range(10)]
    assert frames[0].image.mode == "RGB"
    assert frames[0].image.size == (32, 32)
