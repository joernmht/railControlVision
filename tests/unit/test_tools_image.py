"""The image tools: crop, read (against a fake provider) and render."""

from __future__ import annotations

from io import BytesIO
from typing import Any, ClassVar

import pytest
from PIL import Image
from pydantic import ValidationError

from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.tools.tools import (
    READ_RESPONSE_SCHEMA,
    ToolResult,
    crop_tool,
    read_tool,
    render_tool,
)


def _encode(image: Image.Image, fmt: str) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


PNG = _encode(Image.new("RGB", (40, 20), (0, 90, 0)), "PNG")
JPEG = _encode(Image.new("RGB", (40, 20), (0, 90, 0)), "JPEG")


class FakeProvider:
    """Records requests and answers with a fixed text (and optional parsed object)."""

    name: ClassVar[str] = "fake"

    def __init__(self, text: str, parsed: dict[str, Any] | None = None) -> None:
        self.text = text
        self.parsed = parsed
        self.requests: list[VisionRequest] = []

    async def complete(self, request: VisionRequest) -> VisionResponse:
        self.requests.append(request)
        return VisionResponse(
            text=self.text,
            parsed=self.parsed,
            usage=Usage(input_tokens=120, output_tokens=8, cost_usd=0.0004),
            latency_ms=12.0,
        )


# --------------------------------------------------------------------------- crop


def test_crop_tool_returns_zoomed_png():
    image = Image.new("RGB", (100, 60), (255, 255, 255))
    image.paste((255, 0, 0), (10, 10, 30, 20))
    out = crop_tool(_encode(image, "PNG"), (10, 10, 30, 20), zoom=3)
    assert out.startswith(b"\x89PNG")
    with Image.open(BytesIO(out)) as crop:
        assert crop.size == (60, 30)
        assert crop.convert("RGB").getpixel((30, 15)) == (255, 0, 0)


def test_crop_tool_clips_to_image_and_reads_jpeg():
    out = crop_tool(JPEG, (30, 10, 80, 60))
    with Image.open(BytesIO(out)) as crop:
        assert crop.size == (20, 20)


def test_crop_tool_rejects_bad_input():
    with pytest.raises(ValueError, match="decodable"):
        crop_tool(b"not an image", (0, 0, 1, 1))
    with pytest.raises(ValueError, match="zoom"):
        crop_tool(PNG, (0, 0, 10, 10), zoom=0)
    with pytest.raises(ValueError, match="no area"):
        crop_tool(PNG, (50, 50, 60, 60))


# --------------------------------------------------------------------------- read


async def test_read_tool_parses_json_answer():
    provider = FakeProvider('{"text": "W12", "confidence": 0.9}')
    result = await read_tool(PNG, "Which label is shown?", provider=provider, model="m-1")
    assert isinstance(result, ToolResult)
    assert result.ok
    assert result.errors == []
    assert result.payload == {
        "text": "W12",
        "confidence": 0.9,
        "usage": {"input_tokens": 120, "output_tokens": 8, "cost_usd": 0.0004},
    }
    (request,) = provider.requests
    assert request.model == "m-1"
    assert "Which label is shown?" in request.prompt
    assert '"confidence"' in request.prompt
    assert request.response_schema == READ_RESPONSE_SCHEMA
    assert request.images[0].media_type == "image/png"
    assert request.images[0].data == PNG


async def test_read_tool_detects_jpeg_and_reencodes_other_formats():
    provider = FakeProvider('{"text": "+", "confidence": 1}')
    await read_tool(JPEG, "Sign?", provider=provider, model="m")
    bmp = _encode(Image.new("RGB", (8, 8)), "BMP")
    await read_tool(bmp, "Sign?", provider=provider, model="m")
    assert provider.requests[0].images[0].media_type == "image/jpeg"
    assert provider.requests[1].images[0].media_type == "image/png"
    assert provider.requests[1].images[0].data.startswith(b"\x89PNG")


async def test_read_tool_accepts_fenced_json():
    provider = FakeProvider('Sure:\n```json\n{"text": "Hp0", "confidence": 0.6}\n```')
    result = await read_tool(PNG, "Aspect?", provider=provider, model="m")
    assert result.payload["text"] == "Hp0"
    assert result.payload["confidence"] == 0.6


async def test_read_tool_prefers_provider_parsed_object():
    provider = FakeProvider("ignored", parsed={"text": "N1", "confidence": 0.3})
    result = await read_tool(PNG, "Label?", provider=provider, model="m")
    assert result.payload["text"] == "N1"
    assert result.payload["confidence"] == 0.3


async def test_read_tool_falls_back_to_raw_text():
    provider = FakeProvider("  The label reads 12a.  ")
    result = await read_tool(PNG, "Label?", provider=provider, model="m")
    assert result.ok
    assert result.payload["text"] == "The label reads 12a."
    assert result.payload["confidence"] is None


@pytest.mark.parametrize(
    ("answer", "text", "confidence"),
    [
        ('{"text": "7", "confidence": 1.5}', "7", None),
        ('{"text": "7", "confidence": "high"}', "7", None),
        ('{"text": "7", "confidence": true}', "7", None),
        ('{"text": 7, "confidence": 0.5}', "7", 0.5),
        ('{"answer": "7"}', '{"answer": "7"}', None),
    ],
)
async def test_read_tool_normalises_odd_answers(answer: str, text: str, confidence: float | None):
    result = await read_tool(PNG, "Digit?", provider=FakeProvider(answer), model="m")
    assert result.payload["text"] == text
    assert result.payload["confidence"] == confidence


async def test_read_tool_reports_bad_input_without_calling_provider():
    provider = FakeProvider("{}")
    undecodable = await read_tool(b"garbage", "Label?", provider=provider, model="m")
    empty = await read_tool(PNG, "   ", provider=provider, model="m")
    assert not undecodable.ok
    assert undecodable.errors
    assert not empty.ok
    assert provider.requests == []


async def test_read_tool_propagates_provider_errors():
    class Broken(FakeProvider):
        async def complete(self, request: VisionRequest) -> VisionResponse:
            raise NotImplementedError("stub provider")

    with pytest.raises(NotImplementedError):
        await read_tool(PNG, "Label?", provider=Broken(""), model="m")


# --------------------------------------------------------------------------- render


def test_render_tool_draws_minimal_example(minimal_doc: dict[str, Any]):
    png = render_tool(minimal_doc)
    with Image.open(BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.size == (400, 100)


def test_render_tool_estw_style(station_dkw_doc: dict[str, Any]):
    with Image.open(BytesIO(render_tool(station_dkw_doc, style="estw"))) as image:
        assert image.size == (800, 400)
        assert image.convert("RGB").getpixel((1, 1)) == (0, 0, 0)


def test_render_tool_draws_documents_without_geometry(minimal_doc: dict[str, Any]):
    for node in minimal_doc["topology"]["nodes"]:
        node.pop("geometry", None)
    for edge in minimal_doc["topology"]["edges"]:
        edge.pop("geometry", None)
    with Image.open(BytesIO(render_tool(minimal_doc))) as image:
        assert image.size == (400, 100)


def test_render_tool_rejects_malformed_document():
    with pytest.raises(ValidationError):
        render_tool({"schema_version": "v0"})
