"""The harness over Starlette's TestClient and the async client over ASGITransport."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from rail_vision_bench.config import Mode, ModelCatalogue, ModelConfig
from rail_vision_bench.harness import app as harness_app
from rail_vision_bench.harness.app import create_app
from rail_vision_bench.harness.client import HarnessClient, HarnessError
from rail_vision_bench.harness.metrics import REGISTRY
from rail_vision_bench.providers.base import Usage, VisionRequest, VisionResponse
from rail_vision_bench.settings import Settings
from tests.conftest import EXAMPLES_DIR, FIXTURES_DIR, load_json

MODEL = "fake-model"
CATALOGUE = ModelCatalogue(
    models=[
        ModelConfig(
            name=MODEL,
            provider="anthropic",
            model_id="fake-1",
            cost_per_1k_in=1.0,
            cost_per_1k_out=2.0,
        )
    ]
)


def png(width: int = 400, height: int = 100) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "black").save(buffer, format="PNG")
    return buffer.getvalue()


class DocumentProvider:
    """Answers every request with the minimal example (or fails, or talks nonsense)."""

    name: ClassVar[str] = "document"

    def __init__(self, answer: str | None = None, *, fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.requests: list[VisionRequest] = []

    async def complete(self, request: VisionRequest) -> VisionResponse:
        self.requests.append(request)
        if self.fail:
            msg = "backend down"
            raise RuntimeError(msg)
        if self.answer is not None:
            text = self.answer
        elif request.system is not None and request.system.startswith("# agentic/critic"):
            text = json.dumps({"matches": True, "critique": []})
        else:
            text = json.dumps(load_json(EXAMPLES_DIR / "minimal.json"))
        return VisionResponse(text=text, usage=Usage(input_tokens=100, output_tokens=50))


def fake_app(provider: DocumentProvider) -> FastAPI:
    return create_app(
        Settings(_env_file=None),
        catalogue=CATALOGUE,
        provider_factory=lambda model, settings: provider,
    )


def sample(name: str, labels: dict[str, str]) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


@pytest.fixture
def app() -> FastAPI:
    return create_app(Settings(_env_file=None))


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_create_app_without_settings_uses_get_settings():
    app = create_app()
    assert isinstance(app.state.settings, Settings)


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["schema_version"] == "v0"
    assert body["version"]


def test_metrics_exposes_registry(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rvb_frame_latency_seconds" in response.text
    assert "rvb_frames_total" in response.text


def test_validate_ok(client: TestClient, minimal_doc: dict[str, Any]):
    response = client.post("/validate", json=minimal_doc)
    assert response.status_code == 200
    assert response.json() == {"ok": True, "issues": []}


def test_validate_reports_issues(client: TestClient):
    doc = load_json(FIXTURES_DIR / "invalid" / "dkw_bad_path.json")
    response = client.post("/validate", json=doc)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "STATE_PATH_NOT_ALLOWED" in {issue["code"] for issue in body["issues"]}


def test_validate_strict_query(client: TestClient):
    doc = load_json(FIXTURES_DIR / "strict" / "state_missing.json")
    relaxed = client.post("/validate", json=doc).json()
    strict = client.post("/validate", params={"strict": "true"}, json=doc).json()
    assert relaxed["ok"] is True
    assert strict["ok"] is False
    assert {issue["code"] for issue in strict["issues"]} == {"STATE_MISSING"}


def test_validate_rejects_non_object_body(client: TestClient):
    response = client.post("/validate", json=[1, 2, 3])
    assert response.status_code == 422


def test_frame_single_shot():
    provider = DocumentProvider()
    client = TestClient(fake_app(provider))
    labels = {"model": MODEL, "mode": "single_shot", "status": "ok"}
    before = sample("rvb_frames_total", labels)
    tokens_before = sample("rvb_tokens_total", {"model": MODEL, "direction": "in"})
    cost_before = sample("rvb_cost_usd_total", {"model": MODEL})
    response = client.post(
        "/frame", files={"frame": ("f.png", png(), "image/png")}, data={"model": MODEL}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == MODEL
    assert body["mode"] == "single_shot"
    assert body["validation"]["ok"] is True
    assert body["annotation"]["scene_id"] == body["scene_id"]
    assert body["annotation"]["provenance"]["model_id"] == "fake-1"
    assert body["annotation"]["source"]["width"] == 400
    assert body["latency_ms"] >= 0
    assert provider.requests[0].model == "fake-1"
    assert sample("rvb_frames_total", labels) == before + 1
    assert sample("rvb_tokens_total", {"model": MODEL, "direction": "in"}) == tokens_before + 100
    # no provider cost: computed from the catalogue prices (100 * 1.0 + 50 * 2.0) / 1000
    assert sample("rvb_cost_usd_total", {"model": MODEL}) == pytest.approx(cost_before + 0.2)
    assert "rvb_frame_latency_seconds_count" in client.get("/metrics").text


def test_frame_agentic(monkeypatch: pytest.MonkeyPatch):
    from rail_vision_bench.agents import nodes

    monkeypatch.setattr(nodes, "render_tool", lambda document: png(40, 10))
    provider = DocumentProvider()
    client = TestClient(fake_app(provider))
    response = client.post(
        "/frame",
        files={"frame": ("f.png", png(), "image/png")},
        data={"model": MODEL, "mode": "agentic"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "agentic"
    assert body["validation"]["ok"] is True
    assert [node["id"] for node in body["annotation"]["topology"]["nodes"]] == ["n_w", "n_j", "n_e"]


def test_frame_unparseable_output_is_invalid():
    client = TestClient(fake_app(DocumentProvider("no JSON here")))
    labels = {"model": MODEL, "mode": "single_shot", "status": "invalid"}
    before = sample("rvb_frames_total", labels)
    response = client.post(
        "/frame", files={"frame": ("f.png", png(), "image/png")}, data={"model": MODEL}
    )
    assert response.status_code == 200
    assert response.json()["annotation"] is None
    assert response.json()["validation"] is None
    assert sample("rvb_frames_total", labels) == before + 1


def test_frame_errors():
    client = TestClient(fake_app(DocumentProvider()))
    unknown = client.post(
        "/frame", files={"frame": ("f.png", png(), "image/png")}, data={"model": "nope"}
    )
    assert unknown.status_code == 404
    assert "unknown model" in unknown.json()["detail"]
    garbage = client.post(
        "/frame", files={"frame": ("f.png", b"\x89PNG", "image/png")}, data={"model": MODEL}
    )
    assert garbage.status_code == 422
    bad_mode = client.post(
        "/frame",
        files={"frame": ("f.png", png(), "image/png")},
        data={"model": MODEL, "mode": "psychic"},
    )
    assert bad_mode.status_code == 422

    failing = TestClient(fake_app(DocumentProvider(fail=True)))
    labels = {"model": MODEL, "mode": "single_shot", "status": "error"}
    before = sample("rvb_frames_total", labels)
    response = failing.post(
        "/frame", files={"frame": ("f.png", png(), "image/png")}, data={"model": MODEL}
    )
    assert response.status_code == 502
    assert "backend down" in response.json()["detail"]
    assert sample("rvb_frames_total", labels) == before + 1


def test_frame_converts_other_formats():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buffer, format="BMP")
    provider = DocumentProvider()
    client = TestClient(fake_app(provider))
    response = client.post(
        "/frame",
        files={"frame": ("f.bmp", buffer.getvalue(), "image/bmp")},
        data={"model": MODEL},
    )
    assert response.status_code == 200
    assert provider.requests[0].images[0].media_type == "image/png"


def test_frame_catalogue_loaded_lazily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(Settings(_env_file=None)))
    assert client.get("/health").status_code == 200  # no catalogue needed
    response = client.post(
        "/frame", files={"frame": ("f.png", png(), "image/png")}, data={"model": MODEL}
    )
    assert response.status_code == 503
    assert "configs/models.yaml" in response.json()["detail"]


def test_default_catalogue_and_registry(monkeypatch: pytest.MonkeyPatch):
    provider = DocumentProvider()
    monkeypatch.setattr(harness_app, "get_provider", lambda name, settings: provider)
    client = TestClient(create_app(Settings(_env_file=None)))
    response = client.post(
        "/frame", files={"frame": ("f.png", png(), "image/png")}, data={"model": "gpt-4o"}
    )
    assert response.status_code == 200, response.text
    assert provider.requests[0].model == "gpt-4o"


def test_stream_single_shot():
    client = TestClient(fake_app(DocumentProvider()))
    with client.websocket_connect(f"/stream?model={MODEL}") as ws:
        ws.send_bytes(png())
        first = ws.receive_json()
        ws.send_bytes(b"not an image")
        second = ws.receive_json()
        ws.send_text("hello")
        third = ws.receive_json()
        ws.send_json({"type": "control", "action": "end"})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert first["type"] == "final"
    assert first["seq"] == 0
    assert first["validation"]["ok"] is True
    assert first["annotation"]["scene_id"] == first["scene_id"]
    assert first["latency_ms"] >= 0
    assert second["type"] == "error"
    assert second["seq"] == 1
    assert "decodable" in second["detail"]
    assert third["type"] == "error"
    assert third["seq"] == 2
    assert exc.value.code == 1000


def test_stream_agentic_sends_partials(monkeypatch: pytest.MonkeyPatch):
    from rail_vision_bench.agents import nodes

    monkeypatch.setattr(nodes, "render_tool", lambda document: png(40, 10))
    client = TestClient(fake_app(DocumentProvider()))
    with client.websocket_connect(f"/stream?model={MODEL}&mode=agentic") as ws:
        ws.send_bytes(png())
        partial = ws.receive_json()
        final = ws.receive_json()
        ws.send_json({"type": "control", "action": "end"})
    assert partial["type"] == "partial"
    assert partial["round"] == 1
    assert partial["validation"]["ok"] is True
    assert partial["annotation"]["scene_id"] == partial["scene_id"]
    assert final["type"] == "final"
    assert final["scene_id"] == partial["scene_id"]


@pytest.mark.parametrize(
    ("query", "fragment"), [("?model=nope", "unknown model"), (f"?model={MODEL}&mode=x", "mode")]
)
def test_stream_rejects_bad_parameters(query: str, fragment: str):
    client = TestClient(fake_app(DocumentProvider()))
    with client.websocket_connect(f"/stream{query}") as ws:
        message = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert message["type"] == "error"
    assert fragment in message["detail"]
    assert exc.value.code == 1008


async def test_client_over_asgi_transport(app: FastAPI, minimal_doc: dict[str, Any]):
    transport = httpx.ASGITransport(app=app)
    async with HarnessClient("http://test", transport=transport) as harness:
        health = await harness.health()
        assert health.status == "ok"
        assert health.schema_version == "v0"

        report = await harness.validate(minimal_doc)
        assert report.ok
        assert report.issues == []

        strict_doc = load_json(FIXTURES_DIR / "strict" / "state_missing.json")
        assert (await harness.validate(strict_doc)).ok
        assert not (await harness.validate(strict_doc, strict=True)).ok

        with pytest.raises(HarnessError) as exc:
            await harness.post_frame(b"\x89PNG", model="m")
        assert exc.value.status_code == 404  # "m" is not in configs/models.yaml
        assert "unknown model" in exc.value.detail
        assert "404" in str(exc.value)


async def test_client_post_frame(minimal_doc: dict[str, Any]):
    transport = httpx.ASGITransport(app=fake_app(DocumentProvider()))
    async with HarnessClient("http://test", transport=transport) as harness:
        response = await harness.post_frame(png(), model=MODEL, mode=Mode.SINGLE_SHOT)
        assert response.annotation is not None
        assert response.validation is not None
        assert response.validation.ok
        assert response.model == MODEL
        with pytest.raises(HarnessError) as exc:
            await harness.post_frame(png(), model="nope")
        assert exc.value.status_code == 404


async def test_client_aclose(app: FastAPI):
    harness = HarnessClient("http://test", transport=httpx.ASGITransport(app=app))
    assert (await harness.health()).status == "ok"
    await harness.aclose()
