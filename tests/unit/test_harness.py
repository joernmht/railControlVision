"""The harness over Starlette's TestClient and the async client over ASGITransport."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from rail_vision_bench.harness.app import create_app
from rail_vision_bench.harness.client import HarnessClient, HarnessError
from rail_vision_bench.settings import Settings
from tests.conftest import FIXTURES_DIR, load_json


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


def test_frame_is_not_implemented(client: TestClient):
    response = client.post(
        "/frame",
        files={"frame": ("f.png", b"\x89PNG", "image/png")},
        data={"model": "m"},
    )
    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"]


def test_stream_closes_with_1011(client: TestClient):
    with client.websocket_connect("/stream") as ws, pytest.raises(WebSocketDisconnect) as exc:
        ws.receive_text()
    assert exc.value.code == 1011


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
        assert exc.value.status_code == 501
        assert "not implemented" in exc.value.detail
        assert "501" in str(exc.value)


async def test_client_aclose(app: FastAPI):
    harness = HarnessClient("http://test", transport=httpx.ASGITransport(app=app))
    assert (await harness.health()).status == "ok"
    await harness.aclose()
