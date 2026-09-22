"""The harness service: health, metrics and validation are live; inference is not yet.

``create_app`` is also the factory ``bench serve --reload`` hands to uvicorn,
so it must work with no arguments.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from prometheus_client import make_asgi_app

from rail_vision_bench import SCHEMA_VERSION, __version__
from rail_vision_bench.config import Mode
from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.harness.metrics import REGISTRY
from rail_vision_bench.harness.models import FrameResponse, HealthResponse, ValidateResponse
from rail_vision_bench.settings import Settings, get_settings

# Documented /stream protocol (not served in the skeleton):
#   client -> server: binary frames, or {"type": "control", "action": "end"}
#   server -> client: {"type": "partial" | "final" | "error", "seq": int, "scene_id": str,
#                      "annotation"?: SceneAnnotation, "validation"?: ValidationReport,
#                      "latency_ms"?: float}
STREAM_NOT_IMPLEMENTED_REASON = "stream inference is not implemented in the skeleton"
FRAME_NOT_IMPLEMENTED_DETAIL = "frame inference is not implemented in the skeleton"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the harness application.

    Args:
        settings: Runtime settings; ``get_settings()`` when omitted.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(title="rail-vision-bench harness", version=__version__)
    app.state.settings = settings if settings is not None else get_settings()
    # The mount answers /metrics/ directly; a bare /metrics is redirected there by Starlette.
    app.mount("/metrics", make_asgi_app(registry=REGISTRY))

    @app.get("/health")
    def health() -> HealthResponse:
        """Report liveness together with the package and schema versions."""
        return HealthResponse(version=__version__, schema_version=SCHEMA_VERSION)

    @app.post("/validate")
    def validate(document: dict[str, Any], strict: bool = False) -> ValidateResponse:
        """Validate a posted scene document (schema, models, semantic rules)."""
        report = validate_document(document, strict=strict)
        return ValidateResponse(ok=report.ok, issues=report.issues)

    @app.post("/frame")
    async def frame(
        frame: Annotated[UploadFile, File()],
        model: Annotated[str, Form()],
        mode: Annotated[Mode, Form()] = Mode.SINGLE_SHOT,
    ) -> FrameResponse:
        """Run one inference over an uploaded frame (not implemented in the skeleton)."""
        raise HTTPException(status_code=501, detail=FRAME_NOT_IMPLEMENTED_DETAIL)

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        """Accept a streaming session and close it: inference is not implemented."""
        await websocket.accept()
        await websocket.close(code=1011, reason=STREAM_NOT_IMPLEMENTED_REASON)

    return app
