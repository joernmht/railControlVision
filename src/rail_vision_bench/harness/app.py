"""The harness service: health, metrics, validation and real-time inference.

``create_app`` is also the factory ``bench serve --reload`` hands to uvicorn,
so it must work with no arguments: the model catalogue is then read lazily
from ``configs/models.yaml`` relative to the working directory on the first
inference request, and providers come from the registry.
"""

from __future__ import annotations

import io
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError
from prometheus_client import make_asgi_app

from rail_vision_bench import SCHEMA_VERSION, __version__
from rail_vision_bench.agents.documents import default_source, scene_id_for
from rail_vision_bench.agents.graph import AgenticResult, run_agentic
from rail_vision_bench.agents.single_shot import SingleShotResult, single_shot_attempt
from rail_vision_bench.agents.state import AgentState
from rail_vision_bench.config import Mode, ModelCatalogue, ModelConfig, load_model_catalogue
from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.harness.metrics import (
    COST_USD,
    FRAME_LATENCY,
    FRAMES_TOTAL,
    REGISTRY,
    TOKENS,
)
from rail_vision_bench.harness.models import FrameResponse, HealthResponse, ValidateResponse
from rail_vision_bench.providers.base import ImageInput, Usage, VisionProvider
from rail_vision_bench.providers.registry import provider_for_model
from rail_vision_bench.schema.issues import IssueCode, ValidationReport
from rail_vision_bench.schema.models import SceneAnnotation
from rail_vision_bench.settings import Settings, get_settings

# /stream protocol:
#   connect:        /stream?model=<catalogue name>&mode=<single_shot|agentic>
#   client -> server: binary frames, or {"type": "control", "action": "end"}
#   server -> client: {"type": "partial" | "final" | "error", "seq": int, "scene_id": str,
#                      "annotation"?: SceneAnnotation, "validation"?: ValidationReport,
#                      "latency_ms"?: float, "detail"?: str}
DEFAULT_CATALOGUE_PATH: Final = Path("configs/models.yaml")
"""Catalogue read (relative to the working directory) when ``create_app`` gets none."""
WS_POLICY_VIOLATION: Final = 1008

ProviderFactory = Callable[[ModelConfig, Settings], VisionProvider]
"""Builds the provider of a catalogue entry; the default goes through the registry."""
FrameStatus = Literal["ok", "invalid", "error"]

_FORMATS: Final = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


class FrameError(Exception):
    """An inference request that cannot be served; carries the HTTP status and detail."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _registry_provider(model: ModelConfig, settings: Settings) -> VisionProvider:
    """Default :data:`ProviderFactory`: the registry's provider for the entry."""
    return provider_for_model(model, settings)


def image_input(data: bytes) -> ImageInput:
    """Turn uploaded bytes into an :class:`ImageInput`, detecting the format.

    PNG, JPEG and WebP are passed through; any other format Pillow decodes is
    re-encoded as PNG.

    Args:
        data: The uploaded bytes.

    Returns:
        The image with its media type.

    Raises:
        FrameError: 422 when the bytes are not a decodable image.
    """
    try:
        with Image.open(io.BytesIO(data)) as decoded:
            media_type = _FORMATS.get(decoded.format or "")
            if media_type is not None:
                decoded.verify()
                return ImageInput.model_validate({"data": data, "media_type": media_type})
            buffer = io.BytesIO()
            decoded.save(buffer, format="PNG")
            return ImageInput(data=buffer.getvalue(), media_type="image/png")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise FrameError(422, f"the frame is not a decodable image: {exc}") from exc


class InferenceService:
    """Resolves catalogue models to providers and runs one frame through a mode.

    Providers are created once per catalogue name and reused across requests.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        catalogue: ModelCatalogue | None,
        catalogue_path: Path,
        provider_factory: ProviderFactory,
        max_rounds: int,
    ) -> None:
        self.settings = settings
        self.catalogue_path = catalogue_path
        self.provider_factory = provider_factory
        self.max_rounds = max_rounds
        self._catalogue = catalogue
        self._providers: dict[str, VisionProvider] = {}

    def catalogue(self) -> ModelCatalogue:
        """Return the catalogue, loading it from ``catalogue_path`` on first use.

        Raises:
            FrameError: 503 when the catalogue file is missing or invalid.
        """
        if self._catalogue is None:
            path = self.catalogue_path
            try:
                self._catalogue = load_model_catalogue(path)
            except FileNotFoundError as exc:
                raise FrameError(503, f"model catalogue {path} not found") from exc
            except ValueError as exc:
                raise FrameError(503, f"model catalogue {path} is invalid: {exc}") from exc
        return self._catalogue

    def model(self, name: str) -> ModelConfig:
        """Look up a catalogue entry.

        Raises:
            FrameError: 404 for an unknown name, 503 when the catalogue is unavailable.
        """
        try:
            return self.catalogue().get(name)
        except KeyError as exc:
            raise FrameError(404, str(exc.args[0]) if exc.args else str(exc)) from exc

    def provider(self, model: ModelConfig) -> VisionProvider:
        """Return the (cached) provider of a catalogue entry.

        Raises:
            FrameError: 503 when the provider cannot be created.
        """
        if model.name not in self._providers:
            try:
                self._providers[model.name] = self.provider_factory(model, self.settings)
            except (KeyError, ValueError) as exc:
                raise FrameError(503, f"provider for {model.name!r} unavailable: {exc}") from exc
        return self._providers[model.name]

    async def infer(
        self,
        data: bytes,
        *,
        model: ModelConfig,
        mode: Mode,
        on_round: Callable[[AgentState], Awaitable[None]] | None = None,
    ) -> FrameResponse:
        """Run one frame through single-shot or agentic inference and record the metrics.

        Args:
            data: The encoded frame.
            model: The catalogue entry to run.
            mode: Single-shot or agentic.
            on_round: Agentic mode only: awaited with the state after every critic step.

        Returns:
            The response; ``annotation`` is None when the output did not parse,
            ``validation`` reports on the annotation (or on the raw document).

        Raises:
            FrameError: 422 for an undecodable frame, 404/503 as in :meth:`model`
                and :meth:`provider`, 502 when the provider call fails.
        """
        image = image_input(data)
        provider = self.provider(model)
        scene_id = scene_id_for(image)
        try:
            source = default_source(image, scene_id=scene_id)
            if mode is Mode.AGENTIC:
                result: SingleShotResult | AgenticResult = await run_agentic(
                    provider,
                    image,
                    model=model.model_id,
                    scene_id=scene_id,
                    max_rounds=self.max_rounds,
                    source=source,
                    on_round=on_round,
                )
            else:
                result = await single_shot_attempt(
                    provider, image, model=model.model_id, scene_id=scene_id, source=source
                )
        except Exception as exc:
            FRAMES_TOTAL.labels(model=model.name, mode=mode.value, status="error").inc()
            raise FrameError(502, f"inference failed: {type(exc).__name__}: {exc}") from exc
        validation = frame_validation(result.annotation, result.document)
        status: FrameStatus = (
            "ok"
            if result.annotation is not None and validation is not None and validation.ok
            else "invalid"
        )
        record_metrics(model, mode, status=status, usage=result.usage, latency_ms=result.latency_ms)
        return FrameResponse(
            scene_id=scene_id,
            model=model.name,
            mode=mode,
            annotation=result.annotation,
            validation=validation,
            latency_ms=result.latency_ms,
        )


def frame_validation(
    annotation: SceneAnnotation | None, document: dict[str, Any] | None
) -> ValidationReport | None:
    """Validate the parsed annotation, or the raw document when it did not parse.

    Args:
        annotation: The parsed ``SceneAnnotation`` or None.
        document: The raw JSON document or None.

    Returns:
        The report, or None when the model produced no JSON document at all.
    """
    if annotation is not None:
        return validate_document(annotation)
    if document is not None:
        return validate_document(document)
    return None


def record_metrics(
    model: ModelConfig, mode: Mode, *, status: FrameStatus, usage: Usage, latency_ms: float
) -> None:
    """Update the Prometheus metrics after one frame.

    The cost is the provider's when it reports one, else it is computed from
    the catalogue prices (per 1k tokens) when those are set.

    Args:
        model: The catalogue entry.
        mode: The inference mode.
        status: ``ok``, ``invalid`` or ``error``.
        usage: Token counts and cost of the frame.
        latency_ms: Wall-clock inference time.
    """
    FRAME_LATENCY.labels(model=model.name, mode=mode.value).observe(latency_ms / 1000.0)
    FRAMES_TOTAL.labels(model=model.name, mode=mode.value, status=status).inc()
    TOKENS.labels(model=model.name, direction="in").inc(usage.input_tokens)
    TOKENS.labels(model=model.name, direction="out").inc(usage.output_tokens)
    cost = usage.cost_usd
    if cost is None and model.cost_per_1k_in is not None and model.cost_per_1k_out is not None:
        cost = (
            usage.input_tokens * model.cost_per_1k_in + usage.output_tokens * model.cost_per_1k_out
        ) / 1000.0
    if cost:
        COST_USD.labels(model=model.name).inc(cost)


def create_app(
    settings: Settings | None = None,
    *,
    catalogue: ModelCatalogue | None = None,
    catalogue_path: Path | None = None,
    provider_factory: ProviderFactory | None = None,
    max_rounds: int = 3,
) -> FastAPI:
    """Build the harness application.

    Args:
        settings: Runtime settings; ``get_settings()`` when omitted.
        catalogue: The models ``/frame`` and ``/stream`` may run; loaded
            lazily from ``catalogue_path`` when omitted.
        catalogue_path: Catalogue file, default ``configs/models.yaml``
            relative to the working directory at the first inference request.
        provider_factory: Builds the provider of a catalogue entry; the
            registry (``provider_for_model(entry, settings)``) when omitted.
        max_rounds: Round budget of agentic inference.

    Returns:
        The configured FastAPI application.
    """
    app = FastAPI(title="rail-vision-bench harness", version=__version__)
    app.state.settings = settings if settings is not None else get_settings()
    service = InferenceService(
        settings=app.state.settings,
        catalogue=catalogue,
        catalogue_path=catalogue_path if catalogue_path is not None else DEFAULT_CATALOGUE_PATH,
        provider_factory=provider_factory if provider_factory is not None else _registry_provider,
        max_rounds=max_rounds,
    )
    app.state.inference = service
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
        """Run one inference over an uploaded frame with a catalogue model."""
        try:
            config = service.model(model)
            return await service.infer(await frame.read(), model=config, mode=mode)
        except FrameError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        """Run inference on every binary frame of a session until the client ends it."""
        await websocket.accept()
        try:
            mode = Mode(websocket.query_params.get("mode", Mode.SINGLE_SHOT.value))
            config = service.model(websocket.query_params.get("model", ""))
        except (ValueError, FrameError) as exc:
            detail = exc.detail if isinstance(exc, FrameError) else f"invalid mode: {exc}"
            await websocket.send_json({"type": "error", "seq": 0, "scene_id": "", "detail": detail})
            await websocket.close(code=WS_POLICY_VIOLATION, reason=detail[:120])
            return
        seq = 0
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is None:
                    if _is_end(message.get("text")):
                        await websocket.close(code=1000)
                        return
                    await websocket.send_json(
                        {
                            "type": "error",
                            "seq": seq,
                            "scene_id": "",
                            "detail": 'expected a binary frame or {"type": "control", '
                            '"action": "end"}',
                        }
                    )
                    seq += 1
                    continue
                await _stream_frame(websocket, service, data, seq=seq, model=config, mode=mode)
                seq += 1
        except WebSocketDisconnect:
            return

    return app


def _is_end(text: str | None) -> bool:
    """Whether a text message is the control message that ends the stream."""
    if text is None:
        return False
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(message, dict)
        and message.get("type") == "control"
        and message.get("action") == "end"
    )


async def _stream_frame(
    websocket: WebSocket,
    service: InferenceService,
    data: bytes,
    *,
    seq: int,
    model: ModelConfig,
    mode: Mode,
) -> None:
    """Infer one streamed frame and send its partial, final or error messages."""

    async def send_partial(state: AgentState) -> None:
        candidate = state.get("candidate")
        message: dict[str, Any] = {
            "type": "partial",
            "seq": seq,
            "scene_id": state.get("scene_id", ""),
            "round": state.get("round", 0),
        }
        if candidate is not None:
            report = frame_validation(None, candidate)
            message["validation"] = report.model_dump(mode="json") if report else None
            if report is not None and not any(
                issue.code is IssueCode.SCHEMA_INVALID for issue in report.issues
            ):
                message["annotation"] = candidate
        await websocket.send_json(message)

    try:
        response = await service.infer(
            data,
            model=model,
            mode=mode,
            on_round=send_partial if mode is Mode.AGENTIC else None,
        )
    except FrameError as exc:
        await websocket.send_json(
            {"type": "error", "seq": seq, "scene_id": "", "detail": exc.detail}
        )
        return
    final: dict[str, Any] = {
        "type": "final",
        "seq": seq,
        "scene_id": response.scene_id,
        "latency_ms": response.latency_ms,
    }
    if response.annotation is not None:
        final["annotation"] = response.annotation.model_dump(mode="json", exclude_none=True)
    if response.validation is not None:
        final["validation"] = response.validation.model_dump(mode="json")
    await websocket.send_json(final)
