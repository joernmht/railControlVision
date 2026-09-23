"""Async client of the harness, usable against a live server or an in-process ASGI app."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

import httpx

from rail_vision_bench.config import Mode
from rail_vision_bench.harness.models import FrameResponse, HealthResponse, ValidateResponse


class HarnessError(Exception):
    """A non-2xx answer of the harness."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"harness answered {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _raise_for_status(response: httpx.Response) -> None:
    """Turn a non-2xx response into a :class:`HarnessError`.

    The detail is FastAPI's JSON ``detail`` field when present, else the body text.

    Args:
        response: The response to check.

    Raises:
        HarnessError: If the status is not 2xx.
    """
    if response.is_success:
        return
    detail = response.text
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and "detail" in body:
        detail = str(body["detail"])
    raise HarnessError(response.status_code, detail)


class HarnessClient:
    """Thin ``httpx.AsyncClient`` wrapper over the harness endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        # An explicit transport (httpx.ASGITransport) lets tests talk to the app in-process.
        self._client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout)

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def health(self) -> HealthResponse:
        """Call ``GET /health``.

        Returns:
            The health report.

        Raises:
            HarnessError: On a non-2xx answer.
        """
        response = await self._client.get("/health")
        _raise_for_status(response)
        return HealthResponse.model_validate(response.json())

    async def validate(
        self, document: Mapping[str, Any], *, strict: bool = False
    ) -> ValidateResponse:
        """Call ``POST /validate`` with a scene document.

        Args:
            document: The raw scene document (JSON-compatible mapping).
            strict: Promote the strict-mode warnings to errors.

        Returns:
            The validation outcome.

        Raises:
            HarnessError: On a non-2xx answer.
        """
        response = await self._client.post(
            "/validate", json=dict(document), params={"strict": strict}
        )
        _raise_for_status(response)
        return ValidateResponse.model_validate(response.json())

    async def post_frame(
        self,
        image: bytes,
        *,
        model: str,
        mode: Mode = Mode.SINGLE_SHOT,
        media_type: str = "image/png",
    ) -> FrameResponse:
        """Call ``POST /frame`` with one encoded image.

        Args:
            image: The encoded image bytes.
            model: Catalogue name of the model to run.
            mode: Single-shot or agentic inference.
            media_type: MIME type of ``image``.

        Returns:
            The inference result.

        Raises:
            HarnessError: On a non-2xx answer: 404 for an unknown model, 422 for an
                undecodable frame or invalid mode, 502 when the provider call fails,
                503 when the catalogue or provider is unavailable.
        """
        response = await self._client.post(
            "/frame",
            files={"frame": ("frame", image, media_type)},
            data={"model": model, "mode": mode.value},
        )
        _raise_for_status(response)
        return FrameResponse.model_validate(response.json())
