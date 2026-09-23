"""Helpers shared by the provider implementations (never imported by the registry).

Image encoding, credential checks, the tenacity retry loop, latency timing and
the final ``text -> parsed`` step live here so that the six providers differ
only in how they speak to their SDK.
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final, TypeVar

import httpx
from pydantic import SecretStr
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from rail_vision_bench.providers.base import ImageInput
from rail_vision_bench.providers.parsing import extract_json

T = TypeVar("T")

RETRY_ATTEMPTS: Final[int] = 4
"""Total attempts per completion (the first call plus three retries)."""

RETRY_WAIT: wait_base = wait_exponential_jitter(initial=1.0, max=20.0, jitter=1.0)
"""Back-off between attempts; tests replace it with ``tenacity.wait_none()``."""

TRANSIENT_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429})
"""Client-side HTTP statuses worth retrying; every 5xx is retried as well."""

SCHEMA_INSTRUCTION: Final[str] = (
    "Answer with a single JSON object that conforms to this JSON Schema:\n{schema}"
)
"""Appended to the prompt by providers whose API cannot take the schema itself."""

_SCHEMA_NAME_INVALID = re.compile(r"[^A-Za-z0-9_-]")


def is_transient_status(status: object) -> bool:
    """Whether an HTTP status code marks a retryable failure.

    Args:
        status: The status code carried by an SDK exception (anything non-int is not transient).

    Returns:
        True for 408, 409, 425, 429 and every 5xx status.
    """
    if isinstance(status, bool) or not isinstance(status, int):
        return False
    return status in TRANSIENT_STATUS or status >= 500


def is_transport_error(exc: BaseException) -> bool:
    """Whether an exception is a network-level failure (connection, timeout).

    Args:
        exc: The exception raised by an SDK call.

    Returns:
        True for httpx transport errors and the builtin connection/timeout errors.
    """
    return isinstance(exc, httpx.TransportError | ConnectionError | TimeoutError)


def require_secret(secret: SecretStr | None, env_key: str, provider: str) -> str:
    """Return a credential's value or fail with a message naming its env key.

    Args:
        secret: The settings field holding the credential.
        env_key: The environment key that sets it (for the error message).
        provider: The provider name (for the error message).

    Returns:
        The non-empty secret value.

    Raises:
        ValueError: If the credential is unset or empty.
    """
    value = secret.get_secret_value() if secret is not None else ""
    if not value:
        msg = f"the {provider} provider needs an API key: set {env_key} in the environment or .env"
        raise ValueError(msg)
    return value


def b64(image: ImageInput) -> str:
    """Base64-encode an image's bytes as ASCII text.

    Args:
        image: The image to encode.

    Returns:
        The standard base64 encoding of ``image.data``.
    """
    return base64.b64encode(image.data).decode("ascii")


def data_url(image: ImageInput) -> str:
    """Encode an image as a ``data:`` URL (the OpenAI-style ``image_url`` form).

    Args:
        image: The image to encode.

    Returns:
        ``data:<media type>;base64,<data>``.
    """
    return f"data:{image.media_type};base64,{b64(image)}"


def with_schema_instruction(prompt: str, schema: dict[str, Any] | None) -> str:
    """Append the JSON Schema to a prompt, for APIs that only offer a JSON mode.

    Args:
        prompt: The request prompt.
        schema: The response schema, or None to leave the prompt unchanged.

    Returns:
        The prompt, followed by :data:`SCHEMA_INSTRUCTION` when a schema is given.
    """
    if schema is None:
        return prompt
    compact = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    return f"{prompt}\n\n{SCHEMA_INSTRUCTION.format(schema=compact)}"


def schema_name(schema: dict[str, Any]) -> str:
    """Name a schema for an OpenAI-style ``json_schema`` response format.

    Args:
        schema: The response schema.

    Returns:
        The schema's ``title`` reduced to ``[A-Za-z0-9_-]`` and 64 characters, or
        ``response`` when it has no usable title.
    """
    title = schema.get("title")
    name = _SCHEMA_NAME_INVALID.sub("_", title)[:64] if isinstance(title, str) else ""
    return name or "response"


def parse_answer(text: str, structured: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Pick the parsed JSON object of an answer.

    Args:
        text: The answer text.
        structured: An object the API already returned in structured form (tool input).

    Returns:
        ``structured`` when given, otherwise :func:`extract_json` of ``text``.
    """
    if structured is not None:
        return structured
    return extract_json(text)


def dump_raw(response: object) -> Any:
    """Turn an SDK response into a JSON-serialisable value for ``VisionResponse.raw``.

    Args:
        response: The SDK's response object (pydantic model or dict).

    Returns:
        A JSON-compatible dump, or None when the object cannot be dumped.
    """
    dump = getattr(response, "model_dump", None)
    value: Any = response
    if callable(dump):
        try:
            value = dump(mode="json")
        except Exception:  # an SDK object that is not a pydantic v2 model
            return None
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None
    return value


async def call_with_retry(
    call: Callable[[], Awaitable[T]], is_transient: Callable[[BaseException], bool]
) -> tuple[T, float]:
    """Await ``call`` with exponential back-off on transient errors.

    Args:
        call: Zero-argument factory of the SDK coroutine (a fresh one per attempt).
        is_transient: Predicate deciding whether an exception is worth another attempt.

    Returns:
        The call's result and the latency of the successful attempt in milliseconds.

    Raises:
        BaseException: The last error once the attempts are exhausted, or the first
            non-transient error unchanged.
    """
    retrying = AsyncRetrying(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT,
        retry=retry_if_exception(is_transient),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            start = time.perf_counter()
            result = await call()
            return result, (time.perf_counter() - start) * 1000.0
    msg = "unreachable: tenacity either returns a result or re-raises"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover
