"""Provider ``complete()`` implementations against mocked SDK transports (no network).

The Anthropic, OpenAI, Gemini, Mistral and Ollama providers get their real SDK
client wired to an ``httpx.MockTransport``, so the tests assert the wire payload
each SDK actually sends; LiteLLM is exercised by replacing ``litellm.acompletion``.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import anthropic
import httpx
import httpx2
import litellm
import ollama
import openai
import pytest
from google import genai
from google.genai import types as genai_types
from litellm import exceptions as litellm_errors
from mistralai.client import Mistral
from pydantic import SecretStr
from tenacity import wait_none

from rail_vision_bench.providers import _common
from rail_vision_bench.providers.base import ImageInput, VisionRequest, VisionResponse
from rail_vision_bench.providers.claude import RESULT_TOOL, AnthropicProvider
from rail_vision_bench.providers.gemini import GeminiProvider
from rail_vision_bench.providers.litellm_router import LiteLLMProvider
from rail_vision_bench.providers.mistral import MistralProvider
from rail_vision_bench.providers.ollama_local import OllamaProvider
from rail_vision_bench.providers.openai_compat import OpenAICompatProvider
from rail_vision_bench.settings import Settings

PNG = b"\x89PNG\r\n\x1a\nfake-png"
JPEG = b"\xff\xd8\xff\xe0fake-jpeg"
PNG_B64 = base64.b64encode(PNG).decode()
JPEG_B64 = base64.b64encode(JPEG).decode()
SCHEMA: dict[str, Any] = {
    "title": "Scene Annotation",
    "type": "object",
    "properties": {"schema_version": {"const": "v0"}},
    "required": ["schema_version"],
}
ANSWER = {"schema_version": "v0", "tracks": []}
FENCED = "Here you go:\n```json\n" + json.dumps(ANSWER) + "\n```"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries run without sleeping."""
    monkeypatch.setattr(_common, "RETRY_WAIT", wait_none())


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def _request(*, schema: dict[str, Any] | None = None, system: str | None = "sys") -> VisionRequest:
    return VisionRequest(
        model="test-model",
        prompt="Describe the panel.",
        images=[ImageInput(data=PNG), ImageInput(data=JPEG, media_type="image/jpeg")],
        system=system,
        response_schema=schema,
        max_tokens=321,
        temperature=0.25,
    )


class Recorder:
    """An httpx mock transport that replays queued responses and records requests."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    @property
    def body(self) -> dict[str, Any]:
        """JSON body of the last request."""
        loaded = json.loads(self.requests[-1].content)
        assert isinstance(loaded, dict)
        return loaded

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def client2(self) -> httpx2.AsyncClient:
        """The Anthropic SDK is built on ``httpx2`` and rejects plain ``httpx`` clients."""
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))


def _assert_response_roundtrips(response: VisionResponse) -> None:
    """The response (including ``raw``) survives a JSON round-trip through pydantic."""
    assert VisionResponse.model_validate_json(response.model_dump_json()) == response


# --------------------------------------------------------------------------- Anthropic


def _anthropic_message(content: list[dict[str, Any]]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": content,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 340},
        },
    )


def _anthropic(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> AnthropicProvider:
    provider = AnthropicProvider(_settings())
    client = anthropic.AsyncAnthropic(api_key="k", max_retries=0, http_client=recorder.client2())
    monkeypatch.setattr(provider, "_client", client)
    return provider


async def test_anthropic_forces_schema_tool(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(
        _anthropic_message(
            [{"type": "tool_use", "id": "tu_1", "name": RESULT_TOOL, "input": ANSWER}]
        )
    )
    response = await _anthropic(monkeypatch, recorder).complete(_request(schema=SCHEMA))

    body = recorder.body
    assert body["model"] == "test-model"
    assert body["max_tokens"] == 321
    assert body["system"] == "sys"
    assert "temperature" not in body  # the installed SDK no longer takes sampling params
    content = body["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
    }
    assert content[1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": JPEG_B64}
    assert content[2] == {"type": "text", "text": "Describe the panel."}
    assert body["tools"][0]["name"] == RESULT_TOOL
    assert body["tools"][0]["input_schema"] == SCHEMA
    assert body["tool_choice"] == {"type": "tool", "name": RESULT_TOOL}

    assert response.parsed == ANSWER
    assert json.loads(response.text) == ANSWER
    assert response.usage.input_tokens == 1200
    assert response.usage.output_tokens == 340
    assert response.usage.cost_usd is None
    assert response.latency_ms >= 0.0
    assert response.raw["id"] == "msg_1"
    _assert_response_roundtrips(response)


async def test_anthropic_text_falls_back_to_extract_json(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_anthropic_message([{"type": "text", "text": FENCED}]))
    response = await _anthropic(monkeypatch, recorder).complete(_request(system=None))

    body = recorder.body
    assert "system" not in body
    assert "tools" not in body
    assert "tool_choice" not in body
    assert response.text == FENCED
    assert response.parsed == ANSWER


async def test_anthropic_retries_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(
        httpx2.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}}),
        httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}}),
        _anthropic_message([{"type": "text", "text": "{}"}]),
    )
    response = await _anthropic(monkeypatch, recorder).complete(_request())
    assert len(recorder.requests) == 3
    assert response.parsed == {}


async def test_anthropic_does_not_retry_bad_request(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(
        httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error"}})
    )
    with pytest.raises(anthropic.BadRequestError):
        await _anthropic(monkeypatch, recorder).complete(_request())
    assert len(recorder.requests) == 1


async def test_anthropic_gives_up_after_bounded_attempts(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(
        httpx2.Response(503, json={"type": "error", "error": {"type": "api_error"}})
    )
    with pytest.raises(anthropic.APIStatusError):
        await _anthropic(monkeypatch, recorder).complete(_request())
    assert len(recorder.requests) == _common.RETRY_ATTEMPTS


# --------------------------------------------------------------------------- OpenAI


def _chat_completion(text: str | None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020},
        },
    )


def _openai(
    monkeypatch: pytest.MonkeyPatch, recorder: Recorder, **settings: Any
) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(_settings(**settings))
    client = openai.AsyncOpenAI(api_key="k", max_retries=0, http_client=recorder.client())
    monkeypatch.setattr(provider, "_client", client)
    return provider


async def test_openai_payload_and_json_schema(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_chat_completion(json.dumps(ANSWER)))
    response = await _openai(monkeypatch, recorder).complete(_request(schema=SCHEMA))

    body = recorder.body
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.25
    assert body["max_completion_tokens"] == 321
    assert "max_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    user = body["messages"][1]
    assert user["role"] == "user"
    assert user["content"][0] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
    }
    assert user["content"][1]["image_url"]["url"] == f"data:image/jpeg;base64,{JPEG_B64}"
    assert user["content"][2] == {"type": "text", "text": "Describe the panel."}
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Scene_Annotation", "schema": SCHEMA, "strict": False},
    }

    assert response.parsed == ANSWER
    assert response.usage.input_tokens == 900
    assert response.usage.output_tokens == 120
    _assert_response_roundtrips(response)


async def test_openai_compatible_server_uses_max_tokens(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_chat_completion(FENCED))
    provider = _openai(monkeypatch, recorder, openai_base_url="http://localhost:8000/v1")
    response = await provider.complete(_request(system=None))

    body = recorder.body
    assert body["max_tokens"] == 321
    assert "max_completion_tokens" not in body
    assert "response_format" not in body
    assert body["messages"][0]["role"] == "user"
    assert response.parsed == ANSWER


async def test_openai_null_content_gives_empty_text(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_chat_completion(None))
    response = await _openai(monkeypatch, recorder).complete(_request())
    assert response.text == ""
    assert response.parsed is None


async def test_openai_retries_server_error(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(httpx.Response(500, json={"error": {}}), _chat_completion("{}"))
    response = await _openai(monkeypatch, recorder).complete(_request())
    assert len(recorder.requests) == 2
    assert response.parsed == {}


def test_openai_base_url_without_key_builds_client():
    provider = OpenAICompatProvider(_settings(openai_base_url="http://localhost:8000/v1"))
    client = provider._get_client()
    assert str(client.base_url).startswith("http://localhost:8000/v1")


# --------------------------------------------------------------------------- Gemini


def _gemini_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "thinking...", "thought": True}, {"text": text}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 800,
                "candidatesTokenCount": 150,
                "thoughtsTokenCount": 50,
            },
        },
    )


def _gemini(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> GeminiProvider:
    provider = GeminiProvider(_settings())
    client = genai.Client(
        api_key="k", http_options=genai_types.HttpOptions(httpx_async_client=recorder.client())
    )
    monkeypatch.setattr(provider, "_client", client)
    return provider


async def test_gemini_payload_json_mime_and_schema_in_prompt(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_gemini_response(json.dumps(ANSWER)))
    response = await _gemini(monkeypatch, recorder).complete(_request(schema=SCHEMA))

    request = recorder.requests[-1]
    assert request.url.path.endswith("models/test-model:generateContent")
    body = recorder.body
    parts = body["contents"][0]["parts"]
    assert body["contents"][0]["role"] == "user"
    # the SDK sends URL-safe base64
    assert parts[0]["inlineData"]["mimeType"] == "image/png"
    assert base64.urlsafe_b64decode(parts[0]["inlineData"]["data"]) == PNG
    assert parts[1]["inlineData"]["mimeType"] == "image/jpeg"
    assert base64.urlsafe_b64decode(parts[1]["inlineData"]["data"]) == JPEG
    assert parts[2]["text"].startswith("Describe the panel.")
    assert json.dumps(SCHEMA, separators=(",", ":")) in parts[2]["text"]
    assert body["systemInstruction"]["parts"] == [{"text": "sys"}]
    config = body["generationConfig"]
    assert config["maxOutputTokens"] == 321
    assert config["temperature"] == 0.25
    assert config["responseMimeType"] == "application/json"
    assert "responseSchema" not in config
    assert "responseJsonSchema" not in config

    assert response.text == json.dumps(ANSWER)  # the thought part is dropped
    assert response.parsed == ANSWER
    assert response.usage.input_tokens == 800
    assert response.usage.output_tokens == 200
    _assert_response_roundtrips(response)


async def test_gemini_without_schema(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_gemini_response(FENCED))
    response = await _gemini(monkeypatch, recorder).complete(_request(system=None))
    body = recorder.body
    assert body["contents"][0]["parts"][2] == {"text": "Describe the panel."}
    assert "systemInstruction" not in body
    assert "responseMimeType" not in body["generationConfig"]
    assert response.parsed == ANSWER


async def test_gemini_retries_resource_exhausted(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(
        httpx.Response(429, json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}),
        _gemini_response("{}"),
    )
    response = await _gemini(monkeypatch, recorder).complete(_request())
    assert len(recorder.requests) == 2
    assert response.parsed == {}


# --------------------------------------------------------------------------- Mistral


def _mistral_response(content: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": "test-model",
            "created": 0,
            "usage": {"prompt_tokens": 700, "completion_tokens": 90, "total_tokens": 790},
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _mistral(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> MistralProvider:
    provider = MistralProvider(_settings())
    monkeypatch.setattr(provider, "_client", Mistral(api_key="k", async_client=recorder.client()))
    return provider


async def test_mistral_payload_json_mode(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_mistral_response(json.dumps(ANSWER)))
    response = await _mistral(monkeypatch, recorder).complete(_request(schema=SCHEMA))

    body = recorder.body
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.25
    assert body["max_tokens"] == 321
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    chunks = body["messages"][1]["content"]
    assert chunks[0] == {"type": "image_url", "image_url": f"data:image/png;base64,{PNG_B64}"}
    assert chunks[1]["image_url"] == f"data:image/jpeg;base64,{JPEG_B64}"
    assert chunks[2]["type"] == "text"
    assert chunks[2]["text"].startswith("Describe the panel.")
    assert '"required":["schema_version"]' in chunks[2]["text"]

    assert response.parsed == ANSWER
    assert response.usage.input_tokens == 700
    assert response.usage.output_tokens == 90
    _assert_response_roundtrips(response)


async def test_mistral_chunked_content_and_no_json_mode(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_mistral_response([{"type": "text", "text": FENCED}]))
    response = await _mistral(monkeypatch, recorder).complete(_request(system=None))
    body = recorder.body
    assert "response_format" not in body
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"][2] == {"type": "text", "text": "Describe the panel."}
    assert response.text == FENCED
    assert response.parsed == ANSWER


async def test_mistral_retries_rate_limit(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(
        httpx.Response(429, json={"message": "rate limited"}), _mistral_response("{}")
    )
    response = await _mistral(monkeypatch, recorder).complete(_request())
    assert len(recorder.requests) == 2
    assert response.parsed == {}


# --------------------------------------------------------------------------- Ollama


def _ollama_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": text},
            "done": True,
            "prompt_eval_count": 600,
            "eval_count": 80,
        },
    )


def _ollama(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> OllamaProvider:
    provider = OllamaProvider(_settings())
    client = ollama.AsyncClient(
        host="http://ollama.test:11434", transport=httpx.MockTransport(recorder)
    )
    monkeypatch.setattr(provider, "_client", client)
    return provider


async def test_ollama_payload_format_schema(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_ollama_response(json.dumps(ANSWER)))
    response = await _ollama(monkeypatch, recorder).complete(_request(schema=SCHEMA))

    assert recorder.requests[-1].url.path == "/api/chat"
    body = recorder.body
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0.25, "num_predict": 321}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    user = body["messages"][1]
    assert user["role"] == "user"
    assert user["content"] == "Describe the panel."
    assert user["images"] == [PNG_B64, JPEG_B64]

    assert response.parsed == ANSWER
    assert response.usage.input_tokens == 600
    assert response.usage.output_tokens == 80
    _assert_response_roundtrips(response)


async def test_ollama_without_schema_falls_back(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(_ollama_response(FENCED))
    response = await _ollama(monkeypatch, recorder).complete(_request(system=None))
    body = recorder.body
    assert not body.get("format")
    assert body["messages"][0]["role"] == "user"
    assert response.parsed == ANSWER


async def test_ollama_retries_server_error(monkeypatch: pytest.MonkeyPatch):
    recorder = Recorder(httpx.Response(503, json={"error": "loading"}), _ollama_response("{}"))
    response = await _ollama(monkeypatch, recorder).complete(_request())
    assert len(recorder.requests) == 2
    assert response.parsed == {}


async def test_ollama_retries_connection_errors(monkeypatch: pytest.MonkeyPatch):
    attempts: list[Any] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return _ollama_response("{}")

    provider = OllamaProvider(_settings())
    client = ollama.AsyncClient(
        host="http://ollama.test:11434", transport=httpx.MockTransport(refuse)
    )
    monkeypatch.setattr(provider, "_client", client)
    response = await provider.complete(_request())
    assert len(attempts) == 2
    assert response.parsed == {}


def test_ollama_client_uses_configured_host():
    provider = OllamaProvider(_settings(ollama_host="http://gpu-box:11434"))
    client = provider._get_client()
    assert str(client._client.base_url).startswith("http://gpu-box:11434")


# --------------------------------------------------------------------------- LiteLLM


class FakeAcompletion:
    """Stands in for ``litellm.acompletion``: records kwargs, replays results or errors."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _model_response(text: str) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model="claude-opus-5",
        choices=[{"index": 0, "message": {"role": "assistant", "content": text}}],
        usage=litellm.Usage(prompt_tokens=500, completion_tokens=60, total_tokens=560),
    )


async def test_litellm_payload_and_credentials(monkeypatch: pytest.MonkeyPatch):
    fake = FakeAcompletion(_model_response(json.dumps(ANSWER)))
    monkeypatch.setattr(litellm, "acompletion", fake)
    provider = LiteLLMProvider(_settings(anthropic_api_key=SecretStr("sk-ant")))
    request = _request(schema=SCHEMA).model_copy(update={"model": "anthropic/claude-opus-5"})
    response = await provider.complete(request)

    kwargs = fake.calls[-1]
    assert kwargs["model"] == "anthropic/claude-opus-5"
    assert kwargs["api_key"] == "sk-ant"
    assert "api_base" not in kwargs
    assert kwargs["max_tokens"] == 321
    assert kwargs["temperature"] == 0.25
    assert kwargs["drop_params"] is True
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    content = kwargs["messages"][1]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
    }
    assert content[2] == {"type": "text", "text": "Describe the panel."}
    assert kwargs["response_format"]["json_schema"]["schema"] == SCHEMA
    assert kwargs["response_format"]["json_schema"]["strict"] is False

    assert response.parsed == ANSWER
    assert response.usage.input_tokens == 500
    assert response.usage.output_tokens == 60
    assert response.usage.cost_usd is None
    _assert_response_roundtrips(response)


async def test_litellm_ollama_prefix_uses_host_and_no_key(monkeypatch: pytest.MonkeyPatch):
    fake = FakeAcompletion(_model_response(FENCED))
    monkeypatch.setattr(litellm, "acompletion", fake)
    provider = LiteLLMProvider(_settings(ollama_host="http://gpu-box:11434"))
    request = _request().model_copy(update={"model": "ollama_chat/qwen2.5vl:7b"})
    response = await provider.complete(request)
    kwargs = fake.calls[-1]
    assert kwargs["api_base"] == "http://gpu-box:11434"
    assert "api_key" not in kwargs
    assert "response_format" not in kwargs
    assert response.parsed == ANSWER


async def test_litellm_unknown_prefix_leaves_credentials_to_litellm(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = FakeAcompletion(_model_response("{}"))
    monkeypatch.setattr(litellm, "acompletion", fake)
    request = _request().model_copy(update={"model": "bedrock/some-model"})
    await LiteLLMProvider(_settings()).complete(request)
    assert "api_key" not in fake.calls[-1]
    assert "api_base" not in fake.calls[-1]


async def test_litellm_retries_rate_limit(monkeypatch: pytest.MonkeyPatch):
    error = litellm_errors.RateLimitError("slow down", llm_provider="anthropic", model="m")
    fake = FakeAcompletion(error, _model_response("{}"))
    monkeypatch.setattr(litellm, "acompletion", fake)
    request = _request().model_copy(update={"model": "bedrock/some-model"})
    response = await LiteLLMProvider(_settings()).complete(request)
    assert len(fake.calls) == 2
    assert response.parsed == {}


async def test_litellm_does_not_retry_bad_request(monkeypatch: pytest.MonkeyPatch):
    error = litellm_errors.BadRequestError("bad", model="m", llm_provider="anthropic")
    fake = FakeAcompletion(error)
    monkeypatch.setattr(litellm, "acompletion", fake)
    request = _request().model_copy(update={"model": "bedrock/some-model"})
    with pytest.raises(litellm_errors.BadRequestError):
        await LiteLLMProvider(_settings()).complete(request)
    assert len(fake.calls) == 1


# --------------------------------------------------------------------------- credentials


@pytest.mark.parametrize(
    ("provider_cls", "model", "env_key"),
    [
        (AnthropicProvider, "claude-opus-5", "ANTHROPIC_API_KEY"),
        (OpenAICompatProvider, "gpt-4o", "OPENAI_API_KEY"),
        (GeminiProvider, "gemini-2.5-flash", "GOOGLE_API_KEY"),
        (MistralProvider, "pixtral-12b-2409", "MISTRAL_API_KEY"),
        (LiteLLMProvider, "anthropic/claude-opus-5", "ANTHROPIC_API_KEY"),
        (LiteLLMProvider, "gemini/gemini-2.5-flash", "GOOGLE_API_KEY"),
        (LiteLLMProvider, "mistral/pixtral-12b-2409", "MISTRAL_API_KEY"),
        (LiteLLMProvider, "openai/gpt-4o", "OPENAI_API_KEY"),
    ],
)
async def test_missing_key_is_reported_at_call_time(
    provider_cls: Any, model: str, env_key: str, monkeypatch: pytest.MonkeyPatch
):
    fake = FakeAcompletion(_model_response("{}"))
    monkeypatch.setattr(litellm, "acompletion", fake)
    provider = provider_cls(_settings())  # constructing without credentials is fine
    with pytest.raises(ValueError, match=env_key):
        await provider.complete(VisionRequest(model=model, prompt="p"))
    assert fake.calls == []


@pytest.mark.parametrize(
    ("provider_cls", "field"),
    [
        (AnthropicProvider, "anthropic_api_key"),
        (OpenAICompatProvider, "openai_api_key"),
        (GeminiProvider, "google_api_key"),
        (MistralProvider, "mistral_api_key"),
    ],
)
def test_client_is_created_lazily_once(provider_cls: Any, field: str):
    provider = provider_cls(_settings(**{field: SecretStr("k")}))
    assert provider._client is None
    client = provider._get_client()
    assert client is not None
    assert provider._get_client() is client


# --------------------------------------------------------------------------- helpers


def test_is_transient_status():
    assert _common.is_transient_status(429)
    assert _common.is_transient_status(503)
    assert _common.is_transient_status(529)
    assert not _common.is_transient_status(400)
    assert not _common.is_transient_status(None)
    assert not _common.is_transient_status(True)


def test_transport_errors_are_transient():
    assert _common.is_transport_error(httpx.ConnectError("refused"))
    assert _common.is_transport_error(ConnectionError())
    assert not _common.is_transport_error(ValueError())


def test_schema_name_is_sanitised():
    assert _common.schema_name({"title": "Scene Annotation v0!"}) == "Scene_Annotation_v0_"
    assert _common.schema_name({}) == "response"


def test_dump_raw_drops_unserialisable_objects():
    assert _common.dump_raw({"a": 1}) == {"a": 1}
    assert _common.dump_raw(object()) is None


async def test_call_with_retry_reraises_non_transient():
    calls = 0

    async def boom() -> None:
        nonlocal calls
        calls += 1
        raise KeyError("x")

    with pytest.raises(KeyError):
        await _common.call_with_retry(boom, lambda exc: isinstance(exc, OSError))
    assert calls == 1
