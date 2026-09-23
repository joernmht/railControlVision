"""A ``deepagents`` variant of the agentic approach: one planning agent with tools.

Instead of the fixed six-node loop, :func:`build_deep_agent` hands the crop,
read, validate and render tools to ``deepagents.create_deep_agent``, which
plans its own reading strategy (todo list, sub-agents). The model is a
LangChain chat-model adapter over any :class:`VisionProvider`
(:class:`VisionChatModel`). Providers have no native tool calling, so the
adapter describes the bound tools in the system prompt and turns a JSON answer
``{"tool_calls": [{"name", "args"}]}`` into LangChain tool calls.

Images cannot travel through tool arguments as JSON, so the tools address them
by name in a :class:`ImageWorkspace`: the scene image is ``"scene"``, every
crop and rendering gets a new name that the tool reports back.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from deepagents import create_deep_agent
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from rail_vision_bench.agents.documents import (
    add_usage,
    default_source,
    parse_scene,
    scene_id_for,
    stamp_document,
)
from rail_vision_bench.agents.graph import AgenticResult
from rail_vision_bench.providers.base import ImageInput, Usage, VisionProvider, VisionRequest
from rail_vision_bench.providers.parsing import extract_json
from rail_vision_bench.schema.models import Source
from rail_vision_bench.tools.tools import crop_tool, read_tool, render_tool, validate_tool

SCENE_IMAGE: Final = "scene"
"""Workspace name of the image being annotated."""

_MEDIA_TYPES: Final = ("image/png", "image/jpeg", "image/webp")

DEEP_AGENT_PROMPT: Final = """\
You transcribe one railway control image (German Gleisbild / Stelltisch panel, ESTW or CTC
screen, control-room stream frame) into one rail-vision-bench schema v0 JSON document.

Images are addressed by name: the image to annotate is "scene". Use `crop` to cut out and
magnify a region (it returns the new image's name), `read` to ask a focused question about any
named image (labels such as W12, 12a/12b, N1; indications such as +/-, Hp0, Vr2, Ks1, Sh1),
`validate` to check a candidate document against schema v0 and its graph rules, and `render`
to redraw a candidate as a panel image you can `read` and compare with the scene.

Plan the regions to read, read them, map the symbols to schema v0 kinds and states, connect
the nodes into edges between ports (buffer_stop/boundary: A; joint: A, B; switch: toe,
straight, diverging; crossing/ekw/dkw: A, B | C, D), assemble the document, validate it and
fix every error-level issue. Do not guess: unreadable values are "unknown" or null.

Your final answer is the document as one JSON object and nothing else."""

_TOOL_PROTOCOL: Final = """\

## Tools

You can call these tools (JSON Schema of their arguments in `parameters`):

{tools}

To call tools, answer with exactly one JSON object and nothing else:
{{"tool_calls": [{{"name": "<tool name>", "args": {{...}}}}]}}
The results come back in the next turn. When you need no tool, answer normally."""


def _image_from_block(block: Mapping[str, Any]) -> ImageInput | None:
    """Decode a LangChain image content block (``image_url`` data URL or base64 ``image``)."""
    media_type: str | None = None
    encoded: str | None = None
    if block.get("type") == "image_url":
        url = block.get("image_url")
        url = url.get("url") if isinstance(url, Mapping) else url
        if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
            header, encoded = url[len("data:") :].split(";base64,", 1)
            media_type = header
    elif block.get("type") == "image":
        encoded = block.get("base64") or block.get("data")
        media_type = block.get("mime_type")
    if encoded is None or media_type not in _MEDIA_TYPES:
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    return ImageInput.model_validate({"data": data, "media_type": media_type})


def _content_text(content: str | list[str | dict[str, Any]]) -> tuple[str, list[ImageInput]]:
    """Split message content into its text and its decodable images."""
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[ImageInput] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif block.get("type") == "text":
            texts.append(str(block.get("text", "")))
        elif (image := _image_from_block(block)) is not None:
            images.append(image)
    return "\n".join(texts), images


def messages_to_request(
    messages: Sequence[BaseMessage], *, model: str, tools: Sequence[Mapping[str, Any]] = ()
) -> VisionRequest:
    """Flatten a LangChain conversation into one provider request.

    System messages become the system prompt (followed by the tool protocol
    when tools are bound); the rest becomes a transcript in the user turn with
    every image attached in order.

    Args:
        messages: The conversation.
        model: The SDK model id.
        tools: OpenAI-format tool definitions bound to the model.

    Returns:
        The request.
    """
    system_parts: list[str] = []
    transcript: list[str] = []
    images: list[ImageInput] = []
    for message in messages:
        text, found = _content_text(message.content)
        images.extend(found)
        attached = f" [{len(found)} image(s) attached]" if found else ""
        if isinstance(message, SystemMessage):
            system_parts.append(text)
        elif isinstance(message, HumanMessage):
            transcript.append(f"[user]{attached}\n{text}")
        elif isinstance(message, ToolMessage):
            transcript.append(f"[tool result {message.name or message.tool_call_id}]\n{text}")
        elif isinstance(message, AIMessage):
            if text:
                transcript.append(f"[assistant]\n{text}")
            if message.tool_calls:
                calls = [
                    {"name": call["name"], "args": call["args"]} for call in message.tool_calls
                ]
                transcript.append(f"[assistant]\n{json.dumps({'tool_calls': calls})}")
        else:
            transcript.append(f"[{message.type}]\n{text}")
    if tools:
        listed = "\n".join(json.dumps(tool["function"], ensure_ascii=False) for tool in tools)
        system_parts.append(_TOOL_PROTOCOL.format(tools=listed))
    return VisionRequest(
        model=model,
        system="\n\n".join(part for part in system_parts if part) or None,
        prompt="\n\n".join(transcript),
        images=images,
    )


def _tool_calls(text: str, tool_names: set[str]) -> list[ToolCall]:
    """Recover the tool calls of an answer that follows the tool protocol."""
    parsed = extract_json(text)
    if parsed is None or not isinstance(parsed.get("tool_calls"), list):
        return []
    calls: list[ToolCall] = []
    for entry in parsed["tool_calls"]:
        if not isinstance(entry, Mapping) or entry.get("name") not in tool_names:
            continue
        args = entry.get("args", entry.get("arguments", {}))
        calls.append(
            ToolCall(
                name=str(entry["name"]),
                args=dict(args) if isinstance(args, Mapping) else {},
                id=f"call_{uuid.uuid4().hex[:12]}",
                type="tool_call",
            )
        )
    return calls


class VisionChatModel(BaseChatModel):
    """LangChain chat model that forwards every turn to a :class:`VisionProvider`.

    ``usage`` accumulates the provider's token counts over every call, so a
    caller can report the cost of a whole agent run.
    """

    provider: Any
    """The :class:`VisionProvider` (typed loosely: a Protocol is no pydantic field type)."""
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0

    _usage: Usage = PrivateAttr(default_factory=Usage)
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return f"rail-vision-bench-{getattr(self.provider, 'name', 'provider')}"

    @property
    def usage(self) -> Usage:
        """Usage summed over every call made through this adapter."""
        return self._usage

    @property
    def calls(self) -> int:
        """Number of provider calls made through this adapter."""
        return self._calls

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools: their definitions are sent in the system prompt of every call.

        Args:
            tools: LangChain tools, functions, pydantic models or OpenAI tool dicts.
            tool_choice: Accepted for API compatibility; the model always chooses.
            **kwargs: Further arguments bound to the model call.

        Returns:
            The model with ``tools`` bound.
        """
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[dict[str, Any]] = list(kwargs.get("tools") or [])
        request = messages_to_request(messages, model=self.model, tools=tools).model_copy(
            update={"max_tokens": self.max_tokens, "temperature": self.temperature}
        )
        response = await cast("VisionProvider", self.provider).complete(request)
        self._usage = add_usage(self._usage, response.usage)
        self._calls += 1
        names = {tool["function"]["name"] for tool in tools}
        calls = _tool_calls(response.text, names) if names else []
        message = AIMessage(
            content="" if calls else response.text,
            tool_calls=calls,
            usage_metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Providers are async only; the sync path runs the call on a private event loop.
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))


class ImageWorkspace:
    """Named images the deep agent's tools read from and write to."""

    def __init__(self) -> None:
        self.images: dict[str, ImageInput] = {}
        self.usage = Usage()
        self._counter = 0

    def put(self, name: str, image: ImageInput) -> None:
        """Store ``image`` under ``name`` (replacing any previous one)."""
        self.images[name] = image

    def add(self, prefix: str, data: bytes) -> str:
        """Store PNG bytes under a new ``<prefix>-<n>`` name and return the name."""
        self._counter += 1
        name = f"{prefix}-{self._counter}"
        self.images[name] = ImageInput(data=data, media_type="image/png")
        return name

    def get(self, name: str) -> ImageInput:
        """Return the image stored under ``name``.

        Raises:
            KeyError: If no image has that name; the message lists the known names.
        """
        try:
            return self.images[name]
        except KeyError:
            msg = f"no image named {name!r}; known: {sorted(self.images)}"
            raise KeyError(msg) from None


def build_tools(
    provider: VisionProvider, *, model: str, workspace: ImageWorkspace
) -> list[BaseTool]:
    """Wrap crop, read, validate and render as LangChain tools over ``workspace``.

    Args:
        provider: The model backend the read tool asks.
        model: The SDK model id of the read tool's requests.
        workspace: Where the tools find and store images.

    Returns:
        The four tools, named as in ``tools.TOOL_NAMES``.
    """

    def crop(image: str, bbox: list[float], zoom: float = 2.0) -> str:
        """Cut the region bbox = [x0, y0, x1, y1] (pixels) out of a named image and magnify it.

        Returns the name of the new image.
        """
        if len(bbox) != 4:
            return "error: bbox needs four numbers [x0, y0, x1, y1]"
        try:
            data = crop_tool(
                workspace.get(image).data, (bbox[0], bbox[1], bbox[2], bbox[3]), zoom=zoom
            )
        except (KeyError, ValueError, OSError) as exc:
            return f"error: {exc}"
        return f"stored the crop as image {workspace.add('crop', data)!r}"

    async def read(image: str, question: str) -> str:
        """Ask a focused question about a named image (a label, a lamp, an indication)."""
        try:
            source = workspace.get(image)
        except KeyError as exc:
            return f"error: {exc}"
        result = await read_tool(source.data, question, provider=provider, model=model)
        usage = result.payload.get("usage")
        if isinstance(usage, Mapping):
            workspace.usage = add_usage(workspace.usage, Usage.model_validate(dict(usage)))
        elif isinstance(usage, Usage):
            workspace.usage = add_usage(workspace.usage, usage)
        if not result.ok:
            return f"error: {'; '.join(result.errors) or 'read failed'}"
        return json.dumps(
            {"text": result.payload.get("text"), "confidence": result.payload.get("confidence")},
            ensure_ascii=False,
        )

    def validate(document: dict[str, Any]) -> str:
        """Validate a schema v0 document; returns the report (ok and issues)."""
        return json.dumps(validate_tool(document).payload, ensure_ascii=False)

    def render(document: dict[str, Any]) -> str:
        """Redraw a schema v0 document as a panel image; returns the new image's name."""
        try:
            data = render_tool(document)
        except (ValueError, OSError) as exc:
            return f"error: {exc}"
        return f"stored the rendering as image {workspace.add('render', data)!r}"

    return [
        StructuredTool.from_function(func=crop, name="crop"),
        StructuredTool.from_function(coroutine=read, name="read"),
        StructuredTool.from_function(func=validate, name="validate"),
        StructuredTool.from_function(func=render, name="render"),
    ]


def build_deep_agent(
    provider: VisionProvider,
    *,
    model: str,
    workspace: ImageWorkspace | None = None,
    chat_model: VisionChatModel | None = None,
) -> Any:
    """Create a deep agent that owns the crop, read, validate and render tools.

    Args:
        provider: The model backend.
        model: The SDK model id every request carries.
        workspace: The named images the tools use; put the scene under
            :data:`SCENE_IMAGE` before invoking the agent. A new, empty one when omitted.
        chat_model: The adapter to drive the agent with (to read its usage
            afterwards); a new :class:`VisionChatModel` over ``provider`` when omitted.

    Returns:
        The compiled deep agent (a LangGraph graph; invoke it with ``{"messages": [...]}``).
    """
    images = workspace if workspace is not None else ImageWorkspace()
    llm = chat_model if chat_model is not None else VisionChatModel(provider=provider, model=model)
    return create_deep_agent(
        model=llm,
        tools=build_tools(provider, model=model, workspace=images),
        system_prompt=DEEP_AGENT_PROMPT,
    )


async def run_deep_agent(
    provider: VisionProvider,
    image: ImageInput,
    *,
    model: str,
    scene_id: str | None = None,
    source: Source | None = None,
    recursion_limit: int = 60,
) -> AgenticResult:
    """Run the deep agent over one image and collect the outcome like :func:`run_agentic`.

    Args:
        provider: The model backend.
        image: The panel or screen image.
        model: The SDK model id.
        scene_id: The scene id; derived from the image bytes when omitted.
        source: The source block; derived from the image when omitted.
        recursion_limit: LangGraph step budget of the agent.

    Returns:
        The final answer as a stamped document (parsed when it matches schema
        v0), the usage of every model and read-tool call, the latency, and in
        ``rounds`` the number of model calls.

    Raises:
        ValueError: If ``source`` is omitted and the image cannot be decoded.
    """
    scene = scene_id if scene_id is not None else scene_id_for(image)
    src = source if source is not None else default_source(image, scene_id=scene)
    workspace = ImageWorkspace()
    workspace.put(SCENE_IMAGE, image)
    llm = VisionChatModel(provider=provider, model=model)
    agent = build_deep_agent(provider, model=model, workspace=workspace, chat_model=llm)
    data_url = f"data:{image.media_type};base64,{base64.b64encode(image.data).decode('ascii')}"
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"Annotate the image {SCENE_IMAGE!r} (attached): scene id {scene}, source "
                    f"kind {src.kind.value}, {src.width} x {src.height} pixels. Answer with the "
                    "schema v0 document only."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ]
    )
    started = time.perf_counter()
    final = await agent.ainvoke(
        {"messages": [message]}, config={"recursion_limit": recursion_limit}
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    messages = final.get("messages", [])
    answer = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None
    )
    raw_text = _content_text(answer.content)[0] if answer is not None else ""
    recovered = extract_json(raw_text)
    usage = add_usage(llm.usage, workspace.usage)
    if recovered is None:
        return AgenticResult(
            scene_id=scene,
            raw_text=raw_text,
            parse_error="the deep agent's final answer contains no JSON object",
            usage=usage,
            latency_ms=latency_ms,
            rounds=llm.calls,
        )
    document = stamp_document(recovered, scene_id=scene, source=src, model=model)
    annotation, error = parse_scene(document)
    return AgenticResult(
        scene_id=scene,
        annotation=annotation,
        document=document,
        raw_text=raw_text,
        parse_error=error,
        usage=usage,
        latency_ms=latency_ms,
        rounds=llm.calls,
        done=annotation is not None,
        validation=validate_tool(document).payload,
    )
