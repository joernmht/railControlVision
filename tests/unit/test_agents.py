"""Single-shot runner, the six agentic nodes end to end, assembly and the deep agent."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from PIL import Image

from rail_vision_bench.agents import nodes
from rail_vision_bench.agents.assembly import assemble_candidate
from rail_vision_bench.agents.deep import (
    SCENE_IMAGE,
    ImageWorkspace,
    VisionChatModel,
    build_deep_agent,
    build_tools,
    messages_to_request,
    run_deep_agent,
)
from rail_vision_bench.agents.documents import add_usage, default_source, scene_id_for
from rail_vision_bench.agents.graph import build_agent_graph, run_agentic
from rail_vision_bench.agents.prompts import fill_prompt, load_prompt, split_prompt
from rail_vision_bench.agents.single_shot import (
    ParseError,
    build_single_shot_request,
    run_single_shot,
    single_shot_attempt,
)
from rail_vision_bench.graph.validator import validate_document
from rail_vision_bench.providers.base import ImageInput, Usage, VisionRequest, VisionResponse
from rail_vision_bench.schema.models import Source, SourceKind

USAGE = Usage(input_tokens=10, output_tokens=5)


def png(width: int = 400, height: int = 100) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "black").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def image() -> ImageInput:
    return ImageInput(data=png())


class ScriptedProvider:
    """Answers by looking at the request; records every request it saw."""

    name: ClassVar[str] = "scripted"

    def __init__(self, answer: Callable[[VisionRequest], str | dict[str, Any]]) -> None:
        self.answer = answer
        self.requests: list[VisionRequest] = []

    async def complete(self, request: VisionRequest) -> VisionResponse:
        self.requests.append(request)
        answer = self.answer(request)
        if isinstance(answer, dict):
            return VisionResponse(text=json.dumps(answer), usage=USAGE)
        return VisionResponse(text=answer, usage=USAGE)


# --- single shot -------------------------------------------------------------------------


def test_split_and_fill_prompt():
    system, user = split_prompt(load_prompt("single_shot"))
    assert system.startswith("You transcribe railway control imagery")
    assert "## User" not in system
    assert "{scene_id}" in user
    filled = fill_prompt(user, scene_id="s1", source_kind="synthetic", image="a.png")
    assert "`s1`" in filled
    assert "{width}" in filled  # only the named placeholders are replaced
    assert split_prompt("no sections here") == ("no sections here", "")


def test_single_shot_request(image: ImageInput):
    source = default_source(image, scene_id="s1", kind=SourceKind.SYNTHETIC)
    request = build_single_shot_request(image, model="m-1", scene_id="s1", source=source)
    assert request.model == "m-1"
    assert request.system is not None
    assert '"schema_version": "v0"' in request.system  # literal JSON braces survive
    assert "`400` x `100`" in request.prompt
    assert "`synthetic`" in request.prompt
    assert request.images == [image]
    assert request.response_schema is not None
    assert request.response_schema["title"] == "SceneAnnotation"


async def test_single_shot_stamps_bookkeeping(image: ImageInput, minimal_doc: dict[str, Any]):
    tampered = {**minimal_doc, "scene_id": "made-up", "provenance": {"kind": "ground_truth"}}
    provider = ScriptedProvider(lambda _: f"Here you go:\n```json\n{json.dumps(tampered)}\n```")
    result = await single_shot_attempt(provider, image, model="m-1", scene_id="scene-7")
    assert result.parse_error is None
    assert result.annotation is not None
    assert result.annotation.scene_id == "scene-7"
    assert result.annotation.provenance.kind == "prediction"
    assert result.annotation.provenance.model_id == "m-1"
    assert result.annotation.source.width == 400
    assert result.usage == USAGE
    assert result.latency_ms >= 0
    assert "Here you go" in result.raw_text
    assert validate_document(result.annotation).ok


async def test_single_shot_prefers_parsed(image: ImageInput, minimal_doc: dict[str, Any]):
    class ParsedProvider:
        name: ClassVar[str] = "parsed"

        async def complete(self, request: VisionRequest) -> VisionResponse:
            return VisionResponse(text="", parsed=minimal_doc)

    annotation = await run_single_shot(ParsedProvider(), image, model="m")
    assert annotation.scene_id == scene_id_for(image)
    assert [node.id for node in annotation.topology.nodes] == ["n_w", "n_j", "n_e"]


async def test_single_shot_without_json(image: ImageInput):
    provider = ScriptedProvider(lambda _: "I cannot see a panel.")
    result = await single_shot_attempt(provider, image, model="m")
    assert result.annotation is None
    assert result.document is None
    assert result.parse_error == "the model output contains no JSON object"
    with pytest.raises(ParseError) as exc:
        await run_single_shot(provider, image, model="m")
    assert exc.value.raw_text == "I cannot see a panel."
    assert isinstance(exc.value, ValueError)


async def test_single_shot_schema_invalid(image: ImageInput):
    provider = ScriptedProvider(lambda _: {"topology": {"nodes": [{"id": "x", "kind": "tree"}]}})
    with pytest.raises(ParseError, match="does not match schema v0") as exc:
        await run_single_shot(provider, image, model="m", scene_id="s")
    assert exc.value.document is not None
    assert exc.value.document["scene_id"] == "s"


# --- agentic loop ------------------------------------------------------------------------

INTERPRETATION = {
    "nodes": [
        {"id": "n_w", "kind": "boundary", "bbox": [0, 40, 10, 60]},
        {"id": "n_j", "kind": "joint", "label": "1-2", "bbox": [195, 40, 205, 60]},
        {"id": "n_e", "kind": "boundary", "bbox": [390, 40, 400, 60]},
    ],
    "signals": [
        {
            "id": "sig_1",
            "kind": "main",
            "system": "H/V",
            "label": "A",
            "bbox": [170, 20, 180, 40],
            "state": {"aspect": "proceed", "raw": "Hp1"},
        }
    ],
    "derailers": [],
    "legend": {},
}
GEOMETRY = {
    "edges": [
        {
            "id": "e1",
            "kind": "main",
            "a": {"node": "n_w", "port": "A"},
            "b": {"node": "n_j", "port": "A"},
            "polyline": [[0, 50], [200, 50]],
        },
        {
            "id": "e2",
            "kind": "main",
            "a": {"node": "n_j", "port": "B"},
            "b": {"node": "n_e", "port": "A"},
            "polyline": [[200, 50], [400, 50]],
        },
    ],
    "attachments": {"sig_1": {"edge": "e1", "offset": 0.9, "direction": "a_to_b"}},
    "tracks": {
        "e1": {"occupancy": "free", "route_set": False},
        "e2": {"occupancy": "free", "route_set": False},
    },
}


def node_of(request: VisionRequest) -> str:
    if request.system is None:
        return "read"
    return request.system.splitlines()[0].removeprefix("# agentic/")


def loop_answers(
    *, builder: str | dict[str, Any] = "no idea", verdicts: list[bool] | None = None
) -> Callable[[VisionRequest], str | dict[str, Any]]:
    pending = list(verdicts or [True])

    def answer(request: VisionRequest) -> str | dict[str, Any]:
        node = node_of(request)
        if node == "planner":
            return {"plan": [{"item": "read the signal label", "bbox": [150, 0, 250, 100]}]}
        if node == "read":
            readings = {"readings": [{"text": "A", "what": "label", "bbox": [40, 20, 60, 40]}]}
            return {"text": json.dumps(readings), "confidence": 0.9}
        if node == "interpreter":
            return INTERPRETATION
        if node == "geometer":
            return GEOMETRY
        if node == "builder":
            return builder
        if node == "critic":
            verdict = pending.pop(0) if pending else True
            return {"matches": verdict, "critique": [] if verdict else ["signal A is missing"]}
        raise AssertionError(node)

    return answer


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []

    def render(document: Any, **_: Any) -> bytes:
        rendered.append(document)
        return png(40, 10)

    monkeypatch.setattr(nodes, "render_tool", render)
    return rendered


async def test_agentic_loop_accepts_in_one_round(
    image: ImageInput, fake_render: list[dict[str, Any]]
):
    provider = ScriptedProvider(loop_answers())
    result = await run_agentic(provider, image, model="m-1", scene_id="s1", max_rounds=3)
    assert result.done
    assert result.rounds == 1
    assert result.parse_error is None
    assert result.annotation is not None
    assert result.annotation.scene_id == "s1"
    assert result.annotation.provenance.model_id == "m-1"
    assert {edge.id for edge in result.annotation.topology.edges} == {"e1", "e2"}
    assert result.annotation.state.signals["sig_1"].raw == "Hp1"
    assert validate_document(result.annotation, strict=True).ok
    assert result.validation is not None
    assert result.validation["ok"] is True
    # planner, read, interpreter, geometer, builder, critic: six calls
    assert [node_of(r) for r in provider.requests] == [
        "planner",
        "read",
        "interpreter",
        "geometer",
        "builder",
        "critic",
    ]
    assert result.usage == Usage(input_tokens=60, output_tokens=30)
    assert json.loads(result.raw_text) == result.document
    assert all(request.model == "m-1" for request in provider.requests)
    # the builder's non-JSON answer is recorded and the deterministic draft is kept
    assert any("keeping the draft" in error for error in result.errors)
    assert len(fake_render) == 1
    critic_request = provider.requests[-1]
    assert len(critic_request.images) == 2


async def test_reader_maps_crop_bboxes_to_the_full_image(
    image: ImageInput, fake_render: list[dict[str, Any]]
):
    provider = ScriptedProvider(loop_answers())
    graph = build_agent_graph(provider, model="m", max_rounds=1)
    final = await graph.ainvoke({"scene_id": "s", "image": image})
    assert final["crops"] == [{"bbox": [150.0, 0.0, 250.0, 100.0], "item": "read the signal label"}]
    # crop pixels [40, 20, 60, 40] at zoom 2 over the region starting at (150, 0)
    assert final["readings"][0]["bbox"] == [170.0, 10.0, 180.0, 20.0]
    assert final["model"] == "m"
    assert final["max_rounds"] == 1


async def test_agentic_loop_feeds_the_critique_back(
    image: ImageInput, fake_render: list[dict[str, Any]], minimal_doc: dict[str, Any]
):
    provider = ScriptedProvider(loop_answers(builder=minimal_doc, verdicts=[False, True]))
    rounds: list[int] = []

    async def on_round(state: Any) -> None:
        rounds.append(state["round"])

    result = await run_agentic(
        provider, image, model="m", scene_id="s", max_rounds=3, on_round=on_round
    )
    assert result.done
    assert result.rounds == 2
    assert rounds == [1, 2]
    planners = [r for r in provider.requests if node_of(r) == "planner"]
    assert "signal A is missing" in planners[1].prompt
    assert result.annotation is not None
    assert result.annotation.scene_id == "s"  # the model's minimal doc was re-stamped


async def test_agentic_loop_stops_at_the_budget(
    image: ImageInput, fake_render: list[dict[str, Any]]
):
    provider = ScriptedProvider(loop_answers(verdicts=[False, False, False]))
    result = await run_agentic(provider, image, model="m", scene_id="s", max_rounds=2)
    assert not result.done
    assert result.rounds == 2
    assert result.critique is not None
    assert "signal A is missing" in result.critique
    assert result.annotation is not None  # the last candidate is still reported


async def test_agentic_loop_survives_unusable_answers(
    image: ImageInput, fake_render: list[dict[str, Any]]
):
    def answer(request: VisionRequest) -> str | dict[str, Any]:
        return "sorry" if node_of(request) != "read" else {"text": "", "confidence": 0.0}

    provider = ScriptedProvider(answer)
    result = await run_agentic(provider, image, model="m", scene_id="s", max_rounds=1)
    assert result.rounds == 1
    assert result.annotation is not None  # an empty but well-formed draft
    assert result.annotation.topology.nodes == []
    assert any("planner round 1" in error for error in result.errors)
    assert any("no usable comparison verdict" in error for error in result.errors)


async def test_critic_rejects_invalid_candidate(image: ImageInput, station_dkw_doc: dict[str, Any]):
    broken = json.loads(json.dumps(station_dkw_doc))
    broken["state"]["switches"]["sw1"]["active_paths"] = [["A", "C"]]
    update = await nodes.critic(
        {"image": image, "model": "m", "candidate": broken, "round": 1, "max_rounds": 3},
        provider=ScriptedProvider(lambda _: {"matches": True, "critique": []}),
    )
    assert update["done"] is False
    assert update["validation"] is not None
    assert update["validation"]["ok"] is False
    assert update["critique"] is not None
    assert "STATE_WRONG_FIELD_FOR_KIND" in update["critique"]


async def test_critic_without_candidate(image: ImageInput):
    update = await nodes.critic(
        {"image": image, "candidate": None}, provider=ScriptedProvider(lambda _: "")
    )
    assert update["done"] is False
    assert update["validation"] is None


# --- assembly ----------------------------------------------------------------------------


def test_assemble_minimal_scene_is_strict_valid():
    body = assemble_candidate(INTERPRETATION, GEOMETRY, width=400, height=100)
    doc = {
        "schema_version": "v0",
        "scene_id": "s",
        "source": {"kind": "synthetic", "image": "s.png", "width": 400, "height": 100},
        "provenance": {"kind": "prediction"},
        **body,
    }
    assert validate_document(doc, strict=True).ok
    assert body["meta"] == {}


def test_assemble_repairs_and_drops():
    interpretation = {
        "nodes": [
            {"id": "sw1", "kind": "switch", "state": {"position": "sideways", "raw": "-"}},
            {"id": "bad id!", "kind": "joint"},
            {"id": "t1", "kind": "tree"},
            {"id": "n_w", "kind": "boundary", "bbox": [-5, 10, 5, 2000]},
        ],
        "signals": [{"id": "s1", "kind": "main"}, {"id": "s2", "kind": "wizard"}],
    }
    geometry = {
        "edges": [
            {"id": "e1", "a": {"node": "n_w", "port": "a"}, "b": {"node": "sw1", "port": "TOE"}},
            {
                "id": "e2",
                "a": {"node": "sw1", "port": "straight"},
                "b": {"node": "x9", "port": "A"},
                "polyline": [[10, 10], [500, 10]],
            },
            {"id": "e3", "a": {"node": "n_w", "port": "A"}, "b": {"node": "n_w", "port": "A"}},
        ],
        "attachments": {"s2": {"edge": "e2", "direction": "b_to_a"}},
        "tracks": {"e1": {"occupancy": "occupied", "route_set": "yes"}},
    }
    body = assemble_candidate(interpretation, geometry, width=400, height=100)
    nodes_by_id = {node["id"]: node for node in body["topology"]["nodes"]}
    # sw1 has no diverging edge: dropped, its two edge ends terminate at new boundaries
    assert "sw1" not in nodes_by_id
    assert nodes_by_id["x9"]["kind"] == "boundary"  # undeclared, inferred from its one A end
    assert nodes_by_id["n_w"]["geometry"]["bbox"] == [0.0, 10.0, 5.0, 100.0]
    edges = {edge["id"]: edge for edge in body["topology"]["edges"]}
    assert set(edges) == {"e1", "e2"}
    assert edges["e1"]["b"]["port"] == "A"
    assert edges["e1"]["b"]["node"].startswith("bnd")
    assert edges["e2"]["geometry"]["polyline"] == [[10.0, 10.0], [400.0, 10.0]]
    assert [signal["id"] for signal in body["signals"]] == ["s2"]
    assert body["signals"][0]["kind"] == "unknown"
    assert body["signals"][0]["at"]["offset"] == 0.5
    assert body["state"]["tracks"]["e1"] == {"occupancy": "occupied", "route_set": None}
    assert body["state"]["switches"] == {}
    notes = "\n".join(body["meta"]["notes"])
    for fragment in ("'sw1'", "'bad id!'", "'t1'", "'e3'", "signal 's1'"):
        assert fragment in notes
    doc = {
        "schema_version": "v0",
        "scene_id": "s",
        "source": {"kind": "synthetic", "image": "s.png", "width": 400, "height": 100},
        "provenance": {"kind": "prediction"},
        **body,
    }
    assert validate_document(doc).ok


def test_add_usage_cost():
    assert add_usage(Usage(), Usage()).cost_usd is None
    total = add_usage(Usage(input_tokens=1, cost_usd=0.5), Usage(output_tokens=2))
    assert total == Usage(input_tokens=1, output_tokens=2, cost_usd=0.5)


def test_default_source_rejects_garbage():
    with pytest.raises(ValueError, match="cannot decode"):
        default_source(ImageInput(data=b"nope"), scene_id="s")
    source = default_source(ImageInput(data=png(), media_type="image/png"), scene_id="s")
    assert source == Source(kind=SourceKind.STREAM_FRAME, image="s.png", width=400, height=100)


# --- deep agent --------------------------------------------------------------------------


def test_build_deep_agent_constructs(image: ImageInput):
    agent = build_deep_agent(ScriptedProvider(lambda _: "{}"), model="m")
    assert {"model", "tools"} <= set(agent.get_graph().nodes)


def test_messages_to_request_carries_images_and_tools(image: ImageInput):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    tools = build_tools(ScriptedProvider(lambda _: ""), model="m", workspace=ImageWorkspace())
    from langchain_core.utils.function_calling import convert_to_openai_tool

    request = messages_to_request(
        [
            SystemMessage(content="be precise"),
            HumanMessage(
                content=[
                    {"type": "text", "text": "annotate"},
                    {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
                ]
            ),
            AIMessage(content="", tool_calls=[{"name": "validate", "args": {}, "id": "c1"}]),
            ToolMessage(content="{}", tool_call_id="c1", name="validate"),
        ],
        model="m",
        tools=[convert_to_openai_tool(tool) for tool in tools],
    )
    assert request.system is not None
    assert request.system.startswith("be precise")
    assert '"name": "crop"' in request.system
    assert len(request.images) == 1
    assert "[user] [1 image(s) attached]\nannotate" in request.prompt
    assert '"tool_calls"' in request.prompt
    assert "[tool result validate]" in request.prompt


async def test_run_deep_agent_calls_tools(image: ImageInput, minimal_doc: dict[str, Any]):
    def answer(request: VisionRequest) -> str | dict[str, Any]:
        if "[tool result validate]" in request.prompt:
            return minimal_doc
        return {"tool_calls": [{"name": "validate", "args": {"document": minimal_doc}}]}

    provider = ScriptedProvider(answer)
    result = await run_deep_agent(provider, image, model="m-2", scene_id="deep-1")
    assert result.parse_error is None
    assert result.annotation is not None
    assert result.annotation.scene_id == "deep-1"
    assert result.annotation.provenance.model_id == "m-2"
    assert result.rounds == 2
    assert result.usage == Usage(input_tokens=20, output_tokens=10)
    assert provider.requests[0].images[0].data == image.data


async def test_deep_tools_address_images_by_name(
    image: ImageInput, monkeypatch: pytest.MonkeyPatch
):
    from rail_vision_bench.agents import deep
    from rail_vision_bench.tools.tools import ToolResult

    async def read(data: bytes, question: str, **_: Any) -> ToolResult:
        usage = {"input_tokens": 3, "output_tokens": 1, "cost_usd": None}
        return ToolResult(ok=True, payload={"text": "W12", "confidence": 0.8, "usage": usage})

    monkeypatch.setattr(deep, "read_tool", read)
    workspace = ImageWorkspace()
    workspace.put(SCENE_IMAGE, image)
    tools = {
        t.name: t
        for t in build_tools(ScriptedProvider(lambda _: ""), model="m", workspace=workspace)
    }
    assert "crop-1" in tools["crop"].invoke({"image": "scene", "bbox": [0, 0, 50, 50]})
    assert Image.open(io.BytesIO(workspace.get("crop-1").data)).size == (100, 100)
    answer = json.loads(await tools["read"].ainvoke({"image": "crop-1", "question": "label?"}))
    assert answer == {"text": "W12", "confidence": 0.8}
    assert workspace.usage.input_tokens == 3
    assert "error" in tools["crop"].invoke({"image": "nope", "bbox": [0, 0, 1, 1]})


def test_vision_chat_model_turns_json_into_tool_calls():
    from langchain_core.messages import HumanMessage

    provider = ScriptedProvider(lambda _: {"tool_calls": [{"name": "validate", "args": {"x": 1}}]})
    llm = VisionChatModel(provider=provider, model="m")
    bound = llm.bind_tools(build_tools(provider, model="m", workspace=ImageWorkspace()))
    message = bound.invoke([HumanMessage(content="go")])
    assert message.tool_calls[0]["name"] == "validate"
    assert message.tool_calls[0]["args"] == {"x": 1}
    plain = llm.invoke([HumanMessage(content="go")])
    assert plain.tool_calls == []
    assert llm.calls == 2
