"""Model runs for ground-truth pre-fill: provider, single-shot, runner, matching, consensus,
review store and review app."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from rail_vision_bench import runner
from rail_vision_bench.agents import single_shot
from rail_vision_bench.config import Mode, RunConfig
from rail_vision_bench.dataset.manifest import ManifestRow, manifest_path, write_manifest
from rail_vision_bench.eval.matching import (
    UNMATCHABLE,
    elements,
    label_distance,
    match_elements,
    pair_cost,
)
from rail_vision_bench.providers.base import ImageInput, Usage, VisionRequest, VisionResponse
from rail_vision_bench.providers.openai_compat import OpenAICompatProvider, ProviderError
from rail_vision_bench.review.app import create_review_app
from rail_vision_bench.review.consensus import lenient_document, merge
from rail_vision_bench.review.store import build_drafts, drafts_dir, finalize, write_document
from rail_vision_bench.schema.models import SceneAnnotation, SourceKind
from rail_vision_bench.settings import Settings
from tests.conftest import EXAMPLES_DIR, load_json

STATION = load_json(EXAMPLES_DIR / "station_dkw.json")


def _doc(**changes: Any) -> SceneAnnotation:
    data = copy.deepcopy(STATION)
    data.update(changes)
    return SceneAnnotation.model_validate(data)


def _jpeg(width: int = 800, height: int = 400) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


# -- provider ---------------------------------------------------------------------------------


async def test_openai_compat_payload_and_response():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01},
            },
        )

    settings = Settings(
        _env_file=None,
        openai_base_url="https://openrouter.ai/api/v1/",
        openai_api_key=SecretStr("k"),
    )
    provider = OpenAICompatProvider(settings, transport=httpx.MockTransport(handler))
    response = await provider.complete(
        VisionRequest(
            model="m",
            prompt="p",
            system="sys",
            images=[ImageInput(data=b"abc", media_type="image/jpeg")],
            extra={"reasoning": {"effort": "high"}},
        )
    )
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,YWJj"
    assert body["usage"] == {"include": True}
    assert body["reasoning"] == {"effort": "high"}
    assert response.text == "{}"
    assert response.usage == Usage(input_tokens=10, output_tokens=5, cost_usd=0.01)


@pytest.mark.parametrize(
    ("status", "payload"),
    [(429, {"error": "slow down"}), (200, {"error": {"message": "bad"}}), (200, {"choices": []})],
)
async def test_openai_compat_errors(status: int, payload: dict[str, Any]):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json=payload))
    provider = OpenAICompatProvider(Settings(_env_file=None), transport=transport)
    with pytest.raises(ProviderError) as exc:
        await provider.complete(VisionRequest(model="m", prompt="p"))
    assert exc.value.status_code == status


# -- single shot ------------------------------------------------------------------------------


def test_split_prompt_and_both_packaged_prompts():
    for name in ("single_shot", "prefill"):
        system, user = single_shot.split_prompt(
            (Path(single_shot.__file__).parent.parent / "prompts" / f"{name}.md").read_text()
        )
        assert system.startswith("You transcribe railway control imagery")
        assert "{scene_id}" in user
        assert "{width}" in user
    with pytest.raises(ValueError, match="System"):
        single_shot.split_prompt("no sections")


@pytest.mark.parametrize(
    "text", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Here you go:\n{"a": 1}\nThanks']
)
def test_extract_json(text: str):
    assert single_shot.extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json", "{broken", "[1, 2]"])
def test_extract_json_rejects(text: str):
    with pytest.raises(ValueError):  # noqa: PT011 - every unusable answer raises ValueError
        single_shot.extract_json(text)


def test_rescale_geometry_scales_only_coordinates():
    doc = {
        "geometry": {"point": [10, 20], "bbox": [1, 2, 3, 4], "polyline": [[0, 0], [5, 5]]},
        "at": {"offset": 0.5},
        "nested": [{"geometry": {"point": [1, 1]}}],
        "flag": True,
    }
    out = single_shot.rescale_geometry(doc, 2.0)
    assert out["geometry"] == {
        "point": [20, 40],
        "bbox": [2, 4, 6, 8],
        "polyline": [[0, 0], [10, 10]],
    }
    assert out["at"] == {"offset": 0.5}
    assert out["nested"][0]["geometry"]["point"] == [2, 2]
    assert out["flag"] is True


def test_prepare_image_downscales(tmp_path: Path):
    path = tmp_path / "big.jpg"
    path.write_bytes(_jpeg(4096, 1024))
    prepared = single_shot.prepare_image(path, max_side=2048)
    assert (prepared.width, prepared.height) == (2048, 512)
    assert prepared.scale == 2.0
    small = tmp_path / "small.jpg"
    small.write_bytes(_jpeg(100, 50))
    assert single_shot.prepare_image(small).scale == 1.0


class _FakeProvider:
    name: ClassVar[str] = "fake"

    def __init__(self, answers: list[str | Exception]) -> None:
        self.answers = answers
        self.requests: list[VisionRequest] = []

    async def complete(self, request: VisionRequest) -> VisionResponse:
        self.requests.append(request)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return VisionResponse(
            text=answer, usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.5), latency_ms=3
        )


def _scene(tmp_path: Path, width: int = 1600, height: int = 800) -> single_shot.SceneInput:
    path = tmp_path / "scene.jpg"
    path.write_bytes(_jpeg(width, height))
    return single_shot.SceneInput("sc1", "panel_photo", path, width, height)


async def test_run_single_shot_maps_geometry_and_header(tmp_path: Path):
    scene = _scene(tmp_path, 4096, 2048)
    answer = json.dumps(STATION)
    provider = _FakeProvider([answer])
    result = await single_shot.run_single_shot(provider, "m-id", scene, run_id="r1")
    assert result.parse_error is None
    assert result.annotation is not None
    assert result.annotation.scene_id == "sc1"
    assert result.annotation.source.width == 4096
    assert result.annotation.provenance.model_id == "m-id"
    assert result.annotation.topology.nodes[0].geometry is not None
    assert result.annotation.topology.nodes[0].geometry.point == (40.0, 400.0)  # x2
    sent = provider.requests[0]
    assert "`2048` x `1024`" in sent.prompt
    assert sent.images[0].media_type == "image/jpeg"

    bad = await single_shot.run_single_shot(
        _FakeProvider(['{"topology": 5}']), "m", scene, run_id="r"
    )
    assert bad.annotation is None
    assert bad.parse_error is not None
    assert bad.parse_error.startswith("schema:")


# -- runner -----------------------------------------------------------------------------------


def _bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers: list[str | Exception]
) -> tuple[RunConfig, _FakeProvider]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.yaml").write_text(
        "name: t\nsource_kind: panel_photo\nsplit: s1\nprompt: prefill\n", encoding="utf-8"
    )
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  - {name: m1, provider: openai, model_id: x/m1,\n"
        "     params: {extra_body: {reasoning: {effort: high}}}}\n"
        "  - {name: m2, provider: openai, model_id: x/m2}\n",
        encoding="utf-8",
    )
    image = tmp_path / "img.jpg"
    image.write_bytes(_jpeg())
    rows = [
        ManifestRow(
            scene_id=f"sc{i}",
            split="s1",
            partition="dev",
            source_kind=SourceKind.PANEL_PHOTO,
            image=str(image),
            gt=f"data/gt/s1/sc{i}.json",
            width=800,
            height=400,
        )
        for i in (1, 2)
    ]
    write_manifest(rows, manifest_path(Path("data"), "s1"))
    provider = _FakeProvider(answers)
    monkeypatch.setattr(
        "rail_vision_bench.providers.registry.get_provider", lambda name, settings=None: provider
    )
    monkeypatch.setattr(runner, "RETRY_WAITS_S", (0.0, 0.0))
    config = RunConfig(
        name="r1",
        task=Path("task.yaml"),
        catalogue=Path("models.yaml"),
        models=["m1", "m2"],
        out_dir=Path("runs"),
    )
    return config, provider


def test_run_benchmark_writes_records_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, provider = _bench(tmp_path, monkeypatch, [json.dumps(STATION)])
    out = runner.run_benchmark(config, settings=Settings(_env_file=None, rvb_data_dir=Path("data")))
    records = runner.read_predictions(out / "predictions.jsonl")
    assert sorted((r.scene_id, r.model) for r in records) == [
        ("sc1", "m1"),
        ("sc1", "m2"),
        ("sc2", "m1"),
        ("sc2", "m2"),
    ]
    assert all(r.annotation is not None and r.cost_usd == 0.5 for r in records)
    assert all(r.validation is not None for r in records)
    assert (out / "run.yaml").is_file()
    assert any(req.extra == {"reasoning": {"effort": "high"}} for req in provider.requests)
    assert "every visible element" in provider.requests[0].prompt  # the task's prompt (prefill)
    # resume: nothing new to do
    runner.run_benchmark(config, settings=Settings(_env_file=None, rvb_data_dir=Path("data")))
    assert len(runner.read_predictions(out / "predictions.jsonl")) == 4


def test_run_benchmark_records_failures_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, provider = _bench(
        tmp_path, monkeypatch, [ProviderError(400, "bad request"), "not json"]
    )
    settings = Settings(_env_file=None, rvb_data_dir=Path("data"))
    out = runner.run_benchmark(
        config.model_copy(update={"models": ["m1"], "limit": 1}), settings=settings
    )
    (record,) = runner.read_predictions(out / "predictions.jsonl")
    assert record.annotation is None
    assert record.parse_error is not None
    assert record.parse_error.startswith("provider: HTTP 400")
    provider.answers = [json.dumps(STATION)]
    runner.run_benchmark(
        config.model_copy(update={"models": ["m1"], "limit": 1}),
        settings=settings,
        retry_failed=True,
    )
    records = runner.read_predictions(out / "predictions.jsonl")
    assert [r.attempt for r in records] == [1, 2]
    assert records[-1].annotation is not None


def test_run_benchmark_retries_transient_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config, _ = _bench(tmp_path, monkeypatch, [ProviderError(429, "slow"), json.dumps(STATION)])
    out = runner.run_benchmark(
        config.model_copy(update={"models": ["m1"], "limit": 1}),
        settings=Settings(_env_file=None, rvb_data_dir=Path("data")),
    )
    (record,) = runner.read_predictions(out / "predictions.jsonl")
    assert record.annotation is not None


def test_model_settings_needs_the_key_env(monkeypatch: pytest.MonkeyPatch):
    from rail_vision_bench.config import ModelConfig

    model = ModelConfig(
        name="m",
        provider="openai",
        model_id="x",
        params={"base_url": "https://h/v1", "api_key_env": "RVB_TEST_KEY"},
    )
    monkeypatch.delenv("RVB_TEST_KEY", raising=False)
    with pytest.raises(KeyError, match="RVB_TEST_KEY"):
        runner._model_settings(Settings(_env_file=None), model)
    monkeypatch.setenv("RVB_TEST_KEY", "secret")
    settings = runner._model_settings(Settings(_env_file=None), model)
    assert settings.openai_base_url == "https://h/v1"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "secret"


# -- matching ---------------------------------------------------------------------------------


def test_match_elements_identity_and_geometry_gate():
    doc = _doc()
    for family in ("nodes", "edges", "signals", "derailers", "routes"):
        matches = match_elements(doc, doc, family=family)
        assert all(m.gt_id == m.pred_id for m in matches)
        assert len(matches) == len(elements(doc, family))
    first, second = doc.topology.nodes[0], doc.topology.nodes[3]
    assert first.geometry is not None
    assert pair_cost(first, second, 1000.0) == UNMATCHABLE  # 530 px apart, same-kind or not
    assert label_distance(None, None) == 0.0
    assert label_distance("12a", None) == 1.0
    assert label_distance("12a", "12b") == pytest.approx(1 / 3)


# -- consensus --------------------------------------------------------------------------------


def _shifted(dx: float) -> dict[str, Any]:
    data = copy.deepcopy(STATION)
    for node in data["topology"]["nodes"]:
        node["geometry"] = {
            "point": [node["geometry"]["point"][0] + dx, node["geometry"]["point"][1]]
        }
        node["id"] = "m_" + node["id"]
    for edge in data["topology"]["edges"]:
        edge["a"]["node"] = "m_" + edge["a"]["node"]
        edge["b"]["node"] = "m_" + edge["b"]["node"]
    data["state"]["switches"] = {"m_" + k: v for k, v in data["state"]["switches"].items()}
    for route in data["routes"]:
        route["switch_positions"] = {"m_" + k: v for k, v in route["switch_positions"].items()}
        if not route["end"].startswith("sig_"):
            route["end"] = "m_" + route["end"]
    return data


def test_merge_agrees_across_renamed_and_shifted_models():
    docs = [
        ("a", _doc()),
        ("b", SceneAnnotation.model_validate(_shifted(3))),
        ("c", SceneAnnotation.model_validate(_shifted(-3))),
    ]
    draft = merge(docs, run_id="r")
    review = draft.meta["review"]
    assert len(draft.topology.nodes) == len(STATION["topology"]["nodes"])
    assert len(draft.topology.edges) == len(STATION["topology"]["edges"])
    assert len(draft.signals) == len(STATION["signals"])
    assert len(draft.routes) == len(STATION["routes"])
    assert all(e["support"] == 3 for e in review["elements"].values())
    assert all(e["status"] == "consensus" for e in review["elements"].values()), review["elements"]
    assert review["suggestions"] == {}
    from rail_vision_bench.graph.validator import validate_scene

    assert validate_scene(draft, strict=True).ok


def test_merge_flags_disagreement_and_keeps_single_model_elements_as_suggestions():
    lone = copy.deepcopy(STATION)
    lone["topology"]["nodes"].append(
        {"id": "extra", "kind": "boundary", "geometry": {"point": [700, 50]}}
    )
    other = copy.deepcopy(STATION)
    other["signals"][0]["kind"] = "main"
    draft = merge(
        [("a", SceneAnnotation.model_validate(lone)), ("b", SceneAnnotation.model_validate(other))]
    )
    review = draft.meta["review"]
    suggested = [s for s in review["suggestions"].values() if s["family"] == "nodes"]
    assert len(suggested) == 1
    assert suggested[0]["models"] == ["a"]
    contested = [e for e in review["elements"].values() if e["status"] == "contested"]
    assert any("kind" in e["disagree"] for e in contested)


def test_status_tiers():
    from rail_vision_bench.review.consensus import status

    assert status(3, 5, []) == "consensus"
    assert status(3, 5, ["kind"]) == "contested"
    assert status(2, 5, []) == "minority"
    assert status(2, 4, []) == "minority"


def test_lenient_document_drops_only_invalid_elements():
    raw = copy.deepcopy(STATION)
    raw["topology"]["nodes"][0]["kind"] = "teleporter"
    raw["state"]["signals"]["sig_A"] = {"aspect": "maybe"}
    base = _doc()
    doc = lenient_document(json.dumps(raw), 1.0, base)
    assert doc is not None
    assert len(doc.topology.nodes) == len(STATION["topology"]["nodes"]) - 1
    assert "sig_A" not in doc.state.signals
    assert lenient_document("no json", 1.0, base) is None


def test_merge_requires_documents():
    with pytest.raises(ValueError, match="nothing"):
        merge([])


# -- store and app ----------------------------------------------------------------------------


def test_finalize_provenance():
    draft = merge([("a", _doc()), ("b", _doc())], run_id="r")
    reviewed = draft.model_dump(mode="json")
    reviewed["topology"]["nodes"][0]["label"] = "changed"
    changed_id = reviewed["topology"]["nodes"][0]["id"]
    reviewed["topology"]["nodes"].append(
        {"id": "new1", "kind": "boundary", "geometry": {"point": [1, 1]}}
    )
    reviewed["meta"]["review"]["checked"] = [changed_id]
    gt, report = finalize(SceneAnnotation.model_validate(reviewed), draft, annotator="joern")
    assert gt.provenance.kind == "ground_truth"
    assert gt.provenance.annotator == "joern"
    assert gt.meta["provenance"][changed_id] == "human_edited"
    assert gt.meta["provenance"]["new1"] == "human_added"
    summary = gt.meta["review_summary"]
    assert summary["human_edited"] == 1
    assert summary["human_added"] == 1
    assert summary["checked"] == 1
    assert summary["prefill_accepted"] == summary["final_elements"] - 2
    assert "review" not in gt.meta
    assert not report.ok  # the extra boundary node has no edge


def test_review_app_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    data = Path("data")
    image = tmp_path / "img.jpg"
    image.write_bytes(_jpeg())
    write_manifest(
        [
            ManifestRow(
                scene_id="station_dkw",
                split="s1",
                partition="dev",
                source_kind=SourceKind.SYNTHETIC,
                image=str(image),
                gt="data/gt/s1/station_dkw.json",
                width=800,
                height=400,
            )
        ],
        manifest_path(data, "s1"),
    )
    write_document(
        merge([("a", _doc()), ("b", _doc())], run_id="r"),
        drafts_dir(data, "s1") / "station_dkw.json",
    )
    client = TestClient(create_review_app(data, "s1", annotator="tester"))
    assert "Panel GT Review" in client.get("/").text
    (scene,) = client.get("/api/scenes").json()
    assert scene["status"] == "todo"
    assert scene["elements"] > 0
    payload = client.get("/api/scenes/station_dkw").json()
    assert payload["report"]["ok"]
    assert client.get("/api/scenes/station_dkw/image").headers["content-type"] == "image/jpeg"
    assert client.get("/api/scenes/nope").status_code == 404
    doc = payload["document"]
    assert client.put("/api/scenes/station_dkw/work", json=doc).json()["ok"]
    assert client.get("/api/scenes").json()[0]["status"] == "in_progress"
    assert client.put("/api/scenes/station_dkw/work", json={"bad": 1}).status_code == 422
    done = client.post("/api/scenes/station_dkw/finalize", json=doc).json()
    assert done["ok"]
    assert done["summary"]["prefill_accepted"] == done["summary"]["final_elements"]
    assert Path(done["written"]).is_file()
    assert client.get("/api/scenes").json()[0]["status"] == "done"


def test_build_drafts_skips_existing(tmp_path: Path):
    from rail_vision_bench.eval.records import PredictionRecord

    data = tmp_path / "data"
    image = tmp_path / "img.jpg"
    image.write_bytes(_jpeg())
    write_manifest(
        [
            ManifestRow(
                scene_id="station_dkw",
                split="s1",
                partition="dev",
                source_kind=SourceKind.SYNTHETIC,
                image=str(image),
                gt="x",
                width=800,
                height=400,
            )
        ],
        manifest_path(data, "s1"),
    )
    records = [
        PredictionRecord(
            scene_id="station_dkw",
            run_id="r",
            model=m,
            mode=Mode.SINGLE_SHOT,
            latency_ms=1,
            annotation=_doc(),
        )
        for m in ("a", "b")
    ] + [
        PredictionRecord(
            scene_id="station_dkw",
            run_id="r",
            model="c",
            mode=Mode.SINGLE_SHOT,
            latency_ms=1,
            raw_text=json.dumps(STATION),
            parse_error="schema: ...",
        )
    ]
    written = build_drafts(records, data_dir=data, split="s1", models=["a", "b", "c"], run_id="r")
    assert len(written) == 1
    draft = SceneAnnotation.model_validate_json(written[0].read_bytes())
    assert draft.meta["review"]["models"] == ["a", "b", "c"]  # c recovered leniently
    assert build_drafts(records, data_dir=data, split="s1", models=["a"], run_id="r") == []


def test_repair_truncated_json():
    from rail_vision_bench.review.consensus import repair_truncated_json

    full = json.dumps(STATION)
    cut = full[: len(full) // 2]
    repaired = repair_truncated_json("Sure: " + cut)
    assert repaired is not None
    assert repaired["schema_version"] == "v0"
    assert 0 < len(repaired["topology"]["nodes"]) <= len(STATION["topology"]["nodes"])
    assert repair_truncated_json('{"a": "unterminated [ { string') is None
    assert repair_truncated_json('{"a": "x\\"}"}') == {"a": 'x"}'}
    assert repair_truncated_json("no json") is None
    doc = lenient_document(cut, 1.0, _doc())
    assert doc is not None
    assert doc.topology.nodes


def test_merge_single_model_keeps_everything():
    draft = merge([("only", _doc())], run_id="r")
    review = draft.meta["review"]
    assert len(draft.topology.nodes) == len(STATION["topology"]["nodes"])
    assert review["suggestions"] == {}
    assert all(e["status"] == "consensus" for e in review["elements"].values())
