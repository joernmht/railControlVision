# Architecture

How a source image becomes a scored, validated document, which module does what, and which
contracts (run directory, wire protocol, configs) the pieces agree on. The status column at the
end says what exists today; everything else is design that the stubs' docstrings spell out.

## Pipeline

```
                 ┌────────────────────────── single-shot ───────────────────────────┐
                 │  provider.complete(VisionRequest{prompt=single_shot, image})     │
 ingest ─────────┤                                                                  ├─▶ validator ─▶ eval ─▶ tracking / report
 (image, video,  │  agentic (LangGraph):                                            │   (schema +     (match,   (MLflow or null,
  screen,        │   START ─▶ planner ─▶ reader ─▶ interpreter ─▶ geometer ─▶ builder ─▶ critic ─┐   rules)   metrics,  leaderboard)
  preprocess,    │              ▲                                                      │           parquet)
  quality)       │              └──────────── critic_should_continue ─────────────────┘ (planner | END)
                 │                                                                  │
                 │  deep-agent variant: deepagents.create_deep_agent with the tools │
                 └──────────────────────────────────────────────────────────────────┘
```

- **Ingest** (`rail_vision_bench.ingest`) turns a photo, a video/RTSP stream, a screen capture
  or a synthetic PNG into a Pillow image, optionally perspective-corrected and cropped, and
  measures `QualityMetrics` (blur variance, glare fraction) that feed the difficulty tiers.
- **Single-shot** (`agents/single_shot.py`) sends the image with the packaged `single_shot`
  prompt and schema v0 as `response_schema` to one provider and parses one JSON document.
- **Agentic loop** (`agents/graph.py`, real wiring) is a LangGraph `StateGraph` over
  `AgentState`: the planner chooses regions, the reader transcribes crops (labels, `12a`/`12b`,
  `+`/`-`, `Hp2`), the interpreter maps symbols to kinds and states, the geometer connects
  elements into edges and ports, the builder assembles the `SceneAnnotation`, and the critic
  runs the `validate` tool, compares a re-rendering with the image and either sets `done` or
  writes a `critique` for the next round. `critic_should_continue` routes to `planner` until
  `done` or `round >= max_rounds`. Every node is bound to the provider through a closure and
  raises `NotImplementedError` in the skeleton; the graph still compiles and the tests assert its
  shape.
- **Deep-agent variant** (`agents/deep.py`) will hand the four tools to `deepagents` for a
  planning/sub-agent baseline.
- **Validator** (`graph/validator.py`) is the same code for ground truth, predictions, the
  `validate` tool, `bench validate` and `POST /validate`; see [`schema.md`](schema.md).
- **Eval** (`eval/`) matches predicted elements to ground truth per family (Hungarian assignment
  over a label/kind/geometry cost), computes the metrics and aggregates them to parquet.
- **Tracking / report** logs params, metrics and artifacts to a `Tracker` and renders the
  leaderboard from the Jinja2 template.

## Module map

| Module | Role | Status |
| --- | --- | --- |
| `rail_vision_bench.__init__` | `__version__`, `SCHEMA_VERSION` | implemented |
| `settings` | `Settings` (pydantic-settings, `SecretStr` keys, `RVB_*` knobs), `get_settings()` | implemented |
| `config` | `ProviderName`, `Mode`, `TrackerKind`, `ModelConfig`/`ModelCatalogue`, `TaskConfig`, `RunConfig`, loaders, `apply_overrides`, `resolve_run` | implemented |
| `cli` | typer app `bench` (see the README table) | implemented; stubbed stages exit 3 |
| `runner` | `RUN_LAYOUT`, `run_dir()`; `run_benchmark`, `evaluate_run`, `build_report` | contract real; orchestration stubbed |
| `schema.models` / `schema.issues` / `schema.export` / `schema.validate` | v0 models and enums; `IssueCode`, `ValidationIssue`, `ValidationReport`; JSON Schema export; jsonschema wrapper | implemented |
| `graph.rules` / `graph.build` / `graph.validator` / `graph.generate` | rules tables; `to_networkx`; rules 1-15; valid-by-construction generator | implemented |
| `providers.base` / `providers.registry` | `ImageInput`, `Usage`, `VisionRequest`, `VisionResponse`, `VisionProvider`; lazy factories, `get_provider` | implemented |
| `providers.claude`, `openai_compat`, `gemini`, `mistral`, `ollama_local`, `litellm_router` | one class each; `complete()` | stubbed (no SDK imported) |
| `agents.state` / `agents.nodes` / `agents.graph` / `agents.prompts` | `AgentState`, `NODE_NAMES`; six nodes + router; `build_agent_graph`; `PROMPT_NAMES`, `load_prompt` | wiring, router and prompts real; nodes stubbed |
| `agents.single_shot` / `agents.deep` | baseline runner; deepagents variant | stubbed |
| `tools.tools` / `tools.mcp_server` | `TOOL_NAMES`, `ToolResult`, `crop`/`read`/`validate`/`render`; `serve_mcp` | `validate_tool` real; rest stubbed |
| `ingest.image` / `video` / `screen` / `preprocess` / `quality` | loaders, `Frame`, capture, OpenCV helpers, `QualityMetrics` + `measure` | `Frame` and `QualityMetrics` real; functions stubbed |
| `synth.generate` / `render` / `augment` | dataset generation, SVG rendering, albumentations presets | stubbed |
| `eval.records` / `matching` / `metrics` / `aggregate` | `PredictionRecord`, `MetricResult`; `Match`, `match_elements`; `METRIC_NAMES` + functions; `aggregate_run` | models and names real; computations stubbed |
| `harness.app` / `models` / `metrics` / `client` | FastAPI app; wire models; Prometheus registry; `HarnessClient` | health/metrics/validate real; frame/stream not served |
| `dataset.manifest` / `splits` / `hub` | `ManifestRow` + JSONL; `assign_partition`, `difficulty_tier`; Hub push/pull | manifest and splits real; hub stubbed |
| `tracking.base` / `mlflow_tracker` | `Tracker`, `NullTracker`, `get_tracker`; `MlflowTracker` | null tracker real; MLflow stubbed |
| `report.render` | `load_template`, `render_leaderboard` | template loads; rendering stubbed |

## Run-directory contract

`bench run` writes one directory `<out_dir>/<run_id>/` (`runner.run_dir(runs_root, run_id)`)
holding exactly the files of `runner.RUN_LAYOUT`:

| Key | File | Written by | Content |
| --- | --- | --- | --- |
| `config` | `run.yaml` | `bench run` | the resolved `RunConfig` (after `--model/--mode/--limit/--seed/--out` overrides and the cross-check against task file and catalogue) |
| `predictions` | `predictions.jsonl` | `bench run` | one `eval.records.PredictionRecord` per line: `scene_id`, `run_id`, `model`, `mode`, `attempt`, `latency_ms`, tokens, `cost_usd`, `raw_text`, parsed `annotation`, `parse_error`, `validation` report |
| `metrics` | `metrics.parquet` | `bench eval` | one `MetricResult` row per (metric, scope): `name`, `value`, `n`, `extra` |
| `summary` | `summary.json` | `bench eval` | the aggregated headline numbers of the run |
| `report` | `report.html` | `bench report` | the rendered leaderboard for this run (a cross-run leaderboard goes to `--out`, default `reports/leaderboard.html`) |

`bench run --dry-run` prints the resolved config as YAML and writes nothing; CI runs it on the
committed example. `RunConfig.task` and `RunConfig.catalogue` are resolved relative to the
current working directory at resolution time and copied as written, so `run.yaml` stays
portable across checkouts.

## Harness wire protocol

`bench serve` runs `harness.app.create_app()` under uvicorn (host and port from
`RVB_HARNESS_HOST` / `RVB_HARNESS_PORT`, default `127.0.0.1:8000`). `harness.client.HarnessClient`
(httpx, async) wraps the endpoints and raises `HarnessError(status_code, detail)` on any non-2xx
answer.

| Endpoint | Request | Response | Today |
| --- | --- | --- | --- |
| `GET /health` | — | `HealthResponse {status: "ok", version, schema_version}` | implemented |
| `GET /metrics` | — | Prometheus text exposition (served by an ASGI mount at `/metrics/`; a bare `/metrics` is redirected there) | implemented |
| `POST /validate?strict=false` | JSON body: a `SceneAnnotation` document | `ValidateResponse {ok, issues: [ValidationIssue]}` | implemented |
| `POST /frame` | multipart: file field `frame`, form field `model` (catalogue name), optional form field `mode` (`single_shot` default, or `agentic`) | `FrameResponse {scene_id, model, mode, annotation?, validation?, latency_ms}` | answers **501** `frame inference is not implemented in the skeleton` (an invalid `mode` is a 422 first) |
| `WS /stream` | client sends binary frames, or `{"type": "control", "action": "end"}` | server sends `{"type": "partial" \| "final" \| "error", "seq": int, "scene_id": str, "annotation"?: SceneAnnotation, "validation"?: ValidationReport, "latency_ms"?: float}` | accepts the socket, then closes with code **1011** `stream inference is not implemented in the skeleton` |

Prometheus metrics live in a dedicated `CollectorRegistry` (`harness.metrics.REGISTRY`) so tests
can build many apps in one process:

| Metric | Type | Labels |
| --- | --- | --- |
| `rvb_frame_latency_seconds` | histogram | `model`, `mode` |
| `rvb_frames_total` | counter | `model`, `mode`, `status` (`ok`, `invalid`, `error`) |
| `rvb_cost_usd_total` | counter | `model` |
| `rvb_tokens_total` | counter | `model`, `direction` (`in`, `out`) |

## Tools, MCP and n8n

The agents use four tools (`tools.TOOL_NAMES`): `crop(image, bbox, zoom)` → PNG bytes,
`read(image, question)` → `ToolResult` with the transcription, `validate(document)` →
`ToolResult` whose `payload` is the `ValidationReport` and whose `errors` are the error messages
(real today), and `render(document)` → PNG bytes of the document drawn in Stelltisch style, so
the critic can compare it with the source. `tools.mcp_server.serve_mcp(host, port, transport)`
will expose the same four callables through a FastMCP server over stdio (local agent hosts,
`langchain-mcp-adapters`) or SSE on `host:port`, which is what an n8n workflow (started with
`make compose-up`, http://localhost:5678) calls to drive the loop from outside Python. Nothing
about n8n is configured in this repository beyond the compose service; workflows are the user's.

## Configuration lifecycle

Three YAML files, three models, loaded relative to the working directory (nothing in the package
resolves repository paths):

- **`configs/models.yaml`** — `ModelCatalogue`: one `ModelConfig` per model (`name`, `provider`
  from `PROVIDER_NAMES`, `model_id`, `supports_tools`, `cost_per_1k_in/out`, `params`). A run
  references models by `name`; `providers.get_provider(model.provider, settings)` builds the
  backend. The committed catalogue has exactly one entry per provider and every price is marked
  `verify against provider docs`.
- **`configs/tasks/<task>.yaml`** — `TaskConfig`: `name`, `description`, `source_kind`, `split`
  (a manifest split name), `prompt` (a `PROMPT_NAMES` entry), `metrics` (a subset of
  `METRIC_NAMES`).
- **`configs/run.example.yaml`** — `RunConfig`: `name`, `task`, `catalogue`, `models`, `mode`,
  `limit`, `seed`, `max_rounds`, `out_dir`, `tracker` (`"null"` quoted, because a bare `null`
  is YAML for None). Command-line overrides (`--model`, `--mode`, `--limit`, `--seed`, `--out`)
  are applied before resolution; unknown model names and missing files fail with exit code 1.

Secrets and hosts never go into these files: `settings.Settings` reads them from the
environment or `.env` (`.env.example` lists all 16 keys).

## Reproducibility inputs

A benchmark number is only comparable when all of these are recorded with it; `run.yaml`,
`predictions.jsonl` and the tracker are where they land:

| Input | Where it comes from |
| --- | --- |
| dependency set | `requirements.lock` (universal, produced by uv 0.8.17 from the verbatim `requirements.txt`) |
| model | catalogue `name` + `model_id` + `params` from `run.yaml`; `provenance.model_id` in every prediction |
| prompt | `TaskConfig.prompt` names a packaged, versioned Markdown file |
| schema version | `SCHEMA_VERSION` / `schema_version: "v0"` in every document; `schema/v0.json` committed |
| randomness | `RunConfig.seed` (scene order, `limit` subset, synthetic generation) |
| agent budget | `RunConfig.max_rounds` |
| data | `manifest.jsonl` rows (DVC-tracked content, see `data/README.md`), `split`, `partition` |
| code | the git SHA of the checkout (to be written into `summary.json` by `bench eval`) |

## Implemented vs stubbed

| Area | Implemented | Stubbed |
| --- | --- | --- |
| Schema | models, export, drift check, jsonschema wrapper, examples, fixtures | — |
| Validation | all 15 rules, strict mode, CLI, harness endpoint, `validate` tool | — |
| Generation | deterministic topology/scene generator (used by hypothesis) | synthetic rendering, augmentation, dataset writer |
| Providers | protocol, models, registry | every `complete()` |
| Agents | state, graph wiring, router, prompts | node bodies, single-shot runner, deep agent |
| Tools | `validate` | `crop`, `read`, `render`, MCP server |
| Ingest | `Frame`, `QualityMetrics` | loaders, capture, preprocessing, `measure` |
| Eval | row models, metric names | matching, metrics, aggregation |
| Harness | health, metrics, validate, client | frame and stream inference |
| Dataset | manifest, partition, difficulty | Hub push/pull |
| Tracking / report | null tracker, factory, template | MLflow tracker, leaderboard rendering |
| Runner / CLI | run-directory contract, dry-run, validate, schema export | run, eval, report, synth generate |
