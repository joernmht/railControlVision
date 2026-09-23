# Architecture

How a source image becomes a scored, validated document, which module does what, and which
contracts (run directory, wire protocol, configs) the pieces agree on. The last section says how
each part is verified.

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

- **Ingest** (`rail_vision_bench.ingest`) turns a photo (PNG, JPEG, WebP, HEIC; EXIF
  orientation applied), a video file or RTSP stream (PyAV, imageio fallback, optional fps
  subsampling), a screen capture (`mss`) or a synthetic PNG into a Pillow image, offers
  perspective correction, crop/zoom and tile-seam detection (OpenCV), and measures
  `QualityMetrics` (blur variance, glare fraction) that feed the difficulty tiers.
- **Single-shot** (`agents/single_shot.py`): `single_shot_attempt` fills the packaged
  `single_shot` prompt (or the task's prompt), sends it with the image and schema v0 as
  `response_schema` to one provider, takes the provider's `parsed` JSON or recovers the object
  from the text with `providers.parsing.extract_json`, stamps it and parses it. It never raises
  on bad output: the result carries `annotation` or `parse_error`, the raw text, usage and
  latency. `run_single_shot` is the raising wrapper (`ParseError`).
- **Agentic loop** (`agents/graph.py`, `agents/nodes.py`) is a LangGraph `StateGraph` over
  `AgentState`. The nodes are coroutines, bound to the provider through a closure, so the
  compiled graph is driven asynchronously; `run_agentic` streams its updates
  (`astream(stream_mode="updates")`) and optionally awaits `on_round` after every critic step.
  Per round: the planner picks regions (bbox per work item; the whole image as fallback), the
  reader crops each region at 2x with the `crop` tool and reads the crops concurrently with the
  `read` tool (labels, `12a`/`12b`, `+`/`-`, `Hp2`), the interpreter maps symbols to kinds and
  states, the geometer connects elements into edges with ports, attachments and track states,
  the builder produces the candidate, and the critic runs the `validate` tool, re-renders a
  schema-valid candidate with the `render` tool, asks the model to compare original and
  rendering, and either sets `done` (no error-level issue and no mismatch) or writes a
  `critique` for the next round. `critic_should_continue` routes to `planner` until `done` or
  `round >= max_rounds`. Nodes never raise on a bad model answer: they fall back to a
  conservative result and append to `errors`; token usage of every call is summed in `usage`.
- **Builder and stamping.** `agents/assembly.assemble_candidate` builds a draft from the
  interpreter's and geometer's output in code: enums are coerced to their vocabulary
  (`unknown` otherwise), coordinates clamped to the image, elements without a valid id or
  attachment dropped, nodes whose ports are not exactly connected repaired, so the draft is
  always shaped like schema v0. The model then returns the complete document from that draft
  (adding routes); its version is kept unless it has more error-level validator issues than the
  draft. `agents/documents.stamp_document` overwrites `schema_version`, `scene_id`, `source` and
  `provenance` (`prediction`, `model_id`) on every model-produced document, in single-shot and
  agentic mode alike, so a prediction always joins its ground truth whatever the model echoed.
  Without a manifest row (harness, ad-hoc calls) the scene id is `frame-<12 hex of SHA-256 of the
  image bytes>` and `source` is derived from the image header (`kind: stream_frame`).
- **Deep-agent variant** (`agents/deep.py`) hands the four tools to
  `deepagents.create_deep_agent`, which plans its own strategy. `VisionChatModel` adapts any
  `VisionProvider` to a LangChain chat model: bound tools are described in the system prompt
  and a `{"tool_calls": [{"name", "args"}]}` answer becomes LangChain tool calls (providers have
  no native tool calling here); usage is summed across calls. Images cannot travel as JSON tool
  arguments, so the tools address them by name in an `ImageWorkspace` (the scene is `"scene"`,
  crops and renderings get new names). `run_deep_agent` returns an `AgenticResult` like
  `run_agentic`. It is a library API only: neither `bench run` nor the harness uses it.
- **Validator** (`graph/validator.py`) is the same code for ground truth, predictions, the
  `validate` tool, `bench validate` and `POST /validate`; see [`schema.md`](schema.md).
- **Eval** (`eval/`) matches predicted elements to ground truth per family (Hungarian assignment
  over a label/kind/geometry cost, `eval/matching.py`), scores one scene with
  `eval/scoring.score_prediction` (a record without a parsed annotation scores as an empty
  prediction), and aggregates to parquet (`eval/aggregate.py`).
- **Tracking / report** logs params, counters and artifacts of `bench run` to a `Tracker` and
  renders the leaderboard from the Jinja2 template as HTML or Markdown.

## Module map

| Module | Role | Status |
| --- | --- | --- |
| `rail_vision_bench.__init__` | `__version__`, `SCHEMA_VERSION` | implemented |
| `settings` | `Settings` (pydantic-settings, `SecretStr` keys, `RVB_*` knobs), `get_settings()` | implemented |
| `config` | `ProviderName`, `Mode`, `TrackerKind`, `ModelConfig`/`ModelCatalogue`, `TaskConfig`, `RunConfig`, loaders, `apply_overrides`, `resolve_run` | implemented |
| `cli` | typer app `bench` (see the README table) | implemented |
| `runner` | `RUN_LAYOUT`, `run_dir`, `new_run_id`, `select_scenes`, `price`; `run_benchmark`, `evaluate_run`, `build_report` | implemented |
| `schema.models` / `schema.issues` / `schema.export` / `schema.validate` | v0 models and enums; `IssueCode`, `ValidationIssue`, `ValidationReport`; JSON Schema export; jsonschema wrapper | implemented |
| `graph.rules` / `graph.build` / `graph.validator` / `graph.generate` | rules tables; `to_networkx`; rules 1-15; valid-by-construction generator | implemented |
| `providers.base` / `providers.registry` | `ImageInput`, `Usage`, `VisionRequest`, `VisionResponse`, `VisionProvider`; lazy factories, `get_provider`, `provider_for_model` | implemented |
| `providers.parsing` / `providers._common` | `extract_json`; image encoding, credential checks, tenacity retry on transient errors, schema instructions | implemented |
| `providers.claude`, `openai_compat`, `gemini`, `mistral`, `ollama_local`, `litellm_router` | one class each; `complete()` over the vendor SDK | implemented; tested against mocked transports only |
| `agents.state` / `agents.nodes` / `agents.graph` / `agents.prompts` | `AgentState`, `NODE_NAMES`; six async nodes + router; `build_agent_graph`, `run_agentic`, `AgenticResult`; `PROMPT_NAMES`, `load_prompt` | implemented |
| `agents.documents` / `agents.assembly` | scene ids, default source, `stamp_document`, `parse_scene`, `add_usage`; `assemble_candidate` | implemented |
| `agents.single_shot` / `agents.deep` | `single_shot_attempt`, `run_single_shot`; `VisionChatModel`, `ImageWorkspace`, `build_deep_agent`, `run_deep_agent` | implemented |
| `tools.tools` / `tools.mcp_server` | `TOOL_NAMES`, `ToolResult`, `crop`/`read`/`validate`/`render`; `build_mcp_server`, `serve_mcp` | implemented |
| `ingest.image` / `video` / `screen` / `preprocess` / `quality` | `load_image`; `Frame`, `iter_frames`; `capture_screen`; `perspective_correct`, `crop_zoom`, `clip_bbox`, `detect_tile_seams`; `QualityMetrics`, `measure` | implemented |
| `synth.generate` / `render` / `augment` | `generate_dataset`; `layout_scene`, `render_svg`, `render_png`; `PRESETS`, `build_augmentation`, `augment_image` | implemented |
| `eval.records` / `matching` / `metrics` / `scoring` / `aggregate` | `PredictionRecord`, `MetricResult`; `Match`, `match_elements`, `match_all`; `METRIC_NAMES` + functions; `score_prediction`; `aggregate_run`, `combine_results`, `load_metrics` | implemented |
| `harness.app` / `models` / `metrics` / `client` | FastAPI app and `InferenceService`; wire models; Prometheus registry; `HarnessClient` (health, validate, `post_frame`) | implemented |
| `dataset.manifest` / `splits` / `hub` | `ManifestRow` + JSONL; `assign_partition`, `difficulty_tier`; `check_release`, `push_split`, `pull_split` | implemented |
| `tracking.base` / `mlflow_tracker` | `Tracker`, `NullTracker`, `get_tracker`; `MlflowTracker` | implemented |
| `report.render` | `load_template`, `render_leaderboard` (HTML or Markdown) | implemented |

## Run-directory contract

`bench run` writes one directory `<out_dir>/<run_id>/` (`runner.run_dir(runs_root, run_id)`,
`run_id = <name>-<YYYYmmddTHHMMSSZ>` from `runner.new_run_id`) holding the files of
`runner.RUN_LAYOUT`; an existing run directory is an error:

| Key | File | Written by | Content |
| --- | --- | --- | --- |
| `config` | `run.yaml` | `bench run` | the resolved `RunConfig` (after `--model/--mode/--limit/--seed/--out` overrides and the cross-check against task file and catalogue) |
| `predictions` | `predictions.jsonl` | `bench run` | one `eval.records.PredictionRecord` per (model, scene), appended as each attempt finishes: `scene_id`, `run_id`, `model`, `mode`, `attempt`, `latency_ms`, `tokens_in`, `tokens_out`, `cost_usd`, `raw_text`, parsed `annotation`, `parse_error`, `validation` report |
| `metrics` | `metrics.parquet` | `bench eval` | metric rows, columns below |
| `summary` | `summary.json` | `bench eval` | run identity and reproducibility inputs plus the run-level value of every metric, fields below |
| `report` | `report.html` | `bench report` | the leaderboard of this run alone (the cross-run leaderboard goes to `--out`, default `reports/leaderboard.html`) |

**`bench run`** (`runner.run_benchmark`) resolves the config, reads `RunConfig.manifest` and
selects scenes with `runner.select_scenes`: the rows of the task's split, sorted by `scene_id`,
shuffled with `random.Random(seed)` and cut to `limit`, so `seed` fixes both the order and the
subset. Every model then runs every scene in turn, with `single_shot_attempt` (prompt
`task.prompt`) or `run_agentic` (`max_rounds`). The manifest's `source_kind`, `image`, `width`
and `height` become the stamped `source`; images other than PNG/JPEG/WebP are re-encoded as
PNG. Each attempt is validated (the parsed annotation, or the raw document when it failed the
schema) and appended at once. Any exception during an attempt (provider error, unreadable
image) is recorded as that record's `parse_error` (`<ExceptionType>: <message>`) instead of
aborting the run. `cost_usd` comes from `runner.price`: the provider-reported cost when there is
one, else the catalogue's `cost_per_1k_in/out`, else null. The tracker receives the run config
plus `task_split`, `task_prompt`, `model_ids` and `scenes` as params, the counters
`predictions`, `parsed`, `parse_failures`, `valid`, `cost_usd_total`, and `run.yaml` and
`predictions.jsonl` as artifacts.

**`bench eval`** (`runner.evaluate_run`) reloads `run.yaml`, the task and the manifest, loads
the ground truth of every predicted scene of the task's split and calls
`eval.aggregate.aggregate_run` with the task's `metrics` (all of `METRIC_NAMES` when the task
lists none). Only the latest `attempt` per (model, mode, scene) is scored; latency and cost use
every record. `metrics.parquet` has one row per (model, mode, scope, scene_id, name):

| Column | Content |
| --- | --- |
| `model`, `mode` | catalogue name and mode of the records |
| `scope` | `scene` for a per-scene row, `run` for the aggregate of a model and mode |
| `scene_id` | the scene; null on run rows |
| `name` | one of `METRIC_NAMES` (`detection_prf1`, `label_cer`, `label_wer`, `state_accuracy`, `route_prf1`, `topology_agreement`, `calibration_brier`, `calibration_ece`, `latency_p50_ms`, `latency_p95_ms`, `cost_usd`) |
| `value` | the headline number |
| `n` | the number of items it was computed over; `n = 0` marks the value as undefined |
| `extra` | JSON string of the raw counts and breakdown (per-family PRF1, per-field state accuracy, calibration bins, ...) |

Each metric yields exactly one row per scene; breakdowns live in `extra`, not in extra rows.
Run rows are micro-averaged from the summed counts in `extra` (`combine_results`), so they equal
the metric over the pooled items. Latency and cost rows exist only at run scope. The value and
`extra` keys of every metric are tabulated in the module docstring of `eval/metrics.py`.

`summary.json` holds `run_id`, `name`, `task`, `split`, `mode`, `models`, `seed`, `limit`,
`max_rounds`, `scenes`, `predictions`, `parse_failures`, `package_version`, `schema_version`,
`git_sha` (`git rev-parse HEAD` of the working directory, null outside git), `evaluated_at` and
`metrics` (`{"<model>/<mode>": {"<metric>": value}}` from the run rows).

**`bench report`** (`runner.build_report`) needs `summary.json` and `metrics.parquet` in every
run directory, renders each run's `report.html` and the combined leaderboard (Markdown when
`--out` ends in `.md`, else HTML) from the run rows.

`bench run --dry-run` prints the resolved config as YAML and writes nothing; CI runs it on the
committed example. `RunConfig.task`, `RunConfig.catalogue` and `RunConfig.manifest` are
resolved relative to the current working directory and copied as written, so `run.yaml` stays
portable across checkouts; `bench eval` and `bench report` must run from the same directory as
`bench run`.

## Harness wire protocol

`bench serve` runs `harness.app.create_app()` under uvicorn (host and port from
`RVB_HARNESS_HOST` / `RVB_HARNESS_PORT`, default `127.0.0.1:8000`). With no arguments the app
reads `configs/models.yaml` relative to the working directory on the first inference request
and builds providers through the registry (`provider_for_model`), one per catalogue name,
cached across requests. `harness.client.HarnessClient` (httpx, async) wraps `/health`,
`/validate` and `/frame` and raises `HarnessError(status_code, detail)` on any non-2xx answer.

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | — | `HealthResponse {status: "ok", version, schema_version}` |
| `GET /metrics` | — | Prometheus text exposition (served by an ASGI mount at `/metrics/`; a bare `/metrics` is redirected there) |
| `POST /validate?strict=false` | JSON body: a `SceneAnnotation` document | `ValidateResponse {ok, issues: [ValidationIssue]}` |
| `POST /frame` | multipart: file field `frame`, form field `model` (catalogue name), optional form field `mode` (`single_shot` default, or `agentic`) | see below |
| `WS /stream` | connect to `/stream?model=<name>&mode=single_shot\|agentic`; client sends binary frames, or `{"type": "control", "action": "end"}` | see below |

`POST /frame` answers:

| Status | When |
| --- | --- |
| 200 | `FrameResponse {scene_id, model, mode, annotation?, validation?, latency_ms}`; `annotation` is null when the output did not parse, `validation` reports on the annotation or on the raw document (null when the model produced no JSON at all) |
| 404 | `model` is not in the catalogue |
| 422 | the frame is not a decodable image, or `mode` is not a `Mode` |
| 502 | the provider call raised (`inference failed: <Type>: <message>`) |
| 503 | the catalogue file is missing or invalid, or the provider cannot be built (e.g. an unknown provider name) |

PNG, JPEG and WebP frames are passed through; any other format Pillow decodes is converted to
PNG. The scene id is `frame-<12 hex of the SHA-256 of the image bytes>`. Providers create their
SDK client on the first call, so a missing API key surfaces as a 502 naming the environment key,
not as a 503.

`WS /stream`, after the socket is accepted:

- an unknown `model`, an invalid `mode` or an unavailable catalogue produces one
  `{"type": "error", "seq": 0, "scene_id": "", "detail"}` message, then a close with code
  **1008**;
- every binary message is one frame. In agentic mode the server first sends one `partial`
  message per round (after each critic step) with a `round` field, the round's `validation` and,
  when the candidate passes the schema, its `annotation`; then, in both modes, one `final`
  message `{"type": "final", "seq", "scene_id", "latency_ms", "annotation"?, "validation"?}`;
- a frame that fails (undecodable, provider error, ...) produces
  `{"type": "error", "seq", "scene_id": "", "detail"}` and the stream continues;
- a text message other than the end message produces an error message and the stream
  continues;
- `{"type": "control", "action": "end"}` closes the socket with code **1000**;
- `seq` counts every received message (frames and rejected text messages) from 0.

Every served frame updates the Prometheus metrics, which live in a dedicated
`CollectorRegistry` (`harness.metrics.REGISTRY`) so tests can build many apps in one process:

| Metric | Type | Labels |
| --- | --- | --- |
| `rvb_frame_latency_seconds` | histogram | `model`, `mode` |
| `rvb_frames_total` | counter | `model`, `mode`, `status` (`ok`: parsed and valid, `invalid`, `error`: the provider call raised) |
| `rvb_cost_usd_total` | counter | `model` (provider-reported cost, else catalogue prices) |
| `rvb_tokens_total` | counter | `model`, `direction` (`in`, `out`) |

## Tools, MCP and n8n

The agents use four tools (`tools.TOOL_NAMES`): `crop(image, bbox, zoom)` → PNG bytes,
`read(image, question)` → `ToolResult` whose payload holds the transcription (`text`),
`confidence` and usage, `validate(document)` → `ToolResult` whose `payload` is the
`ValidationReport` and whose `errors` are the error messages, and `render(document)` → PNG bytes
of the document drawn in Stelltisch style (through `synth.render`), so the critic can compare it
with the source.

`tools.mcp_server.build_mcp_server` registers the same four callables on a `FastMCP` server (the
`mcp` package that `langchain-mcp-adapters` builds on); `serve_mcp` (`bench mcp`) runs it over
stdio (local agent hosts) or SSE on `host:port` (default `127.0.0.1:8765`), which is what an n8n
workflow (started with `make compose-up`, http://localhost:5678) calls to drive the tools from
outside Python. Images go in as base64 strings (a `data:` URL prefix is tolerated) and come
back as MCP PNG image content; documents are plain JSON objects. `read` needs a model: start the
server with `--provider` (a registry name such as `anthropic`) and `--model` (the SDK model id).
Without them the server still starts and `read` answers with an error result, so `crop`,
`validate` and `render` stay usable. Nothing about n8n is configured in this repository beyond
the compose service; workflows are the user's.

## Configuration lifecycle

Three YAML files, three models, loaded relative to the working directory (nothing in the package
resolves repository paths):

- **`configs/models.yaml`** — `ModelCatalogue`: one `ModelConfig` per model (`name`, `provider`
  from `PROVIDER_NAMES`, `model_id`, `supports_tools`, `cost_per_1k_in/out`, `params`). A run
  references models by `name`; `providers.registry.provider_for_model(model, settings)` builds
  the backend, letting `params.base_url` (`openai`, `litellm`) or `params.host` (`ollama`)
  override `OPENAI_BASE_URL` / `OLLAMA_HOST` for that entry. The committed catalogue has exactly
  one entry per provider and every price is marked `verify against provider docs`.
- **`configs/tasks/<task>.yaml`** — `TaskConfig`: `name`, `description`, `source_kind`, `split`
  (a manifest split name), `prompt` (a `PROMPT_NAMES` entry), `metrics` (a subset of
  `METRIC_NAMES`).
- **`configs/run.example.yaml`** — `RunConfig`: `name`, `task`, `catalogue`, `manifest`
  (default `data/manifest.jsonl`; the example points at `data/synthetic/manifest.jsonl`, which
  `make synth` writes), `models`, `mode`, `limit`, `seed`, `max_rounds`, `out_dir`, `tracker`
  (`"null"` quoted, because a bare `null` is YAML for None). Command-line overrides (`--model`,
  `--mode`, `--limit`, `--seed`, `--out`) are applied before resolution; unknown model names and
  missing task or catalogue files fail with exit code 1. The manifest is read only when the run
  executes, so a dry run resolves without any data present.

Secrets and hosts never go into these files: `settings.Settings` reads them from the
environment or `.env` (`.env.example` lists all 16 keys).

## Reproducibility inputs

A benchmark number is only comparable when all of these are recorded with it; `run.yaml`,
`predictions.jsonl`, `summary.json` and the tracker are where they land:

| Input | Where it comes from |
| --- | --- |
| dependency set | `requirements.lock` (universal, produced by uv 0.8.17 from the verbatim `requirements.txt`) |
| model | catalogue `name` + `model_id` + `params` (the tracker's `model_ids`); `provenance.model_id` stamped into every prediction |
| prompt | `TaskConfig.prompt` names a packaged, versioned Markdown file |
| schema version | `SCHEMA_VERSION` / `schema_version: "v0"` in every document and in `summary.json`; `schema/v0.json` committed |
| package version | `package_version` in `summary.json` |
| randomness | `RunConfig.seed` (scene order and the `limit` subset); `bench synth generate --seed` (byte-identical output) |
| agent budget | `RunConfig.max_rounds` |
| data | `RunConfig.manifest` rows (DVC-tracked content, see `data/README.md`), `split`, `partition` |
| code | `git_sha` in `summary.json`, written by `bench eval` (the checkout `bench eval` runs in) |

## Testing coverage

No test makes a network call or needs a provider key; the `requires_network` marker exists for
tests that would, and none uses it today.

| Area | Verified by |
| --- | --- |
| Schema, validation | models, export and drift check, all 15 rules, strict mode; hypothesis properties over the valid-by-construction generator and rule-specific mutators; invalid and strict fixtures |
| Providers | the real SDK clients of Anthropic, OpenAI, Gemini, Mistral and Ollama over `httpx.MockTransport` (the wire payload each SDK sends, response parsing, retry on transient errors); LiteLLM by replacing `litellm.acompletion`. Model behaviour is not tested |
| JSON recovery | `extract_json` on fenced, prose-wrapped, nested and truncated text |
| Agents | single-shot, the six nodes end to end, assembly and the deep agent, all with fake providers |
| Tools, MCP | crop/read/validate/render; MCP tool registration and calls without running a transport |
| Ingest | image formats incl. HEIC, a small generated video, preprocessing, quality; screen capture with `mss` faked (the real capture test is `requires_display` and skipped without `DISPLAY`); no RTSP stream |
| Synthetic data | layout, rendering, augmentation presets, generation determinism |
| Eval | matching, every metric, scoring, aggregation and micro-averaging |
| Runner, CLI | run → eval → report on a tiny synthetic split with a fake provider answering ground truth, junk or an exception, so the metric values are known exactly |
| Harness | every endpoint including `/frame` and `/stream` over Starlette's `TestClient` with fake providers; the client over `ASGITransport` |
| Hub | `push_split` / `pull_split` with `huggingface_hub` replaced by in-process fakes |
| Tracking | `MlflowTracker` against a local `sqlite:///` store, not the compose server |
| Report | ranking, HTML and Markdown rendering, escaping |
| Stubs | `tests/unit/test_stubs.py` discovers stub-shaped functions by parsing the package (currently none) and asserts that no `NotImplementedError` hides outside a stub |
