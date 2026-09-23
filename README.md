# rail-vision-bench

A benchmark that measures how well **off-the-shelf generative AI** (VLM/LLM APIs and local
models, single-shot or agentic) turns **railway-control imagery** — track-diagram panels
(Gleisbild/Stelltisch), ESTW/CTC screens, control-room camera or screen-capture streams — into
**structured, validated topology + state** (tracks, switches, signals, derailers, routes) in
real time. The header of [`requirements.txt`](requirements.txt) is the authoritative statement of
that aim; this file paraphrases it.

Python package `rail_vision_bench`, command line `bench`, Python 3.11 or 3.12.

## Status

The full pipeline is implemented: `bench run` → `bench eval` → `bench report`, the six
providers, the single-shot baseline, the six-node agentic loop and the deep-agent variant, the
four tools and their MCP server, ingest (image, video/RTSP, screen, preprocessing, quality),
synthetic rendering and augmentation, harness inference (`POST /frame`, `WS /stream`), the
MLflow tracker, Hub push/pull and leaderboard rendering. The package contains no
`NotImplementedError` stub.

What the test suite does **not** exercise against live services:

| Area | How it is tested |
| --- | --- |
| Provider `complete()` (all six) | Real SDK clients over mocked transports (`httpx.MockTransport`; `litellm.acompletion` replaced). No real API call is made in CI: the wire payloads are checked, model behaviour is not. |
| Agents, runner, harness inference, MCP `read` | Fake providers with canned answers. |
| Video | A small generated file; no RTSP stream. |
| Screen capture | `mss` is faked; the one real capture test is marked `requires_display` and skipped without `DISPLAY`. |
| Hugging Face Hub | `HfApi` / `snapshot_download` replaced by in-process fakes; nothing is uploaded or downloaded. |
| MLflow tracker | A local `sqlite:///` tracking store, not the compose server. |

`tests/unit/test_stubs.py` stays as a guard: it discovers every stub-shaped function by parsing
the package and asserts that it raises `NotImplementedError` with a docstring and a message
naming its intent. It currently finds none; a stub added later is covered automatically.

## Quickstart

```sh
uv venv --python 3.12 .venv
uv pip sync requirements.lock          # the locked runtime set (uv targets ./.venv)
uv pip install --no-deps -e .          # the package itself, editable
uv pip install --group typing          # dev-only typeshed stubs for mypy
cp .env.example .env                   # fill in the keys you use; .env is gitignored
make check                             # lint + typecheck + schema-check + validate-examples + test
```

`make install` runs the same steps (plus `pre-commit install`); `source .venv/bin/activate` is
optional because every Make target calls the venv's binaries directly. Then:

```sh
bench --help
bench validate --strict schema/examples/v0/*.json
bench run --config configs/run.example.yaml --dry-run
make compose-up                        # n8n on :5678 and MLflow on :5000 (needs Docker)
```

End to end on synthetic data:

```sh
make synth                             # bench synth generate --out data/synthetic --n 10 --seed 0
bench run --config configs/run.example.yaml        # prints "run written to runs/<run_id>"
bench eval runs/<run_id>               # metrics.parquet + summary.json
bench report runs/<run_id>             # reports/leaderboard.html + runs/<run_id>/report.html
```

`configs/run.example.yaml` reads `manifest: data/synthetic/manifest.jsonl` (split
`synthetic_clean`, the split of `configs/tasks/panel_topology.yaml`) and runs the catalogue
entry `claude-opus-5`, so `bench run` needs `ANTHROPIC_API_KEY` in the environment or `.env`
(without it the run still completes, but every record's `parse_error` names the missing key).
Pick another catalogue model with `--model NAME`.

## Repository layout

| Path | What lives there |
| --- | --- |
| `LICENSE`, `NOTICE` | Apache-2.0 license text and the attribution notice. |
| `requirements.txt` | The runtime dependency spec, committed verbatim; setuptools reads it as dynamic metadata. |
| `requirements.lock` | Universal lock produced from `requirements.txt` (see below); what CI and `make install` install. |
| `pyproject.toml` | Build (setuptools), extras, `bench` entry point, `[dependency-groups] typing`, ruff/mypy/pytest/coverage config. |
| `Makefile` | Every developer and CI step as a target (`make help` lists them). |
| `.github/workflows/ci.yml` | One job, Python 3.11/3.12 matrix, uv 0.8.17. |
| `src/rail_vision_bench/` | The package (src layout). Subpackages below. |
| `src/rail_vision_bench/schema/` | Schema v0 models, issue codes, JSON Schema export and jsonschema wrapper. |
| `src/rail_vision_bench/graph/` | Rules tables, networkx build, semantic validator, deterministic generator. |
| `src/rail_vision_bench/providers/` | `VisionProvider` protocol, request/response models, lazy registry, the six SDK-backed providers. |
| `src/rail_vision_bench/providers/parsing.py` | `extract_json`: recovers the JSON object from fenced or prose-wrapped model text (used by every provider and agent node). |
| `src/rail_vision_bench/providers/_common.py` | Shared provider helpers: image encoding, credential checks, tenacity retry on transient errors, schema instructions. |
| `src/rail_vision_bench/agents/` | LangGraph state, the six async nodes and router, graph wiring and `run_agentic`, single-shot baseline, deep-agent variant, prompt loader. |
| `src/rail_vision_bench/agents/documents.py` | Scene ids from image bytes, default `source`, `stamp_document` (code-owned bookkeeping fields), non-raising parse, usage sums. |
| `src/rail_vision_bench/agents/assembly.py` | `assemble_candidate`: deterministic schema-v0 draft from the interpreter's and geometer's output. |
| `src/rail_vision_bench/prompts/` | Versioned prompts shipped as package data (`single_shot.md`, `agentic/*.md`). |
| `src/rail_vision_bench/tools/` | The four agent tools (`crop`, `read`, `validate`, `render`) and the FastMCP server. |
| `src/rail_vision_bench/ingest/` | Image/video/screen ingestion, preprocessing and quality metrics (named `ingest`, not `io`). |
| `src/rail_vision_bench/synth/` | Synthetic dataset generation, tile layout and SVG/PNG rendering, albumentations presets. |
| `src/rail_vision_bench/eval/` | Prediction/metric row models, Hungarian matching, metrics, parquet aggregation. |
| `src/rail_vision_bench/eval/scoring.py` | `score_prediction`: matches one prediction to its ground truth and runs the per-scene metrics. |
| `src/rail_vision_bench/harness/` | FastAPI real-time harness (validate, frame, stream), Prometheus registry, wire models, async client. |
| `src/rail_vision_bench/dataset/` | Manifest JSONL, dev/test partition, difficulty tiers, Hub push/pull (named `dataset`, not `data`). |
| `src/rail_vision_bench/tracking/` | Tracker protocol, `NullTracker`, `MlflowTracker`. |
| `src/rail_vision_bench/report/` | Leaderboard template (package data) and HTML/Markdown rendering. |
| `schema/v0.json`, `schema/examples/v0/` | The published JSON Schema and two strict-valid example documents. |
| `configs/` | `run.example.yaml`, `models.yaml` (catalogue), `tasks/panel_topology.yaml`. |
| `data/` | DVC data plane; only `manifest.example.jsonl` and `README.md` are committed. |
| `docker/compose.yaml` | n8n and MLflow for local runs. |
| `docs/` | `schema.md` (human spec of v0) and `architecture.md` (pipeline, run dirs, wire protocol). |
| `tests/` | `unit/`, `property/` (hypothesis), `fixtures/` (invalid and strict documents), `strategies.py`. |
| `notebooks/` | Exploration notebooks (outputs stripped, excluded from lint/type checks). |

## Command line

`bench` exits with **0** on success, **1** on validation errors, schema drift or a bad config,
manifest or run directory (this includes runtime errors such as a missing manifest or image, an
existing run directory, an unevaluated run passed to `bench report` or an unknown augmentation
preset) and **2** on a usage error (typer, including a path argument that does not exist).

| Command | Purpose | Exit codes |
| --- | --- | --- |
| `bench --version` | Print the package version. | 0 |
| `bench validate FILES... [--strict] [--schema PATH] [--json]` | Validate documents against schema v0 and the semantic rules; table or one JSON object `{file: report}`. | 0 / 1 / 2 |
| `bench schema export [--out PATH] [--check]` | Write `schema/v0.json` from the models, or check the committed file for drift. | 0 / 1 / 2 |
| `bench run --config PATH [--model NAME]... [--mode] [--limit] [--seed] [--out] [--dry-run]` | Resolve a run config (task file, catalogue, model names) and, without `--dry-run`, execute it into `<out_dir>/<run_id>/`. | 0 / 1 / 2 |
| `bench eval RUN_DIR` | Score a run into `metrics.parquet` and `summary.json`. | 0 / 1 / 2 |
| `bench report RUN_DIRS... [--out PATH]` | Render a leaderboard of evaluated runs to `--out` (default `reports/leaderboard.html`; Markdown when the path ends in `.md`, else HTML) and write each run directory's `report.html`. | 0 / 1 / 2 |
| `bench synth generate --out DIR [--n] [--seed] [--augment PRESET]` | Generate synthetic panels with ground truth and a manifest; `--augment` takes `default`, `phone` or `cctv`. | 0 / 1 / 2 |
| `bench serve [--host] [--port] [--reload]` | Serve the harness (defaults from `RVB_HARNESS_HOST` / `RVB_HARNESS_PORT`). | 0 / 2 |
| `bench mcp [--transport stdio\|sse] [--host] [--port] [--provider NAME] [--model ID]` | Serve `crop`, `read`, `validate`, `render` over MCP (SSE default bind `127.0.0.1:8765`); `read` needs `--provider` (registry name, e.g. `anthropic`) and `--model` (SDK model id). | 0 / 2 |

A failed provider call does not abort `bench run`: it is recorded as that attempt's
`parse_error` in `predictions.jsonl`.

`--strict` promotes `STATE_MISSING`, `GEOM_OUT_OF_BOUNDS` and `ROUTE_SWITCH_MISSING` from
warnings to errors; CI validates the examples in strict mode.

## Schema and documentation

- [`schema/v0.json`](schema/v0.json) — the published JSON Schema (draft 2020-12) for ground
  truth and predictions; generated, never hand-edited (`make schema`, checked by `make schema-check`).
- [`docs/schema.md`](docs/schema.md) — the human specification of v0: element kinds with the
  German glossary, ports and degrees, DKW/EKW path families, state vocabularies, the route
  walker, every issue code, strict mode and the versioning policy.
- [`docs/architecture.md`](docs/architecture.md) — pipeline, module map, run-directory contract,
  metrics table, harness wire protocol, MCP/n8n integration, configuration lifecycle,
  reproducibility inputs and how each part is tested.
- [`data/README.md`](data/README.md) — the DVC data plane, the manifest contract, splits,
  partitions, difficulty tiers, synthetic generation, Hub publishing and the
  PII/consent/licensing policy.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — developer workflow and the tooling rules that keep the
  tree green.

## Requirements and the lock

`requirements.txt` is the single source of truth for runtime dependencies and is committed
verbatim; `pyproject.toml` reads it as dynamic metadata, so the wheel carries exactly those
requirements. The only sanctioned way to touch the lock is

```sh
uv pip compile requirements.txt -o requirements.lock --universal --python-version 3.11
```

(`make lock`), run in place so existing pins are kept, followed by committing
`requirements.lock`. `make lock-check` runs the same command and fails on any diff; CI runs it
on the 3.12 leg with uv 0.8.17, the version that produced the lock.

The commented-out blocks of `requirements.txt` are exposed as extras: `gpu` (torch,
torchvision, transformers, accelerate, qwen-vl-utils), `vllm`, `s3` (`dvc[s3]`), `wandb` and
`labeling` (label-studio), for example `uv pip install -e ".[gpu]"`. The three typeshed stub
packages mypy needs live in the PEP 735 `[dependency-groups] typing` table (`uv pip install
--group typing`); they are dev-only and are not part of the lock or the wheel.

Note that `pytest`, `ruff`, `mypy`, `pre-commit` and `jupyterlab` are runtime dependencies
because `requirements.txt` lists them; the `Private :: Do Not Upload` classifier makes an
accidental upload to PyPI impossible.

## Parts of the stack that are not pip packages

- **n8n** (orchestration UI) and an **MLflow tracking server**: `make compose-up` starts
  `docker/compose.yaml` (n8n on http://localhost:5678, MLflow on http://localhost:5000);
  `make compose-down` stops them. Set `tracker: mlflow` in a run config and point
  `MLFLOW_TRACKING_URI` at the compose server (`http://localhost:5000`) or at a database store
  such as `sqlite:///mlflow.db`. MLflow 3 refuses the file store (`./mlruns`, `file:` URIs)
  unless `MLFLOW_ALLOW_FILE_STORE=true` is set; with the variable unset MLflow 3 defaults to
  `sqlite:///mlflow.db` in the working directory.
- **ffmpeg**: PyAV and OpenCV wheels bundle their own codecs, but RTSP/video work on a host is
  more robust with a system ffmpeg (`apt install ffmpeg`, `brew install ffmpeg`).
- **libcairo** for `cairosvg` (SVG → PNG in the synthetic pipeline): `apt install libcairo2`
  on Debian/Ubuntu, `brew install cairo` on macOS. GitHub's `ubuntu-latest` already ships it.

The project has no CI badge on purpose: the workflow is the contract, `make check` reproduces
it locally on either interpreter (`make check VENV=/path/to/venv311` targets another venv).

## Schema versioning

`schema_version` is the constant `"v0"`. v0 is mutable until the first dataset release and is
kept in sync with the pydantic models by `bench schema export --check` (pre-commit, CI and a
unit test). Any breaking change after the release produces `schema/v1.json` with new models
and a new literal, so documents can never silently cross versions. Details in
[`docs/schema.md`](docs/schema.md#versioning-policy).

## License

Licensed under the [Apache License, Version 2.0](LICENSE); see [`NOTICE`](NOTICE). Unless
you explicitly state otherwise, any contribution you submit for inclusion is licensed under the
same terms, without additional conditions (Apache-2.0 §5).

The license covers the code, schemas, prompts, configs and documentation. It does **not**
relicense third-party imagery referenced by a dataset manifest: each manifest row carries its
own `license` (see the [licensing policy](data/README.md#pii-consent-and-licensing-policy)).
