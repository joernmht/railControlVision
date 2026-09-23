# rail-vision-bench

A benchmark that measures how well **off-the-shelf generative AI** (VLM/LLM APIs and local
models, single-shot or agentic) turns **railway-control imagery** — track-diagram panels
(Gleisbild/Stelltisch), ESTW/CTC screens, control-room camera or screen-capture streams — into
**structured, validated topology + state** (tracks, switches, signals, derailers, routes) in
real time. The header of [`requirements.txt`](requirements.txt) is the authoritative statement of
that aim; this file paraphrases it.

Python package `rail_vision_bench`, command line `bench`, Python 3.11 or 3.12.

## Status: skeleton

This is the first commit: a coherent, installable, lint/type/test-clean **skeleton**. Every
benchmark stage that would need a model call, an image pipeline or a metric computation is an
honest stub that raises `NotImplementedError` with a one-line statement of intent; nothing
fakes a result. What actually works today:

| Implemented | Stubbed (raises `NotImplementedError`, CLI exit code 3) |
| --- | --- |
| Schema v0: pydantic models, JSON Schema export (`schema/v0.json`), drift check | Provider `complete()` for all six backends |
| Semantic validator (graph rules, state rules, route walker, strict mode) | Agent nodes (planner … critic), single-shot runner, deep-agent variant |
| Valid-by-construction topology/scene generator used by the property tests | `crop`/`read`/`render` tools and the MCP server (`validate` tool is real) |
| `bench validate`, `bench schema export`, `bench run --dry-run` | `bench run` (without `--dry-run`), `bench eval`, `bench report`, `bench synth generate` |
| Harness: `GET /health`, `GET /metrics`, `POST /validate`; `HarnessClient` | Harness inference: `POST /frame` answers 501, `WS /stream` closes with 1011 |
| Configs (run / task / model catalogue), settings, run-directory contract | Ingest (image, video, screen, preprocessing, quality), synthetic rendering and augmentation |
| Manifest JSONL, deterministic dev/test partition, difficulty tiers | Element matching, every metric, parquet aggregation, MLflow tracker, Hub push/pull, report rendering |
| LangGraph wiring of the six-node loop with the critic → planner edge | |

`tests/unit/test_stubs.py` enumerates every stub and asserts it raises with a non-empty docstring,
so a stub cannot quietly become a fake.

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
| `src/rail_vision_bench/providers/` | `VisionProvider` protocol, request/response models, registry, six provider stubs. |
| `src/rail_vision_bench/agents/` | LangGraph state, node stubs, graph wiring, single-shot and deep-agent stubs, prompt loader. |
| `src/rail_vision_bench/prompts/` | Versioned prompts shipped as package data (`single_shot.md`, `agentic/*.md`). |
| `src/rail_vision_bench/tools/` | The four agent tools (`validate` real) and the MCP server stub. |
| `src/rail_vision_bench/ingest/` | Image/video/screen ingestion, preprocessing and quality metrics (named `ingest`, not `io`). |
| `src/rail_vision_bench/synth/` | Synthetic dataset generation, SVG rendering and augmentation stubs. |
| `src/rail_vision_bench/eval/` | Prediction/metric row models, matching, metric names and stubs, aggregation stub. |
| `src/rail_vision_bench/harness/` | FastAPI real-time harness, Prometheus registry, wire models, async client. |
| `src/rail_vision_bench/dataset/` | Manifest JSONL, dev/test partition, difficulty tiers, Hub stubs (named `dataset`, not `data`). |
| `src/rail_vision_bench/tracking/` | Tracker protocol, `NullTracker`, MLflow tracker stub. |
| `src/rail_vision_bench/report/` | Leaderboard template (package data) and render stub. |
| `schema/v0.json`, `schema/examples/v0/` | The published JSON Schema and two strict-valid example documents. |
| `configs/` | `run.example.yaml`, `models.yaml` (catalogue), `tasks/panel_topology.yaml`. |
| `data/` | DVC data plane; only `manifest.example.jsonl` and `README.md` are committed. |
| `docker/compose.yaml` | n8n and MLflow for local runs. |
| `docs/` | `schema.md` (human spec of v0) and `architecture.md` (pipeline, run dirs, wire protocol). |
| `tests/` | `unit/`, `property/` (hypothesis), `fixtures/` (invalid and strict documents), `strategies.py`. |
| `notebooks/` | Exploration notebooks (outputs stripped, excluded from lint/type checks). |

## Command line

`bench` exits with **0** on success, **1** on validation errors, schema drift or a bad config
reference, **2** on a usage error (typer) and **3** when a command reaches a skeleton stub.

| Command | Purpose | Status | Exit codes |
| --- | --- | --- | --- |
| `bench --version` | Print the package version. | implemented | 0 |
| `bench validate FILES... [--strict] [--schema PATH] [--json]` | Validate documents against schema v0 and the semantic rules; table or one JSON object `{file: report}`. | implemented | 0 / 1 / 2 |
| `bench schema export [--out PATH] [--check]` | Write `schema/v0.json` from the models, or check the committed file for drift. | implemented | 0 / 1 / 2 |
| `bench run --config PATH [--model NAME]... [--mode] [--limit] [--seed] [--out] [--dry-run]` | Resolve a run config (task file, catalogue, model names) and, without `--dry-run`, execute it. | `--dry-run` implemented; execution stubbed | 0 / 1 / 2 / 3 |
| `bench eval RUN_DIR` | Score a run into `metrics.parquet` and `summary.json`. | stubbed | 2 / 3 |
| `bench report RUN_DIRS... [--out PATH]` | Render a leaderboard (default `reports/leaderboard.html`). | stubbed | 2 / 3 |
| `bench synth generate --out DIR [--n] [--seed] [--augment]` | Generate synthetic panels with ground truth and a manifest. | stubbed | 2 / 3 |
| `bench serve [--host] [--port] [--reload]` | Serve the harness (defaults from `RVB_HARNESS_HOST` / `RVB_HARNESS_PORT`). | implemented for the live endpoints | 0 / 2 |

`--strict` promotes `STATE_MISSING`, `GEOM_OUT_OF_BOUNDS` and `ROUTE_SWITCH_MISSING` from
warnings to errors; CI validates the examples in strict mode.

## Schema and documentation

- [`schema/v0.json`](schema/v0.json) — the published JSON Schema (draft 2020-12) for ground
  truth and predictions; generated, never hand-edited (`make schema`, checked by `make schema-check`).
- [`docs/schema.md`](docs/schema.md) — the human specification of v0: element kinds with the
  German glossary, ports and degrees, DKW/EKW path families, state vocabularies, the route
  walker, every issue code, strict mode and the versioning policy.
- [`docs/architecture.md`](docs/architecture.md) — pipeline, module map, run-directory contract,
  harness wire protocol, MCP/n8n integration, configuration lifecycle, reproducibility inputs.
- [`data/README.md`](data/README.md) — the DVC data plane, the manifest contract, splits,
  partitions, difficulty tiers and the PII/consent/licensing policy.
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
  `make compose-down` stops them. Point `MLFLOW_TRACKING_URI=http://localhost:5000` at the
  server and set `tracker: mlflow` in a run config once the tracker is implemented.
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
