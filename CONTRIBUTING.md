# Contributing

This document is the developer contract: how the tree is built, checked and extended, and why
the tooling is configured the way it is. Every rule here is enforced by `make check`,
pre-commit or a test, so read it once and let the tools remind you afterwards.

## Developer workflow

```sh
make install        # venv + lock + editable package + typing stubs + pre-commit hooks
make check          # lint typecheck schema-check validate-examples test  (the default goal)
make help           # every target with a one-line description
```

Targets you will use daily:

| Target | Runs |
| --- | --- |
| `lint` / `format` | `ruff check .` + `ruff format --check .` / the fixing variants |
| `typecheck` | `mypy` with the config in `pyproject.toml` (files: `src`, `tests`) |
| `test` / `test-cov` | `pytest` / with coverage and `coverage.xml` for CI |
| `schema` / `schema-check` | regenerate `schema/v0.json` from the models / fail on drift |
| `validate-examples` | `bench validate --strict schema/examples/v0/*.json` |
| `dry-run` | `bench run --config configs/run.example.yaml --dry-run` |
| `lock` / `lock-check` | regenerate `requirements.lock` in place / fail if it drifted |
| `build-check` | `uv build` and verify dynamic deps and package data landed in the wheel |
| `pre-commit` | every hook on the whole tree |
| `serve`, `synth`, `compose-up`, `compose-down`, `clean` | harness with reload, synthetic data (exits 3 in the skeleton), Docker services, cache cleanup |

`make test VENV=/path/to/venv311` runs any target against another venv; CI runs the whole set
on Python 3.11 and 3.12.

## The lock

`requirements.txt` is the user's dependency spec and is committed verbatim: never edit it as a
side effect of a tooling change. `requirements.lock` is derived from it with exactly one command,

```sh
uv pip compile requirements.txt -o requirements.lock --universal --python-version 3.11
```

which `make lock` runs in place (uv keeps the existing pins and only resolves what changed).
Commit the regenerated lock together with the `requirements.txt` change. CI runs `make
lock-check` — the same command followed by `git diff --exit-code -- requirements.lock` — with
uv pinned to 0.8.17 (the version that produced the lock; bump the two together). `uv.lock` is
gitignored: `uv sync` is not the workflow here.

## The schema file

`schema/v0.json` is generated from `SceneAnnotation.model_json_schema(mode="validation")`
with `$schema` and `$id` injected and stdlib `json` serialisation (indent 2, sorted keys,
trailing newline), which is byte-identical on 3.11 and 3.12. After any edit under
`src/rail_vision_bench/schema/`, run `make schema` and commit the result together with an update
of `docs/schema.md`; `make schema-check`, the `schema-check` pre-commit hook and
`tests/unit/test_schema.py` all fail on drift. The installed package never reads the file (the
schema is built in memory); the committed file is the published contract.

## Stub policy

A stub is a function or method whose body is exactly a Google docstring (stating the intended
implementation) followed by

```python
raise NotImplementedError(
    "rail_vision_bench.<module>.<qualname> is not implemented in the skeleton: <one-line intent>"
)
```

Rules: a stub never imports the SDK it will use (this keeps `bench --help` fast and the mypy
override list empty), never contains `yield` (an iterator stub is a plain function that
raises), and every stub is listed in `tests/unit/test_stubs.py`, which asserts the exception and
a non-empty docstring. The CLI maps `NotImplementedError` to exit code 3 so a stub is never
mistaken for success. Nothing may return a placeholder value that looks like a result.

## Tooling rules and why they exist

- **`from __future__ import annotations` is mandatory** in every module under `src/` and
  `tests/`, right after the module docstring. `types-networkx` makes `nx.MultiGraph` generic
  and mypy strict requires `nx.MultiGraph[str]`, but the runtime class is not subscriptable;
  only postponed evaluation makes both true at once. typer, FastAPI, pydantic, pydantic-settings
  and the LangGraph TypedDict state still resolve the string annotations because the imports
  stay at module level.
- **No `python_version` in the mypy config.** numpy on 3.12 (the lock installs different
  numpy versions per interpreter) ships PEP 695 `type` statements in its stubs, and forcing
  `python_version = "3.11"` makes mypy fail inside `numpy/__init__.pyi` even though our code
  never imports numpy (the networkx stubs do). Each CI leg type-checks with its own interpreter.
- **Typeshed stubs live in `[dependency-groups] typing`** (`types-PyYAML`, `types-jsonschema`,
  `types-networkx`) and are installed with `uv pip install --group typing`. They are dev-only:
  setuptools ignores dependency groups, so they are neither in the wheel nor in the lock. To
  bump one, edit the pin in `pyproject.toml`, reinstall the group, run `make typecheck`.
- **No `ignore_missing_imports` overrides exist**, and none may be added. Implemented modules
  import only typed libraries or ones with installed stubs. Libraries without types in the lock
  (pandas, scipy, scikit-learn, pyarrow, shapely, svgwrite, cairosvg, albumentations,
  datasets, tabulate, dvc, tqdm, mistralai) are imported inside function bodies of the code
  that uses them, never at module level, and their objects are typed as `Any` or replaced by
  `Path`/`str` contracts in signatures. Third-party types for annotations only are allowed
  under `if TYPE_CHECKING:` for libraries that ship `py.typed` (PIL, httpx, langgraph, fastapi,
  prometheus_client, starlette).
- **ruff rule choices.** Selected: E, W, F, I, UP, B, A, C4, SIM, PIE, PT, RET, RUF, ANN, D, N,
  TID, PTH, ERA, T20, PLE, PLW, LOG, G, FAST, ASYNC, PERF; Google docstrings; line length 100;
  `tests/**` is exempt from D and ANN. Deliberately **not** selected: TC and FA (they would
  move imports under `TYPE_CHECKING` and break runtime annotation resolution), ARG (stubs have
  unused parameters by design), PD/NPY (no pandas or numpy in implemented code), PLC (lazy
  imports inside command bodies need no `noqa`). `flake8-builtins` runs with
  `strict-checking = true`, so a module or subpackage may not shadow a stdlib name.
- **Naming guard.** Because of the rule above, subpackages are named `ingest`, `dataset`,
  `harness`, `tracking` — never `io`, `data`, `types`, `logging`, `json` or any other stdlib
  module name. `dataset` also avoids confusion with the Hugging Face `datasets` package.
- **No `print`** (T20): the CLI uses rich consoles or `typer.echo`; library code logs or returns.
- **typer options use `Annotated[...]`** (no `B008` call expressions in defaults) and heavy
  modules are imported inside the command body so `bench --help` stays fast. FastAPI form and
  file parameters use `Annotated[UploadFile, File()]` / `Annotated[str, Form()]`.
- **pydantic:** `model_config = ConfigDict(extra="forbid")` on every model,
  `Field(default_factory=...)` for mutable defaults, `StrEnum` vocabularies, no aliases, no
  `frozen`. The schema models are purely structural; every semantic rule lives in
  `graph/validator.py` so an invalid document still parses and gets a precise issue code.
- **Imports from concrete modules** inside `src/` (`rail_vision_bench.schema.models`, not
  `rail_vision_bench.schema`); package `__init__` re-exports exist for users, not for the
  package itself.
- **ASCII only in Python source** (RUF001-003); Markdown and YAML may carry umlauts and arrows.

## Test conventions

- `tests/`, `tests/unit/` and `tests/property/` are packages (`__init__.py`), so
  `from tests.conftest import REPO_ROOT` and `from tests.strategies import ...` work under
  pytest's prepend import mode and mypy sees `tests.*` (with `disallow_untyped_defs` relaxed
  for that tree only).
- `pytest` runs with `--strict-markers --strict-config`, `xfail_strict`, `asyncio_mode = auto`
  and turns `DeprecationWarning`s raised by `rail_vision_bench` into errors.
- Markers: `requires_display` (skipped unless `DISPLAY` is set), `requires_network` (skipped
  unless `RVB_NETWORK_TESTS=1`), `slow` (deselect with `-m "not slow"`). The skip logic lives in
  `tests/conftest.py`; no test needs provider keys, a display or the network by default.
- An autouse fixture clears every `SETTINGS_ENV_KEYS` variable and `get_settings()`'s cache, so
  a developer's `.env` never leaks into a test.
- Hypothesis profiles: `default` (50 examples) and `ci` (200 examples, derandomized, deadline
  off, `too_slow` suppressed), selected with `HYPOTHESIS_PROFILE=ci`. Strategies wrap the
  valid-by-construction generator in `graph/generate.py`; mutators in `tests/strategies.py`
  produce documents that fail a specific rule.
- Parametrize with a names tuple and list values (`("a", "b")`, PT006/PT007);
  `pytest.raises(NotImplementedError)` is fine for stubs.

## pre-commit

The `mypy` and `schema-check` hooks are `language: system`: they run `mypy` and `bench schema
export --check` from the **active** project venv, because pre-commit, mypy and bench are all
installed there from `requirements.txt`. Activate the venv (or run `make pre-commit`, which
calls the venv's binary) before committing. `SKIP=mypy git commit ...` is the documented escape
hatch when mypy already ran; CI uses the same `SKIP=mypy` after its own typecheck step.
`nbstripout` strips notebook outputs; `end-of-file-fixer` and `trailing-whitespace` skip
`requirements.txt` and `requirements.lock` because those files are verbatim.

## How to add things

- **A provider:** add `src/rail_vision_bench/providers/<name>.py` with one class
  (`name: ClassVar[str] = "<key>"`, `__init__(self, settings)`, `async complete(request)`),
  a lazy factory in `providers/registry.py`, the key in `config.ProviderName` and
  `PROVIDER_NAMES`, a catalogue entry in `configs/models.yaml`, and the tests in
  `tests/unit/test_providers.py` (the registry must resolve every name) and
  `tests/unit/test_stubs.py` while it is a stub.
- **A metric:** add the name to `eval/metrics.METRIC_NAMES`, a function returning
  `list[MetricResult]`, its call in the (future) `eval/aggregate.py`, a column in
  `report/templates/leaderboard.html.j2` if it should be on the leaderboard, and document it
  in `docs/architecture.md`.
- **A tool:** add the callable to `tools/tools.py`, its name to `TOOL_NAMES`, register it in
  `tools/mcp_server.py` once that exists, and give the agent prompts a sentence on when to use it.
- **A task:** add `configs/tasks/<name>.yaml` (`TaskConfig` fields: `name`, `description`,
  `source_kind`, `split`, `prompt`, `metrics` from `METRIC_NAMES`), make sure the split exists
  in the manifest, and reference it from a run config.
- **A prompt:** add `src/rail_vision_bench/prompts/<name>.md`, list it in
  `agents/prompts.PROMPT_NAMES`; `tests/unit/test_package_data.py` checks it loads from the
  installed package.
- **A schema change:** edit `schema/models.py` (structural) or `graph/validator.py` and
  `graph/rules.py` (semantic), add a fixture under `tests/fixtures/`, update `docs/schema.md`,
  run `make schema`.

## Packaging notes

`pytest`, `ruff`, `mypy`, `pre-commit`, `nbstripout` and `jupyterlab` are runtime
dependencies of the distribution because the user's `requirements.txt` lists them and the
wheel's metadata is derived from that file. That is intentional and not to be "fixed" in
`pyproject.toml`; the `Private :: Do Not Upload` classifier guarantees that PyPI rejects the
distribution, so the oddity can never reach the public index. `make build-check` verifies that
the wheel carries the dynamic requirements, `py.typed`, the prompts and the report template.

## Pull requests

Use the checklist in `.github/pull_request_template.md`: lock regenerated when
`requirements.txt` changed, schema regenerated when the models changed, `make check` green on
both interpreters, new stubs raise and are listed, no fabricated results. Do not commit a
`LICENSE` file: the license is still undecided (see the README).
