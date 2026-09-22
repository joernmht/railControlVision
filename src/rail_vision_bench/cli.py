"""The ``bench`` command line.

Exit codes: 0 ok, 1 validation errors / schema drift / bad config reference,
2 usage error (typer), 3 not implemented in the skeleton. Heavy modules
(validator, schema export, runner, harness, synth) are imported inside the
command bodies so that ``bench --help`` stays fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from rail_vision_bench import __version__
from rail_vision_bench.config import Mode
from rail_vision_bench.schema.issues import IssueCode, ValidationIssue, ValidationReport

app = typer.Typer(
    name="bench",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="rail-vision-bench command line.",
)
schema_app = typer.Typer(no_args_is_help=True, help="Schema tools.")
app.add_typer(schema_app, name="schema")
synth_app = typer.Typer(no_args_is_help=True, help="Synthetic data.")
app.add_typer(synth_app, name="synth")

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3

console = Console()
err_console = Console(stderr=True)

# Module constants rather than call expressions in the signatures (B008) and so
# the help text shows the same default the Makefile uses.
DEFAULT_SCHEMA_PATH = Path("schema/v0.json")
DEFAULT_REPORT_PATH = Path("reports/leaderboard.html")


def _not_implemented(exc: NotImplementedError) -> NoReturn:
    """Report a skeleton stub on stderr and exit with ``EXIT_NOT_IMPLEMENTED``."""
    err_console.print(f"[not implemented] {exc}", markup=False, highlight=False, soft_wrap=True)
    raise typer.Exit(EXIT_NOT_IMPLEMENTED)


def _fail(message: str) -> NoReturn:
    """Print ``message`` on stderr and exit with ``EXIT_INVALID``."""
    err_console.print(message, markup=False, highlight=False, soft_wrap=True)
    raise typer.Exit(EXIT_INVALID)


def _describe(exc: BaseException) -> str:
    """Return an exception's message without the quoting ``str(KeyError)`` adds."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _version(value: bool) -> None:
    """Eager ``--version`` callback: print the version and stop."""
    if value:
        console.print(f"bench {__version__}", highlight=False)
        raise typer.Exit(EXIT_OK)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """rail-vision-bench command line."""


def _print_reports(reports: dict[str, ValidationReport]) -> None:
    """Render the issues of every file as one table plus a summary line."""
    table = Table("file", "code", "severity", "element", "message")
    n_errors = 0
    n_warnings = 0
    for name, report in reports.items():
        for issue in report.issues:
            table.add_row(
                name, issue.code.value, issue.severity, issue.element_id or "", issue.message
            )
        n_errors += len(report.errors)
        n_warnings += len(report.warnings)
    if table.row_count:
        console.print(table)
    console.print(
        f"{len(reports)} file(s) checked: {n_errors} error(s), {n_warnings} warning(s)",
        highlight=False,
    )


@app.command()
def validate(
    files: Annotated[
        list[Path],
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="SceneAnnotation JSON files."
        ),
    ],
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Promote STATE_MISSING, GEOM_OUT_OF_BOUNDS and ROUTE_SWITCH_MISSING to errors.",
        ),
    ] = False,
    schema: Annotated[
        Path | None,
        typer.Option(
            "--schema",
            exists=True,
            dir_okay=False,
            help="JSON Schema file to use instead of the in-memory export.",
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit one JSON object {file: report} instead of a table.")
    ] = False,
) -> None:
    """Validate documents against schema v0 and the semantic rules; exit 1 on any error."""
    import orjson

    from rail_vision_bench.graph.validator import validate_document
    from rail_vision_bench.schema.validate import load_schema

    schema_doc = load_schema(schema) if schema is not None else None
    reports: dict[str, ValidationReport] = {}
    for file in files:
        try:
            loaded = orjson.loads(file.read_bytes())
        except orjson.JSONDecodeError as exc:
            loaded = None
            problem = f"not valid JSON: {exc}"
        else:
            problem = "expected a JSON object at the top level"
        if isinstance(loaded, dict):
            reports[str(file)] = validate_document(loaded, strict=strict, schema=schema_doc)
        else:
            # A file that does not even hold an object is reported like any schema failure
            # instead of crashing, so a batch of files always yields one report per file.
            issue = ValidationIssue(
                code=IssueCode.SCHEMA_INVALID, severity="error", message=problem, path=""
            )
            reports[str(file)] = ValidationReport.from_issues([issue])
    if json_output:
        payload = {name: report.model_dump(mode="json") for name, report in reports.items()}
        typer.echo(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
    else:
        _print_reports(reports)
    if any(not report.ok for report in reports.values()):
        raise typer.Exit(EXIT_INVALID)


@schema_app.command("export")
def schema_export(
    out: Annotated[Path, typer.Option("--out", help="Destination file.")] = DEFAULT_SCHEMA_PATH,
    check: Annotated[
        bool, typer.Option("--check", help="Only verify that OUT matches the models.")
    ] = False,
) -> None:
    """Write schema v0 as JSON Schema (draft 2020-12), or check the committed file for drift."""
    from rail_vision_bench.schema.export import check_json_schema, write_json_schema

    if check:
        try:
            up_to_date = check_json_schema(out)
        except FileNotFoundError:
            up_to_date = False
        if not up_to_date:
            _fail(f"{out} is out of date; run `make schema`")
        console.print(f"{out} is up to date", highlight=False)
        return
    write_json_schema(out)
    console.print(f"wrote {out}", highlight=False)


@app.command()
def run(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Run configuration YAML.")
    ],
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model", help="Catalogue model name (repeatable); replaces the config's list."
        ),
    ] = None,
    mode: Annotated[Mode | None, typer.Option("--mode", help="single_shot or agentic.")] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Only the first N scenes of the split.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Run-directory root.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Resolve and print the configuration without running."),
    ] = False,
) -> None:
    """Run the benchmark described by a config file (or only resolve it with --dry-run)."""
    import yaml

    from rail_vision_bench import runner
    from rail_vision_bench.config import apply_overrides, load_run_config, resolve_run

    try:
        run_config = apply_overrides(
            load_run_config(config),
            models=model or None,
            mode=mode,
            limit=limit,
            seed=seed,
            out_dir=out,
        )
        run_config, _task, _catalogue = resolve_run(run_config)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        # pydantic's ValidationError is a ValueError, as is load_yaml's non-mapping error.
        _fail(f"{config}: {_describe(exc)}")
    if dry_run:
        typer.echo(yaml.safe_dump(run_config.model_dump(mode="json"), sort_keys=False), nl=False)
        return
    try:
        written = runner.run_benchmark(run_config)
    except NotImplementedError as exc:
        _not_implemented(exc)
    console.print(f"run written to {written}", highlight=False)


@app.command("eval")
def eval_run(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="A run directory.")],
) -> None:
    """Score the predictions of a run into metrics.parquet and summary.json."""
    from rail_vision_bench import runner

    try:
        written = runner.evaluate_run(run_dir)
    except NotImplementedError as exc:
        _not_implemented(exc)
    console.print(f"metrics written to {written}", highlight=False)


@app.command()
def report(
    run_dirs: Annotated[
        list[Path],
        typer.Argument(exists=True, file_okay=False, help="Evaluated run directories."),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Leaderboard destination.")
    ] = DEFAULT_REPORT_PATH,
) -> None:
    """Render a leaderboard comparing evaluated runs."""
    from rail_vision_bench import runner

    try:
        written = runner.build_report(run_dirs, out)
    except NotImplementedError as exc:
        _not_implemented(exc)
    console.print(f"report written to {written}", highlight=False)


@synth_app.command("generate")
def synth_generate(
    out: Annotated[Path, typer.Option("--out", help="Output directory of the dataset.")],
    n: Annotated[int, typer.Option("--n", min=1, help="Number of scenes.")] = 10,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = 0,
    augment: Annotated[
        str | None, typer.Option("--augment", help="Augmentation preset name.")
    ] = None,
) -> None:
    """Generate synthetic panel images with ground truth and a manifest."""
    from rail_vision_bench.synth.generate import generate_dataset

    try:
        manifest = generate_dataset(out, n=n, seed=seed, augment=augment)
    except NotImplementedError as exc:
        _not_implemented(exc)
    console.print(f"manifest written to {manifest}", highlight=False)


@app.command()
def serve(
    host: Annotated[
        str | None, typer.Option("--host", help="Bind host (default: RVB_HARNESS_HOST).")
    ] = None,
    port: Annotated[
        int | None, typer.Option("--port", help="Bind port (default: RVB_HARNESS_PORT).")
    ] = None,
    reload: Annotated[
        bool, typer.Option("--reload", help="Restart on source changes (development only).")
    ] = False,
) -> None:
    """Serve the real-time harness (health, metrics, validate, frame, stream)."""
    import uvicorn

    from rail_vision_bench.harness.app import create_app
    from rail_vision_bench.settings import get_settings

    settings = get_settings()
    bind_host = host if host is not None else settings.rvb_harness_host
    bind_port = port if port is not None else settings.rvb_harness_port
    log_level = settings.rvb_log_level.lower()
    if reload:
        # uvicorn can only restart an application it imports itself, so the reloader
        # gets the factory path instead of an application object.
        uvicorn.run(
            "rail_vision_bench.harness.app:create_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            reload=True,
            log_level=log_level,
        )
        return
    uvicorn.run(create_app(settings), host=bind_host, port=bind_port, log_level=log_level)
