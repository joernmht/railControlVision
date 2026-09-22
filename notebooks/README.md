# Notebooks

Exploration only: crops, error analysis, looking at a run's `predictions.jsonl`, sketching a
metric before it becomes code. Nothing in the package, the CLI, the tests or CI depends on a
notebook, and nothing here is ever imported (`from notebooks...` does not exist and must not be
made to exist; move reusable code into `src/rail_vision_bench/` with a test).

## Conventions

- **Outputs are stripped on commit.** The `nbstripout` pre-commit hook removes cell outputs and
  execution counts, so diffs stay readable and no rendered image or result lands in git. Run
  `make install` once so the hook is active.
- **Excluded from ruff and mypy.** `notebooks/` is in ruff's `extend-exclude` and outside mypy's
  `files`, so notebook code is not linted or type-checked; that is the trade-off for being
  allowed to be messy. Do not lower the bar of `src/` to match.
- **Naming:** `NN_topic.ipynb` with a two-digit ordinal, e.g. `01_crop_gallery.ipynb`,
  `02_route_walker_failures.ipynb`. The number keeps the reading order; the topic says what you
  will find.
- **Data:** read through the package (`read_manifest`, `validate_document`, the example
  documents under `schema/examples/v0/`) rather than re-implementing loaders; never commit data
  into a notebook.
- **Kernel:** the project venv (`.venv`) has `jupyterlab` because `requirements.txt` lists it;
  `.venv/bin/jupyter lab` from the repository root.
