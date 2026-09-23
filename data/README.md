# Data plane

`data/` holds the benchmark imagery and ground truth. Only this file and
`manifest.example.jsonl` are committed; everything else is content tracked by DVC and lives in
the DVC remote, never in git.

## Layout

```
data/
├── README.md                  this file (git)
├── manifest.example.jsonl     two rows pointing at the committed schema examples (git)
├── raw/                       original photos, screen captures, video frames as received (DVC)
├── gt/                        ground-truth SceneAnnotation JSON, one file per scene (DVC)
└── synthetic/                 rendered synthetic panels and their ground truth (DVC)
```

`raw/`, `gt/` and `synthetic/` are created on first use. `data/` is deliberately **not** in
`.gitignore`: DVC appends its own ignore entries (`/raw`, `/gt`, ...) next to the `.dvc` files
it creates, and those `.dvc` pointer files are what git tracks.

## DVC workflow

`dvc init` is run once by the repository owner (it creates `.dvc/` and `.dvcignore`; do not
create those by hand). After that:

```sh
dvc remote add -d storage <url>           # once; s3://, gs://, ssh://, a local path, ...
dvc add data/raw data/gt data/synthetic   # after adding or changing content
git add data/*.dvc .gitignore             # commit the pointers, not the content
dvc push                                  # upload the content to the remote
dvc pull                                  # on another machine: fetch what the pointers name
```

For an S3-compatible remote (MinIO included) install the extra: `uv pip install -e ".[s3]"`
(`dvc[s3]`). `RVB_DATA_DIR` (default `data`) tells the bench where this tree is.

## The manifest contract

Each split is described by a `manifest.jsonl`: one JSON object per line, validated by
`rail_vision_bench.dataset.manifest.ManifestRow` and read/written with `read_manifest` /
`write_manifest`. Paths are relative to the repository root.

| Field | Type | Meaning |
| --- | --- | --- |
| `scene_id` | id (`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$`) | unique per scene; equals `scene_id` inside the ground-truth document |
| `split` | string | named subset, see below |
| `partition` | `dev` \| `test` | from `assign_partition(scene_id)` |
| `difficulty` | `easy` \| `medium` \| `hard` \| null | from `difficulty_tier(quality)` when quality was measured |
| `source_kind` | `synthetic` \| `panel_photo` \| `estw_screen` \| `ctc_screen` \| `stream_frame` | same vocabulary as `source.kind` in the document |
| `image` | path | the source image; must equal `source.image` of the ground truth |
| `gt` | path | the ground-truth `SceneAnnotation` JSON (strict-valid) |
| `width`, `height` | int >= 1 | must equal `source.width` / `source.height` |
| `license` | string \| null | SPDX id or free text of the image licence; see the policy below |
| `quality` | object \| null | `QualityMetrics` fields (`blur_var`, `glare_fraction`) when measured |

`manifest.example.jsonl` holds the rows for `minimal` and `station_dkw`, whose ground truth are
the committed `schema/examples/v0/*.json`; their images (`data/synthetic/example/*.png`) are not
committed, so loaders and tests work without a DVC pull.

## Splits, partitions, difficulty

**Splits** are named subsets a task selects with `TaskConfig.split`. The intended names follow
the source kinds: `synthetic_clean` (rendered panels, no augmentation), `synthetic_aug` (rendered
and augmented: perspective, glare, occlusion, JPEG, motion blur), `panel_photo`, `estw_screen`,
`ctc_screen`, `stream_frame`. A split may carry a version suffix (`panel_photo_v1`) once real
data exists.

**Partition** is deterministic: `assign_partition(scene_id, test_fraction=0.2)` hashes the id
(SHA-1, first 8 hex digits, modulo 1000) and puts a scene into `test` when the bucket is below
`round(test_fraction * 1000)`. Adding or removing scenes never moves another scene between
partitions. Prompts and configs are tuned on `dev`; `test` numbers are reported.

**Difficulty** is derived from image quality by `difficulty_tier(quality, thresholds)` with the
defaults of `DifficultyThresholds`: `hard` when `blur_var < 100` or `glare_fraction > 0.15`,
`medium` when `blur_var < 300` or `glare_fraction > 0.05`, else `easy`. The thresholds are a
model, so a split can carry its own.

## Synthetic data

`bench synth generate --out DIR [--n N] [--seed S] [--augment PRESET]`
(`synth.generate.generate_dataset`; `make synth` runs it with `--out data/synthetic --n 10
--seed 0`) writes

```
DIR/
├── images/<scene_id>.png      rendered panel (augmented when --augment is given)
├── gt/<scene_id>.json         strict-valid SceneAnnotation with exact pixel geometry
└── manifest.jsonl             one ManifestRow per scene
```

Scene ids are `synth-<seed>-<index:05d>`, with `-<preset>` appended when augmented. Rows use
split `synthetic_clean` (no `--augment`) or `synthetic_aug`, the hashed partition, the
difficulty tier of the quality measured on the final image, `license: null` and paths relative
to the working directory when `DIR` lies below it (absolute otherwise). Augmentation presets
(`synth.augment.PRESETS`): `default` (mild perspective, lighting, glare, occlusion, motion blur,
JPEG), `phone` (hand-held photo: stronger perspective, rotation, glare, occlusion) and `cctv`
(control-room camera: downscaling, sensor noise, heavy compression). Every ground-truth
coordinate goes through the same warp as the image, so augmented ground truth stays exact; an
unknown preset exits with code 1. Equal arguments produce byte-identical files.

## Publishing on the Hugging Face Hub

DVC stays the system of record; the Hub is a distribution channel for released splits only.
`rail_vision_bench.dataset.hub` stores each split in its own folder of one Hub *dataset*
repository, so several splits share a repository and a download fetches exactly one:

```
<split>/manifest.jsonl         ManifestRow per scene, paths relative to the repository root
<split>/images/<scene_id>.<ext>
<split>/gt/<scene_id>.json     strict-valid SceneAnnotation
```

- `push_split(split, repo_id, *, manifest, root=Path(), settings=None, private=True)` first
  runs `check_release`: the manifest is not empty, every row belongs to `split`, scene ids are
  unique, every non-synthetic row fills `license`, image and ground-truth files exist, and every
  ground truth parses, carries the row's `scene_id`, `width` and `height` and is strict-valid.
  Any failure raises `HubReleaseError` listing every problem, and nothing is uploaded. Otherwise
  the files are staged under the layout above (renamed to `<scene_id>`, manifest paths
  rewritten), the dataset repository is created when missing (private by default) and the
  folder is uploaded to `<split>/` in one commit with `HfApi.upload_folder`; files of an earlier
  upload of the split that are no longer staged are deleted in that commit. Returns the commit
  URL.
- `pull_split(split, repo_id, *, dest, revision=None, settings=None)` fetches only
  `<split>/**` with `snapshot_download` into `dest`, checks that every file the downloaded
  manifest names is present, re-anchors `image` and `gt` at `dest` and writes
  `dest/manifest.jsonl`. With a relative `dest` (for example `data/hub`) the paths stay
  repository-relative. Returns the manifest path.

A split name must be one safe path segment (`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`). The token is
`HF_TOKEN` (`settings.hf_token`), falling back to the `huggingface_hub` login when unset. There
is no `bench` command for the Hub; call the functions from Python.

## PII, consent and licensing policy

Control rooms are workplaces and interlocking screens can show operational detail. The rules
for anything under `data/raw`:

1. **No photo, screen capture or stream frame is added without documented permission** from
   the operator/owner of the installation and, where people are visible, the people themselves.
   Keep the permission record (who, what, when, which use) next to the data, outside git.
2. **Faces, name plates, badges and personal notes are blurred before the file enters the
   tree**; the original is not kept in the repository or the DVC remote.
3. **Every manifest row fills `license`** (SPDX id or the exact terms granted). A row with
   `license: null` is acceptable only for synthetic data generated by this project.
4. Operational secrets (credentials on sticky notes, internal phone lists, network diagrams)
   are removed or masked like personal data.
5. Synthetic data (`data/synthetic`) is generated by `bench synth generate` from
   `graph.generate` scenes and carries no third-party rights.

If a source cannot meet 1-3 it does not go into the benchmark, however useful the image.
