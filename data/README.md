# Data plane

`data/` holds the benchmark imagery and ground truth. Only this file,
`manifest.example.jsonl`, `ATTRIBUTION.md` and the DVC pointer files are committed; everything
else is content tracked by DVC and lives in the DVC remote, never in git.

## Layout

```
data/
├── README.md                  this file (git)
├── manifest.example.jsonl     two rows pointing at the committed schema examples (git)
├── ATTRIBUTION.md             credits for the third-party images in raw/ (git, generated)
├── raw/                       original photos, screen captures, video frames as received (DVC)
│   └── commons/               curated Wikimedia Commons panel photos: images/ + sources.jsonl
├── incoming/                  staged downloads awaiting curation (gitignored, never in DVC)
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
| `license_url` | string \| null | URL of the licence terms |
| `author` | string \| null | author to credit for a third-party image |
| `source_url` | string \| null | page documenting where a third-party image came from |
| `panel_group` | id \| null | physical panel or site; a dev/test split must keep a group in one partition |
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

## Publishing on the Hugging Face Hub

`rail_vision_bench.dataset.hub.push_split(split, repo_id, manifest=...)` and
`pull_split(split, repo_id, dest=...)` are the intended interface (stubs in the skeleton). They
will build a `datasets.Dataset` from the manifest rows (image column plus the ground-truth JSON
as a string column), push it under the split name with the token from `HF_TOKEN`, and rebuild a
local manifest on download. DVC stays the system of record; the Hub is a distribution channel
for released splits only.

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

6. **Published third-party images** (Wikimedia Commons) need no separate permission record:
   the licence is the permission, and `sources.jsonl` records it per image. Only licences
   without share-alike or non-commercial terms are admitted (see below).

If a source cannot meet 1-3 it does not go into the benchmark, however useful the image.

## Wikimedia Commons source images

Benchmark tiles and crops are adaptations of the source image. Under CC BY-SA an adaptation
must be shared alike, which would bind the derived data, so the main set admits only **CC BY
(any version or port), CC0, public domain and the Commons `Attribution` licence**
(`rail_vision_bench.dataset.licensing`). Other licences are rejected before download.

```sh
bench commons ingest                      # search Commons, stage candidates in data/incoming/commons
bench commons status <source_id> rejected --note "near-duplicate of ..."   # curate, with the reason
bench commons group <source_id> nl-hilversum                              # panel/site group
bench commons blur <source_id> --box x0,y0,x1,y1 [--box ...]               # faces, in staging only
bench commons promote                     # sync data/raw/commons with the accepted staged images
bench commons attribution --out data/ATTRIBUTION.md
cp data/ATTRIBUTION.md data/raw/commons/ATTRIBUTION.md   # the credits travel with the images
dvc add data/raw && dvc push              # then commit data/raw.dvc and data/ATTRIBUTION.md
```

Each image is described by a `SourceRecord` line in `sources.jsonl` (`rail_vision_bench.dataset.sources`):
`source_id` (`commons-<first 12 hex of sha256>`), curation `status`, `image`, `sha256`, `width`,
`height`, Commons `title`, `source_url`, `file_url`, `author`, `credit`, `license` (SPDX or a
`LicenseRef-` id), `license_name` (as Commons states it), `license_url`, `description`,
`search_term`, `ingested_at`, `panel_group`, `curation_note`, `modification` and
`blur_regions`. `sha256` is the hash of the file as stored (after blurring); `source_id` keeps
naming the original download.

- **Panel groups.** `panel_group` names the physical panel or site an image shows (site level,
  e.g. `nl-hilversum`, `dk-padborg`). Several images show the same panel, so any dev/test split
  must assign whole groups to one partition, never single images, or it leaks; `ManifestRow`
  carries the same field.
- **Blurring.** Identifiable people are blurred in staging before promotion (rule 2): `bench
  commons blur` overwrites the staged file, so the unblurred original never reaches the data
  plane or the DVC remote. Boxes are placed by hand and checked visually; they are kept in
  `blur_regions`. Blurring is an adaptation, so `modification` is set to `faces blurred` and the
  attribution entry says *Modified (faces blurred)*.
- **Promotion is a sync.** A record that is no longer accepted in staging is removed from
  `data/raw/commons`, image included. `bench commons import-db` migrates the accepted images of the
raiLPoperator `panelvision` prototype. None of these images has ground truth yet, so they have
no manifest rows; when a scene gets a `SceneAnnotation`, its row copies `license`,
`license_url`, `author` and `source_url` from the source record.
