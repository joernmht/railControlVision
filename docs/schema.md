# Schema v0 — human specification

This document explains the document format that ground truth and model predictions share.
The machine-readable contract is [`schema/v0.json`](../schema/v0.json), generated from the
pydantic models in `src/rail_vision_bench/schema/models.py`; the semantic rules beyond JSON
Schema live in `src/rail_vision_bench/graph/validator.py` and use the tables in
`src/rail_vision_bench/graph/rules.py`. If this page and the code disagree, the code wins and
this page has a bug.

## One document type

A `SceneAnnotation` describes one image. Ground truth and predictions use the same type and are
told apart by `provenance.kind` (`ground_truth` | `prediction`), so the validator, the tools and
the metrics never need two code paths.

```
SceneAnnotation
├── schema_version: "v0"                 (constant)
├── scene_id                             (ElementId)
├── source      {kind, image, width, height, frame_index?, timestamp_ms?}
├── provenance  {kind, run_id?, model_id?, annotator?, tool?, created_at?}
├── topology    {nodes: [Node], edges: [Track]}
├── signals     [Signal]      (attached to edges)
├── derailers   [Derailer]    (attached to edges)
├── routes      [Route]       (start signal, ordered edge path, switch settings)
├── state       {switches, signals, tracks, derailers, routes, observed_at_ms?}
└── meta        free object (station name, difficulty tags, notes)
```

Every `id` (and every key of an id-keyed dict) matches `^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$`
and is unique across nodes, edges, signals, derailers and routes. Objects reject unknown keys
(`additionalProperties: false`) except `meta`. `confidence` fields are floats in `[0, 1]`, all
optional. Geometry (`bbox` `[x0, y0, x1, y1]`, `polyline` `[[x, y], ...]`, `point` `[x, y]`) is in
pixel coordinates of `source.image`, with `x` in `[0, width]` and `y` in `[0, height]`.

## Element kinds and the German glossary

| `kind` value | German panel term | Notes |
| --- | --- | --- |
| node `buffer_stop` | Prellbock | track ends here; one port |
| node `boundary` | Streckengrenze / Bildrand | track leaves the image or the interlocking area; one port |
| node `joint` | Gleisabschnittsgrenze (Isolierstoß) | boundary between two track sections; two ports |
| node `switch` | Weiche | plain switch; ports `toe`, `straight`, `diverging` |
| node `crossing` | Kreuzung | fixed diamond, no moving parts; ports A–D |
| node `ekw` | EKW, einfache Kreuzungsweiche | single slip; ports A–D |
| node `dkw` | DKW, doppelte Kreuzungsweiche | double slip; ports A–D |
| signal `main` | Hauptsignal | |
| signal `distant` | Vorsignal | aspects `expect_*` |
| signal `shunt` | Sperrsignal / Rangiersignal | aspects `stop`, `shunt_proceed` |
| signal `combined` | Ks-Signal (Haupt- und Vorsignal in einem) | |
| signal `repeater` | Vorsignalwiederholer | |
| signal `block_marker` | Blockkennzeichen | |
| derailer | Gleissperre | attached to an edge, like a signal |
| route | Fahrstraße | from a start signal along an ordered list of edges |
| track kind `main` / `siding` / `platform` / `yard` | Hauptgleis / Nebengleis / Bahnsteiggleis / Rangiergleis | `unknown` when not readable |

A `dkw`/`ekw` node may carry `half_labels` for its two Weichenhälften (e.g. `["12a", "12b"]`);
on any other kind that field triggers the `NODE_HALF_LABELS_UNEXPECTED` warning.

## Ports and degrees

An edge (`Track`) runs from `a: {node, port}` to `b: {node, port}`. Which ports a node has
depends on its kind, and every port must carry exactly one edge endpoint:

| Kind | Ports | Degree |
| --- | --- | --- |
| `buffer_stop`, `boundary` | `A` | 1 |
| `joint` | `A`, `B` | 2 |
| `switch` | `toe`, `straight`, `diverging` | 3 |
| `crossing`, `ekw`, `dkw` | `A`, `B`, `C`, `D` | 4 |

For the four-port kinds, `A` and `B` lie on one side and `C` and `D` on the other:

```
   A ──╲     ╱── C          straight paths: A–C and B–D (both kinds, and the crossing)
        ╲   ╱               slip paths:     A–D (ekw and dkw)
         ╲ ╱                                B–C (dkw only)
         ╱ ╲
        ╱   ╲
   B ──╱     ╲── D
```

## Allowed paths and path families

`ALLOWED_PATHS_BY_KIND` lists the port pairs a vehicle can traverse through a node;
`PATH_FAMILIES_BY_KIND` lists which paths can be connected **at the same time**:

| Kind | Allowed paths | Families (sets that may be active together) |
| --- | --- | --- |
| `joint` | A–B | — |
| `switch` | toe–straight, toe–diverging | — (state is a position, not a path list) |
| `crossing` | A–C, B–D | {A–C, B–D} (fixed; a crossing has no switch state) |
| `ekw` | A–C, B–D, A–D | {A–C, B–D} or {A–D} |
| `dkw` | A–C, B–D, A–D, B–C | {A–C, B–D} or {A–D, B–C} |

Paths are unordered pairs: `["A", "C"]` and `["C", "A"]` are the same path.

## Switch state: `position` vs `active_paths`

`state.switches` is keyed by node id and holds a `SwitchState`; which field is used depends on
the node kind (rule `STATE_WRONG_FIELD_FOR_KIND`):

- a **`switch`** uses `position`: `straight`, `diverging`, `moving` or `unknown`;
  `active_paths` must be null;
- a **`dkw` / `ekw`** uses `active_paths`: zero, one or two port pairs that are connected right
  now; `position` must be null;
- `crossing`, `joint` and the terminal kinds carry no switch state at all.

Two paths can be active at once because a slip switch has two independent halves: when both
halves of a DKW stand in `+`, both straight paths A–C and B–D are connected simultaneously, and
when both stand in `-`, both slips A–D and B–C are. The validator (rule `STATE_PATH_NOT_ALLOWED`)
requires every path to have two distinct ports, be allowed for the kind, be pairwise
port-disjoint and lie within one family:

| Kind | Accepted `active_paths` | Rejected |
| --- | --- | --- |
| `dkw` | `[]`, `[AC]`, `[BD]`, `[AC, BD]`, `[AD]`, `[BC]`, `[AD, BC]` | `[AB]`, `[CD]`, `[AC, AD]`, `[AC, BC]`, `[AD, AC]` |
| `ekw` | `[]`, `[AC]`, `[BD]`, `[AC, BD]`, `[AD]` | `[BC]`, `[AD, AC]` |

### Plus, Minus, Grundstellung and `raw`

`straight` and `diverging` describe the geometric path. German panels show **Plus (+)** for the
Grundstellung (normal position, usually but not always the straight path) and **Minus (-)** for
the other position; on a DKW each half shows its own sign (`12a:+ 12b:-`). The literal
indication as read from the panel goes into `SwitchState.raw` (`"+"`, `"-"`, `"12a:+ 12b:+"`),
so the mapping from indication to geometry is auditable and a model that reads the sign
correctly but maps it wrongly can be told apart from one that misread the panel.

## Signal state and aspects

`SignalState.aspect` is an English vocabulary; the literal aspect name goes into `raw`:

| `aspect` | Meaning | Typical `raw` (H/V, Ks, Sh) |
| --- | --- | --- |
| `stop` | Halt | `Hp0`, `Ks0` (`Hp00` on some panels), `Sh0` for a shunt signal |
| `proceed` | Fahrt | `Hp1`, `Ks1` |
| `proceed_reduced` | Langsamfahrt | `Hp2`, `Ks1` with speed indicator |
| `expect_stop` | Halt erwarten (distant) | `Vr0`, `Ks2` |
| `expect_proceed` | Fahrt erwarten (distant) | `Vr1` |
| `expect_proceed_reduced` | Langsamfahrt erwarten (distant) | `Vr2` |
| `shunt_proceed` | Rangierfahrt erlaubt | `Sh1`, `Ra12` |
| `dark` | signal is dark / off | |
| `unknown` | not readable | |

`Signal.system` names the signalling system the aspects belong to (`"H/V"`, `"Ks"`, `"Hl"`,
...); `Signal.kind` says what the signal is (main, distant, shunt, combined, repeater,
block_marker).

## Track state: two independent axes

`TrackState` has two axes that panels display separately and that must not be conflated:
`occupancy` (`free` | `occupied` | `unknown`, the Gleisfreimeldung, usually a red lamp) and
`route_set` (`true` | `false` | `null`, the Fahrstraßenausleuchtung, usually a white or green
light strip). A section can be part of a set route and still be free, or be occupied without
any route set.

`DerailerState.position` is `applied` (Gleissperre aufgelegt), `removed` or `unknown`.
`RouteState.status` is `not_set`, `set`, `locked`, `releasing` or `unknown`.

## Attachments: `EdgeAttachment`

Signals and derailers sit on an edge: `at: {edge, offset, direction}`. `offset` in `[0, 1]` is the
position along the edge from `a` (0.0) to `b` (1.0); `direction` (`a_to_b` | `b_to_a`) is the
travel direction the element governs. A signal at `{e1, 0.9, a_to_b}` stands near the `b` end of
`e1` and applies to trains travelling from `a` towards `b`, so a route starting at that signal
leaves `e1` through its `b` node.

## Routes and the walker (rule 13)

A `Route` has `start` (a signal id), `end` (a signal id or a node id), `path` (edge ids ordered
from start to end, at least one) and `switch_positions` (per switching node on the path: a
`SwitchPosition` string for a `switch`, a `[Port, Port]` path for a `dkw`/`ekw` — the same
vocabulary the state uses).

The validator walks every route:

1. `path[0]` must be the edge the start signal is attached to, else `ROUTE_DISCONTIGUOUS`.
2. The route leaves `path[0]` at its far end as seen from the signal's `direction`: the
   current node is `e0.b.node` for `a_to_b`, `e0.a.node` for `b_to_a`; the entry port is
   `e0`'s port at that node.
3. For every following edge `e`: `e` must be incident to the current node (else
   `ROUTE_DISCONTIGUOUS`, and the walk stops). The pair `{entry port, e's port at the node}`
   must be an allowed path of the node's kind (else `ROUTE_INVALID_TRAVERSAL` with
   `element_id` = that node). If the node is a switching kind, the declared setting in
   `switch_positions` is compared with `expected_setting(kind, pair)`: missing ->
   `ROUTE_SWITCH_MISSING` (warning), wrong type for the kind or a different value ->
   `ROUTE_SWITCH_MISMATCH`. Then the walk moves to the other end of `e`.
4. End check: if `end` is a signal, it must be attached to `path[-1]`; if it is a node, it must
   be the node where the walk ended; otherwise `ROUTE_DISCONTIGUOUS`.
5. Keys of `switch_positions` that are not switching nodes on the walked path ->
   `ROUTE_SWITCH_MISMATCH`.

Parallel edges between the same two nodes are handled because the walk tracks the current node
and the port of each edge at that node, not just node identities. The walk needs the start
signal's edge and every path entry to exist; dangling references are reported by rule 2 and the
walk is skipped for that route.

## Issue codes

`validate_document` first runs the JSON Schema pass, then pydantic, then the semantic rules;
when the structural passes fail, only `SCHEMA_INVALID` issues are reported. Every issue carries
`code`, `severity`, `message`, an optional `element_id` and a slash-joined `path` such as
`topology/edges/2/a`. Reports are sorted by `(path, code, element_id, message)` and validation
never raises.

| Code | Rule | Severity |
| --- | --- | --- |
| `SCHEMA_INVALID` | JSON Schema or pydantic rejects the document; no semantic rule runs | error |
| `ID_DUPLICATE` | an id is used more than once across all element families (reported once per id) | error |
| `TOPO_DANGLING_REF` | a `PortRef.node`, `EdgeAttachment.edge`, route `start`/`end`/`path` entry or `switch_positions` key does not exist | error |
| `TOPO_PORT_UNKNOWN` | a `PortRef.port` is not a port of that node kind | error |
| `TOPO_PORT_DUP` | a (node, port) is used by more than one edge endpoint | error |
| `TOPO_DEGREE` | the number of edge endpoints at a node differs from the kind's degree (an edge-less node included) | error |
| `TOPO_SELF_LOOP` | an edge connects a node to itself | error |
| `NODE_HALF_LABELS_UNEXPECTED` | `half_labels` on a node that is not `dkw`/`ekw` | warning (never promoted) |
| `GEOM_OUT_OF_BOUNDS` | a geometry coordinate outside `[0, width] x [0, height]` | warning; **error in strict mode** |
| `STATE_UNKNOWN_ELEMENT` | a state key does not name an element of the matching family | error |
| `STATE_WRONG_FIELD_FOR_KIND` | switch state on a non-switching node, `active_paths` on a `switch`, `position` on a `dkw`/`ekw` | error |
| `STATE_PATH_NOT_ALLOWED` | an `active_paths` set is not physically possible for the kind | error |
| `STATE_MISSING` | a switch/dkw/ekw, edge, signal, derailer or route has no state entry | warning; **error in strict mode** |
| `ROUTE_START_NOT_SIGNAL` | `route.start` exists but is not a signal | error |
| `ROUTE_DISCONTIGUOUS` | path does not start on the signal's edge, breaks, or does not reach `end` | error |
| `ROUTE_INVALID_TRAVERSAL` | the route enters and leaves a node through a pair that is not an allowed path | error |
| `ROUTE_SWITCH_MISMATCH` | declared setting differs from the required one, has the wrong type, or names a node not switched on the path | error |
| `ROUTE_SWITCH_MISSING` | a switching node on the path has no declared setting | warning; **error in strict mode** |

### Strict mode

`bench validate --strict`, `validate_document(..., strict=True)` and `POST /validate?strict=true`
promote `STATE_MISSING`, `GEOM_OUT_OF_BOUNDS` and `ROUTE_SWITCH_MISSING` to errors. Ground
truth is validated strictly (CI runs `make validate-examples` in strict mode); the `validate`
tool and the harness default to the non-strict mode, the intent being that a prediction which
omits a state entry loses points on the state metrics rather than being rejected outright.
`tests/fixtures/strict/` holds documents that are ok in the default mode and fail in strict mode.

## The two examples

**`schema/examples/v0/minimal.json`** — the smallest strict-valid document: a 400x100 synthetic
image, nodes `n_w` (boundary) — `n_j` (joint, label `1-2`) — `n_e` (boundary), edges `e1`
(`n_w.A` → `n_j.A`) and `e2` (`n_j.B` → `n_e.A`), one main signal `sig_1` (H/V, label `A`) on
`e1` at offset 0.9 governing `a_to_b`, showing `proceed` with `raw: "Hp1"`, both tracks free
with `route_set: false`, no derailers, no routes.

**`schema/examples/v0/station_dkw.json`** — the reference station in an 800x400 image: main
line `n_w` → `sw1` (switch `1`) → `dkw1` (DKW `12`, halves `12a`/`12b`) → `sw2` (switch `2`) →
`n_e`, a platform track `dkw1.D` → `n_j1` → buffer stop `n_bs1`, a siding `sw1.diverging` →
`dkw1.B`, a siding `sw2.diverging` → buffer stop `n_bs2` with derailer `Gs1`; five signals
(distant `a`, main `A`, `F`, `N1`, shunt `Ls1`); two routes: `rt_A_1` from `sig_A` over
`[e1, e3, e5, e6]` to `sig_N1` with `sw1: diverging` and `dkw1: [B, D]` (status `set`, the
tracks on it lit), and `rt_F_main` from `sig_F` over `[e7, e4, e2, e1]` to node `n_w` with
`sw2: straight`, `dkw1: [A, C]`, `sw1: straight` (status `not_set`). The DKW shows both
straights active (`[[A, C], [B, D]]`, `raw: "12a:+ 12b:+"`), `sw1` stands diverging (`-`),
`e8` is occupied and the derailer is applied.

The invalid fixtures in `tests/fixtures/invalid/` are these two documents with one targeted
edit each and are named after the primary code they trigger (a few edits unavoidably raise a
secondary code as well; the tests assert the named one).

## Versioning policy

`schema_version` is a JSON Schema `const` (`"v0"`), so a document can never silently pass as
another version. v0 is **mutable until the first dataset release**: models, this page and
`schema/v0.json` change together, and `bench schema export --check` (pre-commit hook, CI step and
unit test) guarantees the committed file matches the models. After the release, any breaking
change creates `schema/v1.json` with a new `Literal["v1"]`, new models and new examples, and v0
documents stay valid against v0. The `$id` of the published schema is
`https://github.com/joernmht/railControlVision/schema/v0.json`.
