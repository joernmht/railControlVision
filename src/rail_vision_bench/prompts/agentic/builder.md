# agentic/builder

Node `builder` of the agentic loop.

## Reads

`scene_id`, `image` (its width and height), `interpretation`, `geometry`.

## Writes

`candidate`: one complete schema v0 document as a JSON object, or null when
nothing could be assembled.

## Instructions

Assemble the document; do not re-interpret the image.

1. `schema_version` is `"v0"`, `scene_id` is the given id, `source` carries
   the image kind, path, width and height, `provenance.kind` is
   `"prediction"` with the model id.
2. `topology.nodes` come from `interpretation.nodes` (id, kind, label,
   `half_labels` only for dkw/ekw, `geometry.point` or `bbox`), and
   `topology.edges` from `geometry.edges` (id, kind, `a`, `b`,
   `geometry.polyline`).
3. `signals` and `derailers` combine the interpretation entries with their
   attachment from `geometry.attachments` (`at.edge`, `at.offset`,
   `at.direction`).
4. `routes`: only when route lights show a set route, with `start` the signal
   id, `end` a signal or node id, `path` the ordered edge ids from the start
   signal's edge, and `switch_positions` for every switching node on the
   path (`"straight"`/`"diverging"` for a switch, `["A", "C"]`-style paths
   for a dkw/ekw) so that the route is physically possible.
5. `state`: `switches` for every switch, dkw and ekw node (position or
   active_paths plus raw), `signals` for every signal (aspect, raw), `tracks`
   for every edge (occupancy, route_set), `derailers` and `routes` for every
   derailer and route. Every listed element gets an entry; unknown values are
   `"unknown"` or null.
6. Drop any node whose ports are not all connected instead of inventing
   edges; drop any signal without an attachment. Ids must be unique across
   all element lists and match `^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$`.
7. Keep every coordinate within `[0, width] x [0, height]`.
8. Put anything you could not place (station name, unreadable symbols) into
   `meta` as notes.

## Stop condition

Hand the candidate to the critic. Never call the validator yourself; the
critic does.
