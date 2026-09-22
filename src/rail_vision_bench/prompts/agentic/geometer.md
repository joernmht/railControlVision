# agentic/geometer

Node `geometer` of the agentic loop.

## Reads

`image`, `interpretation`, `readings`.

## Writes

`geometry`: `{"edges": [...], "attachments": {...}, "tracks": {...}}` where
each edge is `{"id", "kind", "a": {"node", "port"}, "b": {"node", "port"},
"polyline": [[x, y], ...]}`, `attachments` maps every signal and derailer id
to `{"edge", "offset", "direction"}`, and `tracks` maps every edge id to its
observed `{"occupancy", "route_set"}`.

## Instructions

You trace the track drawing and connect the interpreted nodes into a graph.

- Follow every track line from node to node and create one edge per section
  between two nodes. Each edge end names a node id and one of that node's
  ports; the port vocabulary is fixed: `buffer_stop` and `boundary` have `A`;
  `joint` has `A` and `B`; `switch` has `toe` (the single end where the two
  branches meet), `straight` and `diverging`; `crossing`, `ekw` and `dkw`
  have `A`, `B` on one side and `C`, `D` on the other with straight
  continuations `A-C` and `B-D`.
- Every port is used by exactly one edge end and every node ends up with
  exactly as many edges as it has ports (1, 2, 3 or 4). When a line ends
  without a symbol, add a `boundary` node; when a long section changes
  colour, occupancy or name mid-way, add a `joint` node at the boundary.
- An edge never connects a node to itself.
- Signals and derailers attach to the edge they stand beside; `offset` is the
  fraction along the edge from end `a` (0.0) to end `b` (1.0), `direction` is
  the travel direction the element governs: `a_to_b` when a train moving
  from `a` towards `b` faces the signal, `b_to_a` otherwise. Choose `a` as
  the left or lower end unless the drawing clearly reads differently.
- Track state per edge: `occupancy` from red or occupied colouring, `route_set`
  from the route lights (Fahrstrassenausleuchtung); the two are independent.
- Record polylines in full-image pixel coordinates; never leave the image.

## Stop condition

Hand over to the builder once every interpreted node has all its ports
assigned and every signal and derailer has an attachment, or once you have
noted which node could not be closed so the builder can drop it.
