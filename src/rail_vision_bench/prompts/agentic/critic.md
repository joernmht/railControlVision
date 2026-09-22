# agentic/critic

Node `critic` of the agentic loop; last node of every round.

## Reads

`candidate`, `image`, `round`, `max_rounds`.

## Writes

`validation`: the validate tool's report (`ok`, `issues` with `code`,
`severity`, `message`, `element_id`, `path`).
`critique`: a short list of concrete corrections for the next round, or null.
`done`: true when the candidate is accepted.

## Instructions

1. Run the validate tool on the candidate and store its report in
   `validation`. Read every issue:
   - `SCHEMA_INVALID`: the JSON does not match schema v0 at `path`; the
     builder must fix the shape.
   - `ID_DUPLICATE`, `TOPO_DANGLING_REF`: ids clash or refer to elements that
     do not exist.
   - `TOPO_PORT_UNKNOWN`, `TOPO_PORT_DUP`, `TOPO_DEGREE`, `TOPO_SELF_LOOP`:
     the graph is not physically possible; name the node and what the
     geometer must re-trace.
   - `STATE_UNKNOWN_ELEMENT`, `STATE_WRONG_FIELD_FOR_KIND`,
     `STATE_PATH_NOT_ALLOWED`: a state entry refers to a missing element, uses
     `position` on a dkw/ekw (or `active_paths` on a switch), or lists a
     path set the node cannot connect.
   - `ROUTE_*`: the route's path is not contiguous from its start signal, a
     traversal is impossible at a node, or the declared switch positions do
     not match the path.
   - `STATE_MISSING`, `GEOM_OUT_OF_BOUNDS`, `ROUTE_SWITCH_MISSING`,
     `NODE_HALF_LABELS_UNEXPECTED` are warnings: mention them so the next
     round fills the gaps, but they do not reject the candidate by themselves.
2. Run the render tool on the candidate and compare the re-rendering with the
   original image: count switches, signals and track ends per region, check
   that labels and indications match the readings. Differences become
   critique items as precise as `"signal N1 reads Hp0 in the image but the
   candidate says proceed"`.
3. Decide: set `done` to true when the report has no error-level issue and
   the re-rendering matches the image; otherwise write the critique. The
   loop stops regardless when `round` reaches `max_rounds`, so in the last
   round prefer a candidate with warnings over none.

## Stop condition

The router ends the run when `done` is true or `round >= max_rounds`;
otherwise the planner starts the next round with your critique.
