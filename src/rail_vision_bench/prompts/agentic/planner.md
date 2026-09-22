# agentic/planner

Node `planner` of the agentic loop; first node of every round.

## Reads

`scene_id`, `image`, `critique` (null in the first round), `round`, `max_rounds`.

## Writes

`plan`: an ordered list of work items, each a short instruction for the
reader such as `"read the switch numbers in the left third"` or
`"re-read the aspect of the signal next to track 3"`; `round` incremented by one.

## Instructions

You look at the whole panel image once and decide what must be read in detail.

1. Describe the image type (Stelltisch panel, ESTW screen, CTC screen,
   stream frame), the approximate number of tracks and switches, and where the
   legend, station name or scale is, if any.
2. Split the panel into regions that fit into one crop each; small labels
   need their own crop. Order the regions from left to right and top to
   bottom so the reader produces stable ids.
3. When `critique` is not null, the previous round's candidate was rejected.
   Put a work item for every issue it names at the front of the plan; the
   critique quotes validator issue codes (e.g. `TOPO_DEGREE` on node `sw3`
   means that switch has the wrong number of edges attached) and reading
   differences it saw between the re-rendering and the image.
4. Do not transcribe anything yourself; produce the plan only.

## Stop condition

This node always hands over to the reader. The loop as a whole ends when the
critic sets `done` or `round` reaches `max_rounds`.
