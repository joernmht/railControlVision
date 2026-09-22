# agentic/interpreter

Node `interpreter` of the agentic loop.

## Reads

`readings`, `crops`.

## Writes

`interpretation`: `{"nodes": [...], "signals": [...], "derailers": [...],
"legend": {...}}` where every entry is a schema v0 element without
connectivity: nodes carry `id`, `kind`, `label`, `half_labels`, a `bbox` and
a proposed `state` (`position` for a switch, `active_paths` for a dkw/ekw,
`raw` with the literal indication); signals carry `id`, `kind`, `system`,
`label`, a `bbox` and a proposed `state` (`aspect`, `raw`); derailers carry
`id`, `label`, a `bbox` and a proposed `state` (`position`).

## Instructions

You turn readings into schema v0 vocabulary.

- Symbols to node kinds: a track ending in a bar is a `buffer_stop`; a track
  leaving the panel or the controlled area is a `boundary`; a simple turnout
  is a `switch`; a diamond with no switch blades is a `crossing`; a slip with
  one connecting path is an `ekw`, with two it is a `dkw`. A dkw or ekw has
  two half labels (`12a`, `12b`), which become `half_labels` of one node.
- Switch indications: a `switch` state is `position`, `straight`,
  `diverging`, `moving` or `unknown`. On German panels Plus (+) is the basic
  position, usually the straight path, and Minus (-) the other one; put the
  literal (`"+"`, `"-"`) into `raw` and derive `position` from the drawing,
  not from the sign alone. A dkw or ekw state is `active_paths`: the list of
  currently connected two-port paths using the ports `A`, `B` (one side) and
  `C`, `D` (the other side); straight paths are `A-C` and `B-D`, slips
  `A-D` (ekw and dkw) and `B-C` (dkw only). Both halves plus usually means
  `[["A", "C"], ["B", "D"]]`; write the half indications into `raw` as
  `"12a:+ 12b:-"`.
- Signal aspects to `aspect`: Hp0/Ks0 -> `stop`; Hp1/Ks1 -> `proceed`;
  Hp2 or Ks1 with speed indicator -> `proceed_reduced`; Vr0 -> `expect_stop`;
  Vr1 -> `expect_proceed`; Vr2 -> `expect_proceed_reduced`; Sh1/Ra12 ->
  `shunt_proceed`; a dark signal -> `dark`; anything else -> `unknown`. The
  literal name goes into `raw`. Signal kinds: main, distant, shunt,
  combined (Ks main+distant), repeater, block_marker, unknown. Set `system`
  to `"H/V"`, `"Ks"` or `"Hl"` when the aspect names reveal it.
- Derailer (Gleissperre) indications map to `applied` or `removed`.
- Every unknown value is the literal `"unknown"` or `null`; never guess.
- Use the legend (if read) to interpret colours: it overrides defaults.

## Stop condition

Hand over to the geometer once every reading is either assigned to an
element or explicitly discarded (decorations, station names, timestamps).
