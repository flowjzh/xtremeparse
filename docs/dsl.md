# The routing DSL

The router's entire output is a segment map plus count declarations in a
compact line DSL. The library parses it, validates it, and repairs it
with precise feedback; a host that overrides the router prompt
([prompting.md](prompting.md)) must elicit exactly this language — the
grammar below is a stable public contract. Breaking it is a breaking
change.

## Shape

Output is map lines first, then exactly one declaration line per
repeating (array) unit.

### Map lines

    <start>-<end> <dest>[,<dest>...]
    <start> <dest>            # a single chunk may omit "-<end>"

- `start`/`end` are chunk ids. Lines never overlap and together cover
  every chunk id in `0..top` exactly once; line ORDER carries no
  meaning — the ranges are explicit, and code sorts by start before
  validating (a diff reply's `+` insert lands at its own editing
  position).
- A destination is `<code>` or `<code>.<item>`:
  - `code` is a letter code from the legend the prompt carries —
    assigned by the library in unit order; `-` is reserved.
  - `.item` on a repeating unit is the instance's index, numbered across
    the WHOLE document in the order the map meets the instances — unless
    the unit's card declares a numbering order (e.g. reverse
    chronological), which overrides document order: the map's ascending
    lines then carry the card-ranked indexes, so a document listing the
    instances in the opposite direction puts the largest index on its
    first line.
- A bare repeating code (`4-9 x`) means several instances share the run
  unsplit — that material is extracted once, whole.
- A ranged run (`12-93 x.0-92`) is the compact shared form: those chunks
  carry exactly instances 0..92, inseparably — one destination instead
  of 93 comma-joined indexes, the run's material extracted once until a
  split round redraws it. The base prompt makes it mandatory for long
  entry-list runs (consecutive instances, roughly one to a chunk): the
  initial draw stays one short line — cheap to emit and, when a split
  round rewrites it, cheap to quote in a diff. Ranges take their line
  alone — never comma-joined — unless the range shares a SINGLE chunk
  with another unit's item (`5 x.0,y.0-y.2`): one chunk holding
  instances a..b is the co-chunked shared form spelled compactly, and
  the parser spells it back out. A multi-chunk range never comma-joins
  — the other unit's item would ride every block. No instance may
  appear in two ranges.
- A run may feed several DIFFERENT units at once, comma-joined
  (`5 x.0,y.0`) — a summary or cross-cutting unit rides the lines of the
  unit whose text it shares, item by item. An instance whose own text
  sits inside another unit's run rides that line too, even when its
  unit's other instances get lines of their own (`0-1 e.0,a.0` then
  `2 a.1`): a chunk holding two units' material is ONE line carrying
  both codes — two lines claiming the same chunk are never legal.
- Items of the same unit that separable chunk boundaries CAN separate
  must each get their own line — that is what fans the unit out into
  parallel per-item calls with per-item budgets. Several items share
  one line (`3 x.0,x.1,x.2`) ONLY when chunk boundaries cannot
  separate the instances — one chunk holding material of two or more
  of them; that material is extracted once, whole.
- `-` on its own (never comma-joined) marks chunks irrelevant to every
  unit. A bare section heading (a title introducing the entries after
  it, no instance and no field-bound text of its own) takes `-`.

### Declaration lines

After the map, every repeating unit gets exactly one of:

    x: <count> [@<budget>[,<budget>...]]
    x = <source> [@<budget>]

- `x: <count>` — the document holds `<count>` instances of the unit.
  Item indexes used in the map must run exactly `0..count-1` with no
  gaps, and every declared item must receive at least one chunk.
- `x = <source>` — the unit has no text of its own; its items mirror
  `<source>`'s, a directly mapped repeating unit. Such a unit takes NO
  map lines.
- The budget suffix is tolerated, never validated: one entry per item
  in item-index order (`@500,300,150`), or one entry covering every item.
  Three forms: `@<n>` an absolute character cap, `@<n>%` a ratio of the
  item's mapped material (a verbatim copy is `@100%`), and
  `@<avg>x<count>` an average keyword length times a keyword count.
  Malformed entries are dropped silently — the count still parses, no
  error.

## What validation enforces

Violations are fed back as repair errors (bounded rounds, then
`RouterError`):

- map lines parse, do not overlap, and cover `0..top` exactly (line
  order is free — code sorts by start)
- the item set each unit uses in the map equals `0..declared-1`
- every repeating unit is declared (a missing declaration reads as 0,
  which triggers a separate fresh-conversation recount before it is
  trusted)
- a derivation's `source` is itself a directly mapped repeating unit

Tolerated noise: counts on non-array units are ignored; an exact
duplicate map line collapses; a dotted tail on an item (`2.0`) keeps its
leading index; an item range that repeats the code on its right end
(`5-9 d.0-d.2`) normalizes to `d.0-2`.

## Budgets

The suffix is the router's arrangement — its estimate of the value
characters each item's extraction will run to: the leaf values' own
text, never key names, punctuation, or JSON structure. It declares
the estimate in the form that matches the extraction's
shape:

- `@<n>` — an absolute cap, for fixed-length summaries and anything
  else.
- `@<n>%` — a ratio of the item's mapped material, for verbatim or
  unbounded-refinement extraction (`@100%` is a verbatim copy; one
  shared ratio scales per item against each item's own material —
  co-chunked items' material splits across them).
- `@<avg>x<count>` — an average keyword length times the document's
  keyword count, for lists of short same-shaped entries. One shared
  entry is the document-level total: the resolver splits it across the
  items it covers, and audits judge the total, not the shares.

The library resolves every form to absolute value characters against
the routed material before use; the declared forms ride in the
trace's router budgets verbatim. Budgets shape output through batch
(consecutive small items coalesce into one shared call while their
accumulated budget fits the per-call capacity) and are never enforced
on the output. Overruns are accepted — a retry would cost a full
extra decode. A host's `output_budgets={'path': n | [n, ...]}`
overrides the router's declarations per path.

## Repair rounds

Violations come back as bounded repair rounds, and a reply decodes in
this same language: whatever text the round leaves behind — a diff
applied, a full re-emission — parses as a fresh answer, so every rule
above holds of the RESULT, not the patch. The loops themselves — what
fires them, what they ask, when they decline — are library machinery,
not grammar: [repairs.md](repairs.md).
