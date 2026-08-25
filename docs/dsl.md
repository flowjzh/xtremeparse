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

- `start`/`end` are chunk ids. Lines ascend, never overlap, and together
  cover every chunk id in `0..top` exactly once.
- A destination is `<code>` or `<code>.<item>`:
  - `code` is a letter code from the legend the prompt carries —
    assigned by the library in unit order; `-` is reserved.
  - `.item` on a repeating unit is the instance's index, numbered across
    the WHOLE document in the order the map meets the instances.
- A bare repeating code (`4-9 x`) means several instances share the run
  unsplit — that material is extracted once, whole.
- A run may feed several DIFFERENT units at once, comma-joined
  (`5 x.0,y.0`) — a summary or cross-cutting unit rides the lines of the
  unit whose text it shares, item by item.
- Several items of the SAME unit may share one line (`3 x.0,x.1,x.2`)
  when chunk boundaries cannot separate the instances.
- `-` on its own (never comma-joined) marks chunks irrelevant to every
  unit.

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

- map lines parse, ascend, do not overlap, and cover `0..top` exactly
- the item set each unit uses in the map equals `0..declared-1`
- every repeating unit is declared (a missing declaration reads as 0,
  which triggers a separate fresh-conversation recount before it is
  trusted)
- a derivation's `source` is itself a directly mapped repeating unit

Tolerated noise: counts on non-array units are ignored; an exact
duplicate map line collapses; a dotted tail on an item (`2.0`) keeps its
leading index.

## Budgets

The suffix is the router's arrangement — its estimate of the characters
each item's output JSON will run to (keys and punctuation included).
It declares the estimate in the form that matches the extraction's
shape:

- `@<n>` — an absolute cap, for fixed-length summaries and anything
  else.
- `@<n>%` — a ratio of the item's mapped material, for verbatim or
  unbounded-refinement extraction (`@100%` is a verbatim copy; one
  shared ratio scales per item against each item's own material). The
  resolved estimate adds the item schema's skeleton (key names and
  punctuation, computed from the schema) on top — the ratio covers the
  content, the structure is code-known.
- `@<avg>x<count>` — an average keyword length times the document's
  keyword count, for lists of short same-shaped entries. One shared
  entry is the document-level total: the resolver splits it across the
  items it covers.

The library resolves every form to absolute characters against the
routed material before use; the declared forms ride in the trace's
router budgets verbatim. Budgets shape output through batch scheduling
(consecutive small items coalesce into one shared call while their
accumulated budget fits the per-call capacity) and are never enforced
on the output. Overruns are accepted — a retry would cost a full
extra decode. A host's `output_budgets={'path': n | [n, ...]}`
overrides the router's declarations per path.
