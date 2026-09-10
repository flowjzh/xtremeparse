# The repair loops

Two loops re-contact the model after a first answer, and both ask for
a diff against what it already answered — a rewrite re-decodes
everything and can collapse entries that were correct (measured: a
retry asked for one missing entry returned none). The grammar those
answers must speak is [dsl.md](dsl.md); this doc is the machinery
around it — what fires each loop, what it asks, when it declines.
Hosts cannot reword them ([prompting.md](prompting.md)): repair and
recount feedback is generated from validation errors, never prose —
they shape what is retried through the validator's issues.

## Router repairs

Hint rounds and invalid-map rounds alike ask for a unified diff
against the model's own last map text: `-` lines remove, `+` lines
add, one edit per line; `@@` headers and context lines never touch
the map, and line numbers are hints — the applier anchors on content,
since the diff's lines are the model's own previous answer quoted
back. Edits cannot regress lines the model already fixed, a quote
that misses costs nothing, and the patched text parses as a fresh
answer, so every grammar rule holds of the RESULT, not the patch.

A replay — the same applied map text twice in a row — ends the
repairs for count mismatches: re-asking cannot move a number the
model already could not localize (measured: every line re-quoted as
-x/+x until the budget burned), so the declared-vs-used error passes
through and the extraction's count arbitration owns the number. The
path there starts one notch earlier: a verbatim -x/+x pair applies as
no edit at all — the lines would only re-append, reordering a map
whose line order carries no meaning — and a reply that still repeats
the map is told so, that its pairs cancel out and the full corrected
map is the escape hatch. Only count mismatches pass; a replayed map
with any other error (coverage, syntax, an undeclared unit) still
burns the budget. The budget's end settles rather than raises, on
the same trade: the last parse-valid round finalizes, and a
never-valid map whose only remaining flaws are count mismatches
passes them to the arbitration too. A geometry flaw (an overlap, a
coverage hole) still raises — chunks would be double-claimed
silently, and the caller's fresh draw is the cure for an oscillating
anchor (measured: shown the two conflicting lines by name, the model
re-emits the corrected map in full within the budget).

Four degradations are accepted without ceremony: a reply with no diff
markers parses as a full re-emission, a reply of chain lines alone
reads as the patch it means (asked to add the chain lines, the model
answers with the chain lines alone — replacement would drop the map's
head and spend the next round re-typing lines that were never wrong),
a bare line beside real markers is the lazy "+" it reads as — the
contract bans re-emitting unchanged lines, so a line that speaks the
grammar is an addition, and one the map already holds dedupes to a
no-op (dropped as commentary it once silently zeroed a unit while the
removal beside it landed, and the phantom burned the budget) — and an
empty reply declines the
suggestion (the previous map stands).

## Fan-out suggestions

Two suggestion rounds exist, one round each per routing — the map was
valid before they fire, so they are asks, not repairs (when a merged
unit and a zero-declared unit pend together, their two recounts share
one fresh conversation — see the zero-count recount below):

- Material overflow. A shared run whose mapped material exceeds one
  shared call's capacity (~1000 content chars) is split: the router
  computes the even partition itself — one ranged line per call-sized
  block, the block count sized from the unit's own arranged budget
  over the executor's per-call cap — and splices the block lines into
  the standing map itself; the ask round typed them verbatim every
  time, a round spent re-taking code's dictation. The ask round
  survives only for a run code may not redraw — a line inside it
  carrying another unit's destination or a chain — where the model
  ratifies the computed lines or declines by silence (the model never
  computes block boundaries at all: self-computed boundaries across a
  long run's junk headings are measured round-losers — an overlapping
  block sent the model into per-instance enumeration).
- Merged instances. Instances sharing a run far longer than their
  count get a fresh-conversation recount — count plus each instance's
  opening quotes, the one form that survives without the map — and
  code re-splits the run at the quoted openings; adoption is code's,
  not the model's. A recount that comes back unusable falls back to
  one diff round — fired alone, that is: beside a zero unit's recount
  (below) the two share one fresh conversation, the chunks listing
  being the costly part, and once the zero's answer is spliced in
  there is no diff round at all — a run whose own sections came back
  unusable simply stays shared, a diff round then would re-parse the
  pre-split text and discard the zero's adoption. A line that already
  carries one destination per
  instance is decomposed — its recount could only shave decode, so it
  fires only past a load bar, and a unit declared once obeys the same
  arithmetic (a lone item over a handful of chunks reads as one long
  entry as plausibly as a merge: recounted, confirmed, and never
  adopted while thin — the recount is for the merge, not the
  spelling).
- Merged sub-entries. A lifted sub-array's sub-entries riding one line
  together — a ranged chain or comma-joined chains of one parent
  instance — are the merged shape the previous bullet cannot see:
  chain destinations are excluded there, a redraw would swallow the
  chain. One destination per line stays out of the pool: that fine
  partition is a legal draw, and on genuine multi-stint files its
  chains run long — recounted there, a miscount trims real
  sub-entries (measured: 4 true stints demoted to 3).
  They recount in the same fresh conversation, asked per parent
  instance, and when no top-level unit pends beside them the chunks
  listing is the parents' own material alone — a whole-document
  listing let a fresh count stray into another parent's look-alike
  material. Adoption is code's and per parent: the openings that
  anchor in the parent's footprint ARE its sub-entries — a quote
  landing elsewhere is another parent's instance, dropped — and they
  partition the footprint at their anchors; a recount that names none
  of them and says zero strips the chains and the declaration together
  (the parent keeps its chunks); openings all missing the
  footprint beside a positive count are no evidence to unmake the map —
  the lines stand. A count ABOVE the declaration is refused too,
  anchored or not: an upward read is no demotion (measured: a recount
  reading duty paragraphs as openings re-split 2 true stints into 4),
  the recount's only mandate is trimming — the whole-array addendum's
  card-deference clause keeps the specialist from padding the array,
  so trimming-only stays safe. The
  footprint arithmetic gates it: the footprint
  must outrun the declared count, or the recount could only confirm —
  an inseparable pair on one chunk is a legal draw. Demotion is the
  point: the titles of a narrated
  progression inside one band draw like sub-entries and are not.

A fan-out round is a suggestion: an empty reply declines it, and so
does an answer that fails validation — the standing map was valid
before the ask, and the model never sees a repair prompt about its
own rejected suggestion.

## The zero-count recount

A missing declaration reads as 0, and a zero is never trusted on the
map's own say-so: a fresh conversation without the map re-counts the
zero units, and code splices the answer in — the anchored model would
re-emit its own map verbatim. When a merged unit pends beside the
zero one, the two share that one conversation: the merged unit's
instances quote their openings, the zero unit claims map lines or
confirms 0 — a host's own recount instructions keep the two asks
apart. Unlike the fan-out rounds this
disagreement must be resolved, not declined: a fix attempt that fails
validation keeps the repair loop, because the alternative is trusting
the suspect zero — so in the shared conversation the zero folds
first, before anything is adopted: the diff round a failed splice
forces re-parses the model's text, and an adoption taken before it
would be lost. A confirming answer costs no round: the splice
re-segments through the per-chunk cover, which keeps a parent's
destinations beneath the chain chunks nesting in its run, so even the
covering form (parent line drawn over its chains) round-trips the
adoption and the zero is accepted as counted. The recount sometimes
answers the template's quoted example form with the quotes on
(`"2 b.0"`) — a claim is read through a wrapping quote pair when
unquoting makes it parse.

## Executor corrections

Corrections ask for a JSON Patch (RFC 6902) against the call's
previous result: an array of `{op, path, value}` operations, `add`
(with `"/<index>"`, `-` appending), `remove`, `replace` (plus `move`,
`copy`, `test`). Two tolerances, both observed in the wild: a pointer
may be rooted at the unit path (`/jobs/2` for a bare `/2`), and a
reply that is not a patch at all applies as the full corrected value —
the pre-patch semantics, so a draw that ignores the protocol degrades
gracefully. A patch that fails to apply keeps the previous result and
the next round re-asks in full. An empty operation list is the no-fix
declaration: the paths it was asked for are treated as absent and never
asked again — a no-op on apply, never a wipe. One exception to the diff
discipline: a call whose previous result is an empty array re-asks in
full with no history — nothing to diff, and shown its own `[]` the
model re-declares it instead of looking again (measured).
