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
Two degradations are accepted without ceremony: a reply with no diff
markers parses as a full re-emission, and an empty reply declines the
suggestion (the previous map stands).

## Fan-out suggestions

Two suggestion rounds exist, one round each per routing — the map was
valid before they fire, so they are asks, not repairs:

- Material overflow. A shared run whose mapped material exceeds one
  shared call's capacity (~1000 content chars) is asked to split: the
  router computes the even partition itself and hands the model the
  block lines to ratify — one ranged line per call-sized block, the
  block count sized from the unit's own arranged budget over the
  executor's per-call cap. The reply is a dozen transcription lines
  where an instance-enumerated split would be ~85, and the model never
  computes block boundaries at all (self-computed boundaries across a
  long run's junk headings are measured round-losers: an overlapping
  block sent the model into per-instance enumeration).
- Merged instances. Instances sharing a run far longer than their
  count get a fresh-conversation recount — count plus each instance's
  opening quotes, the one form that survives without the map — and
  code re-splits the run at the quoted openings; adoption is code's,
  not the model's. A recount that comes back unusable falls back to
  one diff round.

A fan-out round is a suggestion: an empty reply declines it, and so
does an answer that fails validation — the standing map was valid
before the ask, and the model never sees a repair prompt about its
own rejected suggestion.

## The zero-count recount

A missing declaration reads as 0, and a zero is never trusted on the
map's own say-so: a fresh conversation without the map re-counts the
zero units, and code splices the answer in — the anchored model would
re-emit its own map verbatim. Unlike the fan-out rounds this
disagreement must be resolved, not declined: a fix attempt that fails
validation keeps the repair loop, because the alternative is trusting
the suspect zero.

## Executor corrections

Corrections ask for a JSON Patch (RFC 6902) against the call's
previous result: an array of `{op, path, value}` operations, `add`
(with `"/<index>"`, `-` appending), `remove`, `replace` (plus `move`,
`copy`, `test`). Two tolerances, both observed in the wild: a pointer
may be rooted at the unit path (`/jobs/2` for a bare `/2`), and a
reply that is not a patch at all applies as the full corrected value —
the pre-patch semantics, so a draw that ignores the protocol degrades
gracefully. A patch that fails to apply keeps the previous result and
the next round re-asks in full; an empty operation list is a no-op,
never a wipe.
