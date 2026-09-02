"""Router: one light agent call mapping chunk ids to extraction units.

Output is a compact segment DSL — the map first (``<start>-<end>
<code>[.<item>][,...]``, one line per contiguous run of chunks, units
compressed to letter codes, ``-`` for irrelevant chunks), then a count
declaration per repeating unit after it (``<code>: <items>``,
optionally suffixed with a per-item output estimate — ``@<n>`` an
absolute cap, ``@<n>%`` a ratio of the item's mapped material, or
``@<n>x<m>`` an average keyword length times a keyword count — that
the executor treats as an arrangement). The map enumerates: the model
numbers a repeating unit's instances as it meets them, so the count is
derived from the map rather than committed up front — a count-first
declaration made the model report one fewer instance (counting is a
weak operation) and then obey its own wrong number, silently dropping
the section's tail entry. A run may feed several units at once
(comma-joined codes) — a summary or cross-cutting unit reading the
same text a detailed unit extracts; exclusive partitioning broke on
exactly that shape. A starred bare run (``12-93 d*``) marks an
entry-style run — every chunk holds exactly one instance — and code
splits it one item per chunk, ascending, its count read off the chunk
total: the model judges instance-per-chunk shape (easy, it sees the
chunks) instead of enumerating every index (weak), and code never
mistakes the run for an unsplit whole. A valid map that leaves fan-out
on the table — a long run shared while its declared count equals the
run's chunk total, or instances sharing one run far longer than their
count — gets its fix in a round of its own: the star hint asks the
model to star via a diff (reply empty to keep it shared; measured:
asked, the model stars the dense lists it had mapped shared), and the
merged form gets a fresh-conversation recount that code re-splits by.
The declared counts make the map self-consistent: item indexes must
run exactly 0..declared-1, every declared item must receive chunks.
Coverage and disjointness hold per line range (line order itself
carries no meaning — ranges are explicit), uniqueness per destination;
violations get bounded repairs with precise feedback (each repair a
diff against the previous answer), then RouterError — a validated
map's ranges are disjoint by construction.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional

from xtremeparse.arbitration import (fresh_check, norm,
                                    sectioned, unmarked)
from xtremeparse.contracts import AgentRunner, JSONSchema
from xtremeparse.units import Unit, type_set, value_branches

NONE = '-'
STAR = '*'  # a starred bare run: one instance per chunk, split by code
STAR_HINT_MIN = 5  # below this a shared whole call costs seconds — the
# confirmation round would cost more than the split saves
# placeholders every prompt template must carry (host overrides are
# validated against these; see docs/prompting.md)
ROUTE_PLACEHOLDERS = frozenset({'top', 'none', 'legend', 'chunks'})
RECOUNT_PLACEHOLDERS = frozenset({'legend', 'chunks'})
_LINE = re.compile(r'^(\d+)(?:-(\d+))?\s+('
                   r'[a-z]+\*?(?:\.\d+)?(?:\s*,\s*[a-z]+\*?(?:\.\d+)*)*'
                   r'|-(?:\.\d+)?)$')
_COUNT = re.compile(r'^([a-z]+)(?::\s*(\d+))?(?:\s*=\s*([a-z]+))?(?:\s*@.*)?$')
_BUDGET = re.compile(r'@\s*([0-9][0-9xX%,\s]*)')


def _budget(token: str):
    """One declared suffix entry: 'n%' a ratio of the item's mapped
    material, 'nxm' an average keyword length times a keyword count,
    'n' an absolute cap — every form counts extracted value
    characters only, never keys or punctuation. None on garbage — the
    suffix is tolerated, never checked."""
    if m := re.fullmatch(r'(\d+)%', token):
        return Budget('ratio', int(m.group(1)))
    if m := re.fullmatch(r'(\d+)[xX](\d+)', token):
        return Budget('kw', int(m.group(1)), int(m.group(2)))
    if token.isdigit():
        return Budget('abs', int(token))
    return None

def budget_kind(token: str) -> Optional[str]:
    """The declared suffix's form — 'abs', 'ratio', or 'kw' — or None
    on garbage. The one grammar for budget tokens: routing parses the
    declaration and the audit reads the same tokens back from the
    trace."""
    b = _budget(token)
    return b.kind if b else None


_INSTRUCTIONS = '''You route document chunks to extraction units. Output the segment
map: one line per contiguous run of chunks ("4-9 x.1" — start-end
code, ".item" appended on repeating units; a bare repeating code
("4-9 x") means several instances share the run unsplit — they are
extracted whole; a single chunk may omit "-end"). Example codes below
are schematic letters, not units of this document. After the map, one
count line per repeating unit ("x: 3" plus its budget suffix)
declaring the instances the DOCUMENT holds — every repeating unit gets
one: 0 only for a unit truly absent, and a unit with no text of its
own that only summarizes another declares its source ("x = y")
instead.

Rules:
- As you write the map, number a repeating unit's instances in the
  order you meet them across the WHOLE document — a new item index per
  instance, early and late instances alike. When the unit's card
  describes a numbering order for its instances (e.g. 'number in
  reverse chronological order'), rank the instances under the card's
  rule FIRST, however the card defines that order, then write the
  map's ascending lines carrying those
  ranked indexes — a document that lists the instances in the
  opposite direction puts the LARGEST index on its first line; the
  count line after the map reports the highest index plus one, 0
  for a unit the document does not contain at all. A unit with no
  text of its own that only summarizes another unit declares just
  its source and takes NO map lines:
  "x = y @30%" — its items (and its count) mirror the source's
  material, but its budget is its own: set it for the lines the
  summary will emit — a ratio scales against the source material it
  summarizes, so a terse summary sits well below 100%.
- A run of entry chunks — each chunk holds exactly ONE instance of the
  unit, an entry list with one listing per chunk — MUST take the star
  mark: "12-93 d*". Code splits the run mechanically into one item per
  chunk in ascending order, and the unit's count is read off the
  run's chunk total — the count line stays, but the number is
  ignored. The unmarked bare run ("4-9 d") extracts its whole material
  in ONE stream, whatever the run's length — reserve it for instances
  that truly share their chunks and cannot be listed per chunk. Within
  one unit, star and numbered destinations cannot mix, and a star
  takes its line alone — never comma-joined with another code.
- A declaration line ends with an output-budget suffix — one entry per
  item in item-index order ("x: 3 @100%,80%,50%", or "x: 3 @100%" when
  items share one): an estimate of the characters the item's extracted
  VALUES run to — the values' own text, never key names, punctuation,
  or JSON structure. A ratio, "@<n>%", is the DEFAULT form — the
  values as a percentage of ALL the text the item's mapped lines
  carry, the parts no field can hold included — never just the
  relevant-looking parts. It fits
  every item whose output mirrors the material field by field: the
  same facts, carried into the schema's fields. Set the ratio from
  the field descriptions in the JSON Schema (the shared context):
  fields told to keep every listed entry or every figure keep their
  fields' own content; a verbatim copy of the whole material is
  "@100%". An item whose lines are mostly narration no field can
  hold sits well under half; an item whose lines are all
  field-bound content stays near full — when items differ like
  that, give each its own percentage instead of one shared number
  ("@30%,90%"). Fields told to summarize or compress sit well
  below (say 30-50%). A card saying an entry holds only certain
  fields prices THOSE fields. And fields that will restate one
  clause of the material (a name distilled beside a description
  carrying the same sentence) run that clause once PER field —
  their sum can exceed the material's own length; price the
  fields, not the source. Two
  shapes take a non-ratio form instead:
  - a list of names: average the entries' own lengths as they
    stand in the assigned chunks, and multiply by the number of
    entries the document holds. An entry's length is the field
    value's own text — the name alone, not the labels, headings,
    or grades printed beside it in the chunk. The length counts
    CHARACTERS, never words — word counts understate a name
    badly;
  - a fixed-length summary — the card or schema pins the output size
    ("one line", "about 100 characters") — or an item whose output
    does not mirror the material at all: a plain character estimate,
    "@<n>" ("x: 3 @300"). A value's cost is its own text alone:
    a date ≈7, a name ≈10, a one-line title ≈20.
  Every declaration line carries one.
- Map lines ascend, never overlap, and together cover EVERY chunk id in 0..{top}.
- A run may feed several DIFFERENT units at once: comma-join their
  codes ("5 x.0,y.0") — a summary or cross-cutting unit rides the lines
  of the unit whose text it shares, item by item:
      5 x.0,y.0
      6 x.1,y.1
  An instance whose own text sits inside another unit's run rides that
  line too, even when its unit's other instances get lines of their
  own ("0-1 e.0,a.0" then "2 a.1"): a chunk holding two units'
  material is ONE line carrying both codes — two lines claiming the
  same chunk are never legal.
- Items of the SAME unit that separable chunk boundaries CAN separate
  MUST each get their own line ("5-9 x.0" then "10-11 x.1"): each
  item's call then fans out on its own and its budget scales against
  its own material. Sharing one line ("3 x.0,x.1,x.2") is ONLY for a
  run whose chunk boundaries cannot separate the instances — one chunk
  holding material of two or more instances; that material is then
  extracted once, whole.
- Every declared item receives at least one chunk. A unit sharing
  another's run may repeat one item across several lines (one instance
  spanning what another unit splits into many).
- Each item is ONE instance of the repeating unit — a complete entry.
  Start a new item whenever the text moves to the next instance; never
  merge distinct instances into one, and never split one instance
  across items.
- Code {none} on its own (no item) marks chunks irrelevant to every
  unit, always ranged ("42-43 {none}"), never a bare {none} line.
  A chunk that is only a label — a section heading or title introducing
  the entries around it, no instance and no field-bound text of its own —
  takes {none}, never a unit's item line.
- Before answering, verify: every repeating unit has a declaration line
  (its count "x: <n>" or its source "x = y"), the map covers 0..{top}
  exactly once in ascending non-overlapping lines, item indexes of
  each unit run 0..used-1 with no gaps, every count line matches
  the map, and a unit whose card declares an instance order numbers
  its items by that order.

Chunks:

{chunks}

Unit codes:
{legend}'''


_DIFF_HOWTO = '''
Reply with a unified diff against your previous map: "-" lines removed,
"+" lines added, one edit per line — never re-emit an unchanged line.
The full map is also accepted.'''
_DIFF_REPLY = _DIFF_HOWTO + ' An empty reply declines the suggestion.'
_DIFF_FIX = _DIFF_HOWTO + ' Fix every error named above.'


_CHECK = '''You are rechecking part of a routing decision. Some repeating units
were left at zero; recount their instances in the DOCUMENT — read the
whole chunk list; a brief mention inside an otherwise irrelevant run
is still an instance.

For each unit marked RECOUNT answer one line only — do not re-answer
units not marked RECOUNT, their counts and map lines are final:
- its source, "x = <code>", when the unit has no text of its own —
  its items just mirror another repeating unit's, one brief line per
  item of that unit: choose that unit's code from the list, never a
  count, never map lines;
- a count, "x: <n>", when the unit has its own text in the chunks —
  then add its map lines in the routing DSL ("4 x.0", "5 x.1"), one
  item per instance, covering exactly its material, numbered by the
  card's declared order when the card declares one.
0 only for a unit the document truly does not contain.

Repeating units (code = unit card; RECOUNT = left at zero):

Chunks:

{chunks}

Unit codes:
{legend}'''


class RouterError(Exception):
    """Raised when the router's segment map is still invalid after repairs."""


@dataclass
class Group:
    """One routed slice: a unit, an optional array item, and the
    deterministic concatenation of its chunks. Groups of one unit (and of
    one item) may coalesce into a single specialist call in the executor."""

    unit: Unit
    item: Optional[int]
    chunk_ids: list
    text: str


@dataclass
class Routing:
    groups: list
    raw: dict  # normalized assignments, counts, budgets and code legend, for the trace
    budgets: dict = None  # unit path → resolved per-item value-char
    # estimate (the declared forms ride in raw['budgets'])


@dataclass(frozen=True)
class Budget:
    """One declared output estimate: 'abs' a fixed character cap,
    'ratio' a percentage of the item's mapped material, 'kw' an
    average keyword length times a keyword count — every form counts
    the characters of the item's extracted VALUES only: key names,
    punctuation, and JSON structure are never the model's to
    estimate."""

    kind: str
    value: int
    count: int = 1  # kw only — the document's entry count

    def __post_init__(self):
        assert self.kind in ('abs', 'ratio', 'kw'), self.kind

    def chars(self, material: int) -> int:
        """One item's absolute estimate in value characters — a ratio
        scales against the mapped material (whitespace stripped) it is
        declared over, a keyword total is its own sum."""
        if self.kind == 'ratio':
            return round(self.value / 100 * material)
        if self.kind == 'kw':
            return self.value * self.count
        return self.value

    def per_item(self, material: dict) -> list:
        """The estimate for every item this declaration covers: a
        shared ratio scales against each item's own material, a shared
        keyword total splits across the items it covers, anything else
        is one number covering all (the executor repeats a lone
        value)."""
        if self.kind == 'ratio' and material:
            return [self.chars(material[i])
                    for i in sorted(material, key=lambda k: (k is None, k))]
        if self.kind == 'kw' and material:
            return [round(self.value * self.count / len(material))] * len(material)
        return [self.chars(0)]

    def __str__(self):
        if self.kind == 'ratio':
            return f'{self.value}%'
        if self.kind == 'kw':
            return f'{self.value}x{self.count}'
        return str(self.value)


@dataclass
class _RouteIssue:
    path: str
    code: str
    message: str
    expected: object = None
    got: object = None


def _listing(chunks: list[str]) -> str:
    """The numbered chunk listing both router prompts embed — one line
    per chunk: embedded newlines would inflate the listing's line count
    and the model's index arithmetic with it. The ' ¶ ' marker is a
    contract, not decoration: every quote anchor inverts it (see
    arbitration.norm), so reformatting it unanchors every model quote."""
    return '\n'.join(f'[{i}] {c.replace(chr(10), " ¶ ")}'
                     for i, c in enumerate(chunks))


def _name_list(unit: Unit) -> bool:
    """The unit's items are each one string value and nothing else —
    the mechanical shape of a list of names: either the item schema is
    a plain string (a scalar repeat) or an object carrying exactly one
    string field. The
    keyword budget form is decided HERE, in code, so the per-unit
    legend marker is a fact, not a judgement call the model can flip
    on. Array units only: a single-string ``$misc`` (a root schema
    with one scalar field) or a single-string summary unit is not a
    name list."""
    if unit.kind != 'array':
        return False
    sub = unit.sub_schema
    if 'string' in type_set(sub):
        return True
    if 'object' not in type_set(sub):
        return False
    props = sub.get('properties') or {}
    if len(props) != 1:
        return False
    branches = value_branches(next(iter(props.values())))
    return sum(b.get('type') == 'string' for b in branches) == 1 \
        and len(branches) <= 2


async def route(runner: AgentRunner, *, payload: str,
                units: list[Unit], chunks: list[str],
                instructions: str = None,
                recount_instructions: str = None) -> Routing:
    """Map chunks to units via one agent call, with bounded repairs —
    each a diff against the previous answer — feeding the validation
    errors back as ``feedback``; plus one
    verification round when any repeating unit comes out declared 0
    (see route loop). ``payload`` is the precomputed shared prefix
    (see prompting). ``instructions``/``recount_instructions`` replace
    the default prompt templates; they must carry their placeholders
    (ROUTE_PLACEHOLDERS / RECOUNT_PLACEHOLDERS — the Extractor
    validates host overrides at construction)."""
    by_code = _codes(units)
    instructions = (instructions or _INSTRUCTIONS).format(
        top=len(chunks) - 1, none=NONE,
        # card headers (not bare paths): the description is the cue for
        # what a unit collects — and for recognizing a unit that only
        # summarizes another (its card names the unit it mirrors)
        legend='\n'.join(f'{code} = {unit.header}'
                         + (' — a list of names: its budget is the keyword '
                            'form "@<avg>x<count>", never a percentage'
                            if _name_list(unit) else '')
                         for code, unit in by_code.items()),
        chunks=_listing(chunks))
    schema: JSONSchema = {'type': 'string',
                          'description': 'Segment map and count declarations '
                                         'only — no prose, no JSON.'}
    feedback, last, history, verified = None, None, [], False
    star_asked = False  # the star hint is asked once per routing
    shared_asked = False  # the shared recount likewise — independent of
    # the star hint (the conditions are mutually exclusive per unit)
    diff_base = None  # the previous round's map text — armed once below,
    # after every answer, so a future hint round cannot forget it
    # error lists: a repaired error that later reappears means the model
    # is rewriting fixed lines away — name it
    for _ in range(5):  # initial call + four bounded repairs
        result = await runner.run(instructions='' if last else instructions,
                                  result_schema=schema, content=payload,
                                  feedback=feedback,
                                  history=last.history if last else None)
        last = result
        base, text = diff_base, result.data
        if base is not None:
            if not str(text or '').strip():
                text = base  # an empty reply declines; the base stands
            elif (applied := _diff_text(text, base)) is not None:
                text = applied
        errors, counts, derived, segments, budgets = \
            _parse(text, by_code, len(chunks))
        # every answer leaves a map text behind — a full re-emission is
        # itself, a diff leaves its applied map, an empty reply the
        # standing base — and the next round diffs against it, hint or
        # repair alike
        diff_base = text
        if not errors:
            if not star_asked and (hints := _star_hints(segments, counts,
                                                        by_code)):
                # a run covering exactly as many chunks as the unit has
                # instances is probably an entry list left unstarred —
                # the costliest form (one whole-material stream). Ask
                # once: the model stars it via a diff, or stays silent
                # (empty declines; a full re-emission is still tolerated
                # and parses as ever). This round must run before the
                # shared recount: the diff applies to the model's own
                # answer text, so a resplit done first would be discarded
                # by the re-parse
                star_asked = True
                feedback = [_RouteIssue('segments', 'route_hint',
                                        h + _DIFF_REPLY) for h in hints]
                continue
            if not shared_asked and (shared := _shared_hints(segments, counts)):
                # instances merged into one long run: the anchored
                # conversation will not unmerge its own map (a repair
                # round shown the map re-emits it, measured — and the
                # splits it does make scatter). A fresh conversation
                # recounts every shared unit at once — count plus each
                # instance's opening text, the one form that survives
                # without the map — and code anchors the quotes back to
                # chunks and re-splits each run; adoption is code's, not
                # the model's. A unit whose recount is unusable falls
                # back to one diff round — only when nothing was
                # adopted, though: a diff re-parses the model's answer
                # text, so a round taken after a partial adoption would
                # discard the adopted splits (the pending unit then
                # simply stays shared)
                shared_asked = True
                answer = await _recount_shared(runner, payload, shared,
                                               by_code, chunks)
                pending = {}
                for code, (span, declared) in shared.items():
                    if merged := _resplit(segments, counts, code,
                                          answer.get(code),
                                          by_code, derived, chunks):
                        segments, counts = merged
                    else:
                        pending[code] = (span, declared)
                if pending and len(pending) == len(shared):
                    feedback = [_RouteIssue(
                        'segments', 'route_hint',
                        _split_hint(code, span, declared) + _DIFF_REPLY)
                        for code, (span, declared) in pending.items()]
                    continue
            if not verified and (zeros := [c for c, u in by_code.items()
                                           if u.kind == 'array' and c not in derived
                                           and counts.get(c) == 0]):
                # a zero is never trusted on the map's own say-so: the
                # model commits to its finished map and will not revisit
                # NONE'd material — not in the same pass, and not in a
                # feedback round that shows it that map (measured: brief
                # sections and summary units rubber-stamped as 0). A
                # fresh conversation without the map re-counts them, and
                # code splices the answer in — the anchored conversation
                # would re-emit its own map verbatim
                verified = True
                answer = await _recount(runner, payload, by_code, zeros, chunks,
                                        instructions=recount_instructions)
                if merged := _splice(segments, counts, derived, answer,
                                     zeros, by_code, len(chunks)):
                    segments, counts, derived = merged
                else:
                    note = (f'{", ".join(zeros)}: a separate recount of '
                            'the document disagreed with this map but '
                            'could not be merged — for each, recheck the '
                            'chunk list yourself: give every instance '
                            'its map lines with a matching count, or '
                            'declare "= <source>" if it only summarizes '
                            'another repeating unit; keep 0 only if '
                            'truly absent')
                    feedback = [_RouteIssue('segments', 'route_invalid',
                                            note + _DIFF_REPLY)]
                    history.append([note])
                    continue
            segments, _ = _expand_stars(segments)
            assignments = [
                {'unit': by_code[code].path, 'item': item,
                 'chunks': list(range(start, end + 1))}
                for start, end, destinations in segments
                for code, item in destinations if code != NONE]
            by_unit = {}
            for a in assignments:
                by_unit.setdefault(a['unit'], []).append(a)
            assignments += [{'unit': by_code[d].path, 'item': a['item'],
                             'chunks': a['chunks']}
                            for d, s in derived.items()
                            for a in by_unit.get(by_code[s].path, [])]
            budgets_by_path = {by_code[c].path: b for c, b in budgets.items()}
            by_path = _resolved(
                budgets_by_path,
                _material(assignments, chunks,
                          {by_code[d].path: by_code[s].path
                           for d, s in derived.items()}) if budgets_by_path else {})
            return Routing(_groups(segments, by_code, chunks, derived),
                           {'counts': {by_code[c].path: k for c, k in counts.items()}
                                      | {by_code[d].path: counts[s]
                                         for d, s in derived.items()},
                            'assignments': assignments,
                            'budgets': {p: ([str(v) for v in b]
                                            if isinstance(b, list) else str(b))
                                        for p, b in budgets_by_path.items()}}, by_path)
        past = set().union(*history[:-1]) if len(history) > 1 else set()
        marked = [f'{e} — this error was already fixed in an earlier round; '
                  'restore that fix while addressing the others'
                  if e in past and e not in history[-1] else e for e in errors]
        feedback = [_RouteIssue('segments', 'route_invalid', m + _DIFF_FIX)
                    for m in marked]
        history.append(errors)
    raise RouterError(f'router segment map invalid after repair: {errors}')


async def _recount(runner: AgentRunner, payload: str, by_code: dict,
                   zeros: list, chunks: list[str],
                   instructions: str = None) -> str:
    """Fresh-attention recount of the units a valid map left at zero.
    Deliberately NOT a repair round: no history, so the model never
    sees the map it is re-examining — anchoring on its own finished
    map is exactly the failure being corrected (a repair round shown
    the map re-emits it verbatim, measured). Returns the raw answer
    for `_splice` to fold in."""
    instructions = (instructions or _CHECK).format(
        # the card's header line carries the unit's own description — the
        # cue that separates a summarizing unit (answers its source) from
        # one with its own text (answers a count); units not being
        # recounted keep only their bare code — enough to name a
        # derivation source, too little to invite a re-answer
        legend='\n'.join(
            f'{c} = {by_code[c].header} RECOUNT' if c in zeros
            else f'{c} = {by_code[c].path}'
            for c, u in by_code.items() if u.kind == 'array'),
        chunks=_listing(chunks))
    return await fresh_check(
        runner, payload, instructions,
        'Declarations and map lines for the RECOUNT units only')


_SHARED_CHECK = '''A document's chunks are listed below. Some repeating units' instances
were left merged as one; recount them.

For each unit marked RECOUNT answer:
1. a count line "<code>: <n>" — how many instances the DOCUMENT holds;
2. then exactly n lines, one per instance in the card's order, each
   quoting VERBATIM the opening text of that instance as it stands in
   the chunk listing — enough text to locate its chunk, no commentary.

Units:
{units}

Chunks:

{chunks}
'''


async def _recount_shared(runner: AgentRunner, payload: str,
                          shared: dict, by_code: dict, chunks: list[str]):
    """Fresh-attention recount of the units left merged into shared
    runs. Every shared unit is recounted in the one conversation — the
    chunks listing, the costly part, is shared. The answer carries, per
    unit, the count and each instance's opening text: quotes anchor to
    chunks by content, the one coordinate the model quotes reliably —
    index arithmetic echoes the prompt's own examples instead of the
    chunks (measured: a stable count over hallucinated indexes).
    Returns ``{code: (count, quotes)}``; units with a malformed or
    missing section are absent."""
    instructions = _SHARED_CHECK.format(
        units='\n'.join(f'{code} = {by_code[code].header} RECOUNT'
                        for code in shared),
        chunks=_listing(chunks))
    result = await fresh_check(
        runner, payload, instructions,
        'Per recounted unit: a count line and the instances\' opening '
        'quotes')
    return _parse_shared_answer(result, shared)


def _parse_shared_answer(result, codes: dict | set) -> dict:
    """``{code: (count, quotes)}`` from a _SHARED_CHECK reply — a count
    line then quote lines, per unit; units with a malformed or missing
    section are absent, a zero count means the document truly lacks it."""
    sections = sectioned(
        result, lambda line: (m.group(1), int(m.group(2) or 0))
        if (m := _COUNT.match(line)) and m.group(1) in codes else None)
    return {code: (count, [q for q in quotes if q])
            for (code, count), quotes in sections.items() if count}


def _resplit(segments: list, counts: dict, code: str, answer,
             by_code: dict, derived: dict, chunks: list[str]):
    """Replace a merged unit's map lines with per-instance lines built
    from a fresh recount: each quoted opening anchors to its chunk
    (content seek, whitespace-normalized) and the unit's owned chunks
    partition at those anchors — quotes are in the card's order, chunks
    in document order, so anchors dedupe and sort rather than assume
    either direction; a quote whose chunk is already anchored reads as
    two instances sharing one chunk and fails the seek. Other units'
    destinations on the same chunks are preserved. Adoption is code's;
    any unusable piece returns None for the caller's fallback. Returns
    the rebuilt ``(segments, counts)``."""
    if not answer or not (count := answer[0]) or count != len(answer[1]):
        return None
    cover = _cover(segments)
    owned = sorted(c for c, dests in cover.items()
                   if code in {d for d, _ in dests})
    if not owned:
        return None
    norms = {c: norm(chunks[c]) for c in owned}
    hits = []
    for q in answer[1]:
        n = norm(q)
        hit = next((c for c in owned if c not in hits and n in norms[c]),
                   None) if n else None
        if hit is None:
            return None
        hits.append(hit)
    hits = sorted(hits)
    # before the first anchor the material rides instance 0
    item_of = {c: max(0, bisect_right(hits, c) - 1) for c in owned}
    per_chunk = {}
    for c, dests in cover.items():
        stripped = tuple((d, i) for d, i in dests if d != code)
        per_chunk[c] = (stripped + ((code, item_of[c]),)
                        if c in item_of else stripped)
    counts = {**counts, code: count}
    rebuilt = _resegment(per_chunk, counts, derived, by_code, len(chunks))
    if rebuilt is None:
        return None
    return rebuilt, counts


def _diff_text(reply, base_text) -> str | None:
    """A unified-diff reply applied to the model's own previous map as
    order-free line algebra: "-" lines drop every line they quote
    (content-anchored — the model quotes its own answer, so a quote
    that misses costs nothing), "+" lines add theirs (one the map
    already holds is a no-op — the model rewrites "-x/+x" pairs for
    lines it means to keep), everything else is commentary. Line order
    carries no meaning (ranges are explicit), so no positions are
    tracked: the result is exactly the model's listed edits — the map
    text the round leaves behind. None when the reply is no diff at
    all (a full re-emission replaces the base instead)."""
    removed, added = [], []
    for l in (s.strip() for s in str(reply).strip().splitlines()):
        if not l or l.startswith(('---', '+++')) or l[0] not in '+-':
            continue
        if content := l[1:].strip():
            (removed if l[0] == '-' else added).append(content)
    if not removed and not added:
        return None
    out = [l for l in (s.strip() for s in str(base_text).strip().splitlines())
           if l and l not in removed]
    kept = set(out)
    out += [a for a in dict.fromkeys(added) if a not in kept]
    return '\n'.join(out)


def _cover(segments: list) -> dict:
    """Per chunk id: the destinations owning it — the expanded form both
    adoption paths (recount splice, shared-run resplit) edit before
    re-segmenting."""
    return {c: dests for s, e, dests in segments for c in range(s, e + 1)}


def _resegment(per_chunk: dict, counts: dict, derived: dict,
               by_code: dict, n: int):
    """The adoption tail shared by both paths that edit the expanded
    map: a per-chunk destination map → coalesced segments (adjacent
    equal destinations merge into one run), validated against the
    declared counts — None when the result is not a valid map."""
    rebuilt = []
    for c in range(n):
        dests = per_chunk.get(c, ())
        if rebuilt and rebuilt[-1][2] == dests:
            rebuilt[-1] = (rebuilt[-1][0], c, dests)
        else:
            rebuilt.append((c, c, dests))
    if _map_errors(rebuilt, counts, derived, by_code, n):
        return None
    return rebuilt


def _splice(segments, counts, derived, answer, zeros, by_code: dict, n: int):
    """Fold a recount answer into a validated routing — the adoption is
    code's, not the model's: a repair round shown the map re-emits it
    verbatim (measured), so nothing is asked of the anchored
    conversation. Claims over chunks the map marked irrelevant splice
    in directly. A unit whose claims land on already-mapped material
    reads as a summarizing unit that answered a count — when exactly
    one other repeating unit shares that count, the derivation is
    adopted (its mirroring beats the model's line placement); anything
    else is unusable (caller falls back to a repair round). Returns the
    merged ``(segments, counts, derived)`` or None."""
    cover = {c: dests for s, e, dests in segments for c in range(s, e + 1)}
    claimed, crossed, seen = {}, set(), set()
    for line in str(answer).strip().splitlines():
        if m := _COUNT.match(line.strip()):
            code, declared, source = m.groups()
            if code not in zeros:
                continue  # not this recount's business
            if source is not None:
                src = by_code.get(source)
                if src is None or src.kind != 'array' or source in derived \
                        or source not in counts:
                    return None
                derived = {**derived, code: source}
            else:
                counts = {**counts, code: int(declared or 0)}
            continue
        if not (m := _LINE.match(line.strip())):
            continue
        start, end = int(m.group(1)), int(m.group(2) or m.group(1))
        dests = []
        for d in (t.strip() for t in m.group(3).split(',')):
            code, _, item = d.partition('.')
            if code == NONE:
                if dests:
                    return None  # NONE cannot ride a claiming line
                dests = None
                break
            if code not in zeros:
                continue  # a line this recount does not own
            if not item:
                return None  # itemless claim for a recounted unit
            index = int(item.split('.')[0])
            if (code, index) in seen:
                return None
            seen.add((code, index))
            dests.append((code, index))
        if not dests:
            continue  # noise about units this recount does not own
        if end >= n or end < start:
            return None
        for c in range(start, end + 1):
            if cover[c] == ((NONE, None),):
                claimed[c] = tuple(dests)
            else:
                crossed.update(d[0] for d in dests)
    for code in crossed - {d[0] for dests in claimed.values() for d in dests}:
        # a unique same-count repeating unit reads as the summarizing
        # source the model counted instead of naming
        candidates = [c for c, u in by_code.items()
                      if u.kind == 'array' and c != code and c not in derived
                      and counts.get(c) == counts.get(code) and counts[c]]
        if len(candidates) != 1:
            return None
        derived = {**derived, code: candidates[0]}
    rebuilt = _resegment({c: claimed.get(c) or cover[c] for c in range(n)},
                         counts, derived, by_code, n)
    if rebuilt is None:
        return None
    return rebuilt, counts, derived


def _content_len(chunk: str) -> int:
    """The chunk's non-blank characters — split() eats every Unicode
    blank (space, tab, newline, CJK full-width) in one C pass."""
    return sum(map(len, chunk.split()))


def _material(assignments, chunks, derived: dict) -> dict:
    """Per unit path, per item: the char count of the mapped material,
    whitespace stripped — a ratio budget scales against the content it
    will mirror, and aligned source text pads its columns with blanks
    (tables, line-broken layouts). Co-chunked items ("2-3 b.0,b.1")
    share one run but each mirrors its own slice: the shared material
    splits across the items. Derived units mirror their source's
    material — their own map lines (a contradictory but tolerated
    combo) are ignored."""
    base = [a for a in assignments if a['unit'] not in derived]
    shares, lens = {}, {}
    for a in base:
        by_chunk = shares.setdefault(a['unit'], {})
        for i in a['chunks']:
            by_chunk[i] = by_chunk.get(i, 0) + 1
            lens.setdefault(i, _content_len(chunks[i]))
    material = {}
    for a in base:
        by_item = material.setdefault(a['unit'], {})
        by_chunk = shares[a['unit']]
        by_item[a['item']] = by_item.get(a['item'], 0) + sum(
            lens[i] / by_chunk[i] for i in a['chunks'])
    for unit, source in derived.items():
        material[unit] = dict(material.get(source, {}))
    return material


def _resolved(budgets, material) -> dict:
    """Absolute per-item value-char estimates the executor compares
    against its batch capacity. A lone Budget expands to every item it
    covers; a list carries each item's own declaration."""
    out = {}
    for path, declared in budgets.items():
        if isinstance(declared, Budget):
            values = declared.per_item(material.get(path, {}))
        else:
            by_item = material.get(path, {})
            values = [b.chars(by_item.get(i, 0)) for i, b in enumerate(declared)]
        out[path] = values[0] if len(values) == 1 else values
    return out


def _codes(units: list) -> dict:
    """Bijective base-26 letter codes in unit order; NONE is reserved."""
    codes = {}
    for i, unit in enumerate(units):
        code, n = '', i
        while True:
            code = chr(ord('a') + n % 26) + code
            if n < 26:
                break
            n = n // 26 - 1
        codes[code] = unit
    return codes


def _parse(text, by_code: dict, n: int):
    """Validate count declarations plus a segment map. Returns (errors,
    counts, derived, segments, budgets); segments are ``(start, end,
    destinations)`` with destinations a tuple of ``(code, item|None|STAR)``
    (STAR marks a bare starred run, valid only until _expand_stars),
    derived a ``{code: source}`` map of summarizing units, budgets a
    ``{code: Budget}`` map of the (tolerated, never checked) ``@``
    declarations."""
    if not isinstance(text, str) or not text.strip():
        return (['output must be the count lines and segment map, nothing else'],
                {}, {}, [], {})
    errors, counts, derived, segments, budgets = [], {}, {}, [], {}
    seen_segments = set()
    in_map = False
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        if m := _COUNT.match(line):
            if not in_map:
                errors.append(f'line {i + 1}: count declarations must '
                              f'follow the map')
                continue
            code, declared, source = m.group(1), m.group(2), m.group(3)
            unit = by_code.get(code)
            if unit is None:
                errors.append(f'line {i + 1}: unknown unit code {code!r}')
            elif unit.kind != 'array':
                pass  # a count on a non-repeating unit is noise, ignored
            elif code in counts or code in derived:
                errors.append(f'line {i + 1}: {code} declared twice')
            elif source == code:
                errors.append(f'line {i + 1}: {code} cannot derive from itself')
            elif declared is None and source is None:
                errors.append(f'line {i + 1}: {code} must declare a count '
                              f'("{code}: <n>") or a source ("{code} = <code>")')
            elif source is not None:
                derived[code] = source  # a number beside "= <source>" is ignored
            else:
                counts[code] = int(declared)
            if unit is not None and (b := _BUDGET.search(line)):
                if values := [v for v in (_budget(t) for t in
                                          re.split(r'[,\s]+', b.group(1).strip()))
                              if v]:
                    budgets[code] = values[0] if len(values) == 1 else values
            continue
        in_map = True
        if not (m := _LINE.match(line)):
            hint = (' — NONE may not be comma-joined with other destinations'
                    if re.search(r',\s*-', line) else '')
            errors.append(f'line {i + 1}: {line!r} is not '
                          f'"<start>-<end> <code>[.<item>][,...]"{hint} — '
                          f'a bare "-" maps nothing, drop the line entirely')
            continue
        start, end = int(m.group(1)), int(m.group(2) or m.group(1))
        if start >= n or end >= n or end < start:
            errors.append(f'line {i + 1}: range {start}-{end} is unordered '
                          f'or outside 0..{n - 1}')
            continue
        destinations, seen = [], set()
        parts = m.group(3).split(',')
        for d in (t.strip() for t in parts):
            starred = d.endswith('*')
            code, _, item = d.removesuffix('*').partition('.')
            unit = by_code.get(code)
            if code == NONE and item:
                errors.append(f'line {i + 1}: {NONE} takes no item index')
            elif code == NONE:
                destinations.append((NONE, None))
            elif unit is None:
                errors.append(f'line {i + 1}: unknown unit code {code!r}')
            elif starred and len(parts) > 1:
                errors.append(f'line {i + 1}: {code}* takes its line alone — '
                              f'a star run never comma-joins another code')
            elif starred and item:
                errors.append(f'line {i + 1}: {code}* takes no item index — '
                              f'the run splits one instance per chunk by code')
            elif starred and unit.kind != 'array':
                errors.append(f'line {i + 1}: {code}* needs a repeating unit '
                              f'— only array units split per instance')
            elif starred:
                destinations.append((code, STAR))
            elif (code, item) in seen:
                errors.append(f'line {i + 1}: destination {d!r} appears twice')
            elif unit.kind == 'array' and item == '':
                seen.add((code, item))
                # bare repeating code: instances unsplit, extracted whole
                destinations.append((code, 0))
            else:
                seen.add((code, item))
                # item tolerated on non-array units, stripped; a dotted
                # tail ("2.0") keeps its leading index
                destinations.append((code, int(item.split('.')[0])
                                     if unit.kind == 'array' and item else None))
        segment = (start, end, tuple(destinations))
        if segment not in seen_segments:  # an exact re-emitted line is
            seen_segments.add(segment)    # redundancy, not an error
            segments.append(segment)
    segments.sort(key=lambda s: s[0])  # line order carries no meaning —
    # ranges are explicit; a diff reply's "+" insert lands at its own
    # editing position, so out-of-order lines are normal, not an error
    errors += _map_errors(segments, counts, derived, by_code, n)
    _, star_counts = _expand_stars(segments)  # starred units read their
    # count off the chunk total — published here, once; the rewrite
    # itself waits until after validation (the hint gate must see the
    # shared form), so only the totals carry over
    counts.update(star_counts)
    # exact duplicates collapse — one repeated destination must not
    # flood the repair feedback with the same message
    errors = list(dict.fromkeys(errors))
    return errors, counts, derived, [] if errors else segments, budgets


def _star_hints(segments: list, counts: dict, by_code: dict) -> list:
    """Advisory messages for array units left in shared form while a run
    covers exactly as many chunks as the unit has instances — probably
    an entry list that should be starred. Already-fanned maps (starred,
    or every instance on its own single chunk) and small units
    (STAR_HINT_MIN) are skipped."""
    covered, fanned, first = {}, {}, {}
    for start, end, dests in segments:
        if _is_star(dests):
            continue
        for c in {d for d, _ in dests if d != NONE}:
            mine = sum(1 for d, _ in dests if d == c)
            fanned[c] = fanned.get(c, True) and mine == 1 and start == end
            covered[c] = covered.get(c, 0) + end - start + 1
            first.setdefault(c, (start, end))
    hints = []
    for code, chunks in covered.items():
        unit = by_code.get(code)
        declared = counts.get(code)
        if unit is None or unit.kind != 'array' \
                or not declared or declared < STAR_HINT_MIN \
                or declared != chunks or fanned[code]:
            continue
        s, e = first[code]
        hints.append(
            f'{code}: this run covers {chunks} chunks for {declared} '
            f'instances. If every chunk holds exactly one instance, re-emit '
            f'the map with every {code} run starred ("{s}-{e} {code}*") — '
            f'it then extracts one instance per chunk in parallel. If the '
            f'instances share their chunks, reply with nothing.')
    return hints


def _split_hint(code: str, span: int, declared: int) -> str:
    """The shared-run hint's ask: separate one line per instance where
    chunk boundaries can, decline by staying silent otherwise."""
    return (f'{code}: {span} chunks carry {declared} instance(s). If the '
            f'chunk boundaries can separate the instances, re-emit with one '
            f'line per instance and a matching count ("5-9 {code}.0", '
            f'"10-11 {code}.1"). If the instances truly share their chunks, '
            f'reply with nothing.')


def _shared_hints(segments: list, counts: dict) -> dict:
    """Array units whose instances share one long run — comma-joined
    items, or a unit declared once whose lone item spans far more
    chunks than one entry plausibly reads alone. The mirror of the
    star hint: there the run holds one instance per chunk and should
    split per chunk; here several chunks per instance may still
    separate at chunk boundaries (measured lazy draw: a whole section
    mapped as one shared item). A genuinely huge single entry is
    indistinguishable from the merged form — the recount lets the
    model confirm it; a declined suggestion costs one round.
    Already-separated maps (each instance its own line) never fire;
    small excesses don't pay for the round. Returns ``{code: (chunk
    span, declared)}``."""
    shared = {}
    for start, end, dests in segments:
        if _is_star(dests) or start == end:
            continue
        per = {}
        for d, _ in dests:
            if d != NONE:
                per[d] = per.get(d, 0) + 1
        for d, n in per.items():
            if n > 1 or counts.get(d) == 1:
                shared[d] = shared.get(d, 0) + end - start + 1
    return {code: (span, declared)
            for code, span in shared.items()
            if (declared := counts.get(code) or 0)
            and span >= declared * 4
            and span - declared >= STAR_HINT_MIN}


def _is_star(dests: tuple):
    """The starred run's unit code, or None — a bare starred run holds
    one instance per chunk, split by code."""
    return dests[0][0] if len(dests) == 1 and dests[0][1] == STAR else None


def _expand_stars(segments: list) -> tuple:
    """Rewrite starred runs into per-chunk single-instance segments —
    a pure map→map transform, so assignments, groups, material, and
    budgets all see an ordinary per-item map. Returns ``(segments,
    totals)``: the rewritten map and each starred unit's chunk total,
    the code-read instance count. One walk is the single source of
    both the assigned indexes and the published counts. Presumes the
    chunker's entry-run guarantee (_pack_entries): every chunk of a
    starred run holds exactly one instance. Runs of one unit number on
    in chunk order."""
    out, totals = [], {}
    for start, end, dests in segments:
        if code := _is_star(dests):
            base = totals.get(code, 0)
            for i, c in enumerate(range(start, end + 1)):
                out.append((c, c, ((code, base + i),)))
            totals[code] = base + end - start + 1
        else:
            out.append((start, end, dests))
    return out, totals


def _map_errors(segments, counts, derived, by_code: dict, n: int) -> list:
    """Whole-map consistency: line-range overlap, declared-vs-used
    items per destination, derivation sanity, and full coverage
    (reported even alongside line errors, so the model gets the
    complete picture in one repair)."""
    errors, prev_end = [], -1
    for start, end, _ in segments:
        if start <= prev_end:
            errors.append(f'segment {start}-{end} overlaps after chunk '
                          f'{prev_end} — every chunk sits on exactly '
                          f'one line; units sharing a chunk ride '
                          f'that line together, e.g. "5 x.0,y.0"')
        prev_end = max(prev_end, end)
    for code, source in derived.items():
        src = by_code.get(source)
        if src is None:
            errors.append(f'{code}: = {source} is not a unit code')
        elif src.kind != 'array':
            errors.append(f'{code}: = {source} is not a repeating unit — '
                          f'derive from an array unit, the one whose items '
                          f'{code} mirrors')
        elif source in derived:
            errors.append(f'{code}: = {source} must reference a directly '
                          f'mapped repeating unit')
        elif source not in counts:
            errors.append(f'{code}: = {source} needs {source}\'s own count '
                          f'declared first')
    for code, unit in by_code.items():
        if unit.kind != 'array' or code in derived:
            continue
        items = {item for _, _, dests in segments for c, item in dests if c == code}
        if STAR in items:  # a starred unit: its chunks ARE the instances
            if items - {STAR}:
                errors.append(f'{code}: star and numbered destinations cannot '
                              f'mix — mark every run of {code} with *')
            continue
        if (declared := counts.get(code)) is None:
            if items:  # used but undeclared is a real inconsistency
                errors.append(f'{code}: items {sorted(items)} used but its '
                              f'count is undeclared — declare "{code}: <n>" '
                              f'first')
            else:
                # absent declarations read as zero — safe under map-first
                # because every zero is re-examined by the recount pass
                counts[code] = 0
            continue
        if items != set(range(declared)):
            hint = (f' — attach its items to the lines of the unit whose text it '
                    f'shares, e.g. "5 x.0,{code}.0"; declare 0 if the document '
                    f'holds none; or, when it merely summarizes another '
                    f'repeating unit, declare "{code} = <source>" instead of '
                    f'a count' if not items else
                    f' — give each instance its own line where chunk boundaries '
                    f'can separate them ("5-9 {code}.0", "10-11 {code}.1"); '
                    f'instances that share one chunk ride one line '
                    f'("5 {code}.0,{code}.1"), and an instance whose text '
                    f'sits inside another unit\'s run rides that line '
                    f'("5 x.0,{code}.1"); or the count is wrong — declare '
                    f'what the document holds')
            errors.append(f'{code}: declared {declared} items but the map '
                          f'uses {sorted(items)}{hint}')
    if missing := sorted(set(range(n)) - {
            c for s, e, *_ in segments for c in range(s, e + 1)}):
        errors.append(f'chunks not covered: {missing}')
    return errors


def _groups(segments: list, by_code: dict, chunks: list[str],
            derived: dict = None) -> list:
    """Validated line ranges are disjoint by construction; each
    destination gets its own group, chunk-adjacent same-destination
    pieces coalesce (a shared unit's items may interleave with the
    detailed unit's). Derived units mirror their source's groups."""
    groups, last = [], {}
    for start, end, destinations in segments:
        ids, text = list(range(start, end + 1)), '\n'.join(
            chunks[i] for i in range(start, end + 1))
        for code, item in destinations:
            if code == NONE:
                continue
            if g := last.get((code, item)):
                if g.chunk_ids[-1] == start - 1:  # adjacent piece, same destination
                    g.chunk_ids.extend(ids)
                    g.text += '\n' + text
                    continue
            g = Group(by_code[code], item, list(ids), text)
            last[(code, item)] = g
            groups.append(g)
    for code, source in (derived or {}).items():
        groups += [Group(by_code[code], g.item, list(g.chunk_ids), g.text)
                   for g in groups if g.unit is by_code[source]]
    return groups
