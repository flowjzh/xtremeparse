"""Router: one light agent call mapping chunk ids to extraction units.

Output is a compact segment DSL — the map first (``<start>-<end>
<code>[.<item>][,...]``, one line per contiguous run of chunks, units
compressed to letter codes, ``-`` for irrelevant chunks; a lifted
sub-array is addressed through its parent instance,
``<code>.<item>.<sub-code>.<sub-item>``), then a count
declaration per repeating unit after it (``<code>: <items>``,
optionally suffixed with a per-item output estimate — ``@<n>`` an
absolute cap, ``@<n>%`` a ratio of the item's mapped material, or
``@<n>x<m>`` an average keyword length times a keyword count — that
the executor treats as an arrangement), and per parent instance for a
lifted sub-array (``c.0.d: 2``). The map enumerates: the model
numbers a repeating unit's instances as it meets them, so the count is
derived from the map rather than committed up front — a count-first
declaration made the model report one fewer instance (counting is a
weak operation) and then obey its own wrong number, silently dropping
the section's tail entry. A run may feed several units at once
(comma-joined codes) — a summary or cross-cutting unit reading the
same text a detailed unit extracts; exclusive partitioning broke on
exactly that shape. A ranged run (``12-93 x.0-81``) is the compact
shared form: those chunks carry exactly instances 0..81,
inseparably — one line the executor keeps whole until a split round
redraws it. A valid map that leaves fan-out on the table — a shared
run whose mapped material overflows one shared call's capacity (the
dense draw), or instances sharing one run far longer than their
count — gets its fix in a round of its own: the overflow ask hands
the model the block lines to ratify — the even partition computed in
code, one line per call-sized block, the count sized from the unit's
own budget — and the merged lazy form gets a fresh-conversation
recount that code re-splits at the quoted openings, the one
conversation carrying every unit the map miscounted — merged and
zero-declared alike. A fan-out round
is a suggestion: silence declines it, and so does an answer that
fails validation — the standing map was valid before the ask
(the zero recount is the exception: its disagreement is resolved,
never declined).
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
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from xtremeparse.arbitration import fresh_check, norm, sectioned
from xtremeparse.contracts import AgentRunner, BATCH_BUDGET_CAP, JSONSchema
from xtremeparse.units import Unit, type_set, value_branches

NONE = '-'
HINT_MIN = 5  # below this a shared whole call costs seconds — the
# round would cost more than the split saves
RESHARE_LOAD = 8  # the recount's load bar, in chunks per instance, for
# every shared shape: a decomposed line (one destination per instance)
# and a declared-once lone item alike re-split only above this many
# chunks per instance — below it the recount is pure decode-shaving,
# confirmed and never adopted (a 2-instance band, a 6-chunk single —
# measured)
SPLIT_MATERIAL_CAP = 1000  # mapped content chars one shared call may
# absorb before the split ask pays for itself: past ~1k chars the
# whole-array decode (~58 tok/s measured) outlasts the extra round.
# Deliberately ~3x BATCH_BUDGET_CAP — a unit crossing this trigger at a
# full ratio splits into at least three blocks, so the ask is never a
# single-block no-op
RECOUNT_DESCRIPTION = 'Declarations and map lines for the RECOUNT units only'
SHARED_RECOUNT_DESCRIPTION = ('Per recounted unit: a count line and the '
                              "instances' opening quotes")
_BOTH_RECOUNT_DESCRIPTION = ('Per recounted unit: a count line, then its '
                             "instances' opening quotes or map lines")
# placeholders every prompt template must carry (host overrides are
# validated against these; see docs/prompting.md)
ROUTE_PLACEHOLDERS = frozenset({'top', 'none', 'legend', 'chunks'})
RECOUNT_PLACEHOLDERS = frozenset({'legend', 'chunks'})
_DEST = (r'[a-z]+(?:\.\d+)*(?:\.[a-z_]+(?:\.\d+)*)?'
         r'(?:-(?:[a-z]+\.)?\d+)?')
_LINE = re.compile(rf'^(\d+)(?:-(\d+))?\s+({_DEST}(?:\s*,\s*{_DEST})*'
                   r'|-(?:\.\d+)?)$')
_MISSING = re.compile(r'^chunks not covered: \[([\d, ]+)\]$')
_ITEM_RANGE = re.compile(r'(\d+)-(\d+)')
_DOUBLED = re.compile(r'(\d+)-[a-z]+\.(\d+)')
_SUBITEM = re.compile(r'(\d+)(?:\.\d+)*(?:-(\d+))?')
# one destination's post-code tail: parent item, then an optional chain
# through a lifted sub-array — its code or field-name spelling, then the
# sub-item ('0.d.0', '0.sub_experiences.0-2', 'd.0', '0.0', '')
_REST = re.compile(r'(\d+(?:-(?:[a-z]+\.)?\d+)?)?'
                   r'(?:\.([a-z_]+|\d+)'
                   r'(?:\.(\d+(?:-(?:[a-z]+\.)?\d+)?))?)?')
_CHAIN = re.compile(r'([a-z]+)\.(\d+)\.([a-z_]+)(?:\.(\d+))?')
_COUNT = re.compile(r'^([a-z]+(?:\.\d+\.[a-z_]+)?(?:\.\d+)?)'
                    r'(?::\s*(\d+))?(?:\s*=\s*([a-z]+))?(?:\s*@.*)?$')
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
code, ".item" appended on repeating units; "4-9 x.0-8" is the ranged
form — those chunks carry exactly instances 0-8, in chunk order; a
bare repeating code ("4-9 x") means several instances share the run
unsplit — they are extracted whole; a single chunk may omit "-end").
Example codes below
are schematic letters, not units of this document. After the map, one
count line per repeating unit ("x: 3" plus its budget suffix)
declaring the instances the DOCUMENT holds — every repeating unit gets
one: 0 only for a unit truly absent, and a unit with no text of its
own that only summarizes another declares its source ("x = y")
instead.

{overall}Rules:
- As you write the map, number a repeating unit's instances in the
  order you meet them across the WHOLE document — a new item index per
  instance, early and late instances alike; the count line after the
  map reports the highest index plus one, 0 for a unit the document
  does not contain at all. A unit with no
  text of its own that only summarizes another unit declares just
  its source and takes NO map lines:
  "x = y @30%" — its items (and its count) mirror the source's
  material, but its budget is its own: set it for the lines the
  summary will emit — a ratio scales against the source material it
  summarizes, so a terse summary sits well below 100%.
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
- A unit whose legend line shows another unit's path ("<code> =
  [parent.field | array]") nests inside that unit's instances. Its
  sub-entries never ride a parent's plain line — each gets its own
  chain line THROUGH the parent instance
  ("<parent>.<item>.<code>.<sub>", e.g. "3-15 c.0.d.0" — parent
  instance 0's first sub-entry), one line per sub-entry where chunk
  boundaries allow. A parent line plus a count line maps no
  sub-entries: only chain lines do. Every chain line also feeds its
  parent instance the chunks it covers, so those chunks take no
  separate parent line:
      12-14 c.0
      15-20 c.0.d.0
      21-26 c.0.d.1
      c.0.d: 2
  After the map, each parent instance holding sub-entries declares
  their count ("c.0.d: 2", budget suffix as for any unit); a parent
  holding none declares nothing. That count line is the lifted
  sub-array's ONLY declaration — never a top-level count ("d: 2"),
  never a source ("d = c.0.d"), and a sub-entry takes no count line
  of its own.
- Items of the SAME unit that separable chunk boundaries CAN separate
  MUST each get their own line ("5-9 x.0" then "10-11 x.1"): each
  item's budget then scales against its own material. Sharing one line
  ("3 x.0,x.1,x.2") is ONLY for a
  run whose chunk boundaries cannot separate the instances — one chunk
  holding material of two or more instances; that material is then
  extracted once, whole.
- A long run whose chunks carry one unit's instances IN ORDER — an
  entry list too dense for chunk boundaries to separate (roughly one
  instance to a chunk) — MUST be ONE ranged line ("12-93 x.0-84": name
  the exact index span the chunks carry). NEVER enumerate that many
  indexes comma-joined ("12 x.0,x.1,x.2,..." is a broken draw). One
  unit, one shape: when chunk boundaries CAN separate the instances,
  each takes its own line and no ranged line may cover them — drawing
  the ranged form and per-instance lines for the same instances
  overlaps and is invalid. Short runs (a few instances) stay per-item
  lines. Length alone never makes a ranged line — density does; and a
  parent whose sub-entries take chain lines takes plain per-item
  lines for its own chunks, never a ranged parent line beside its
  chain lines (the chains feed the parent those chunks).
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
  (its count "x: <n>" or its source "x = y") and every parent instance
  holding a nested unit's sub-entries declares its count ("x.0.y: <n>"),
  the map covers 0..{top}
  exactly once in ascending non-overlapping lines, item indexes of
  each unit run 0..used-1 with no gaps, every count line matches
  the map.

Chunks:

{chunks}

Unit codes (legend lines are definitions for reference — the map and
count lines name codes only):
{legend}'''


_DIFF_HOWTO = '''
Reply with a unified diff against your previous map: "-" lines removed,
"+" lines added, one edit per line — never re-emit an unchanged line.
The full map is also accepted; a partial fragment is not — a reply
without "-" or "+" lines replaces the whole map, so every line left
out is lost.'''
_DIFF_REPLY = _DIFF_HOWTO + ' An empty reply declines the suggestion.'
_DIFF_FIX = _DIFF_HOWTO + ' Fix every error named above.'
_DECLINE = 'reply with nothing.'  # the silent decline every fan-out ask
# ends with — an empty reply keeps the standing map


_CHECK = '''You are rechecking part of a routing decision. The repeating units
listed at the end were left at zero; recount their instances in the
DOCUMENT — read the whole chunk list; a brief mention inside an
otherwise irrelevant run is still an instance.

{overall}For each unit answer one line only, <code> being that unit's code
from the list at the end:
- its count, "<code>: <n>", when the unit has its own text in the
  chunks — then add its map lines in the routing DSL
  ("4 <code>.0", "5 <code>.1"), one item per instance, covering
  exactly its material;
- "<code>: 0" alone when the document truly does not contain the unit.

Chunks:

{chunks}

Units left at zero (code = unit card):
{legend}'''


class RouterError(Exception):
    """Raised when the router's segment map is still invalid after repairs."""


@dataclass
class Group:
    """One routed slice: a unit, an optional array item, and the
    deterministic concatenation of its chunks. Groups of one unit (and of
    one item) may coalesce into a single specialist call in the executor.
    ``items`` marks a ranged run: these chunks carry instances
    items[0]..items[1] inseparably — the executor makes the run one
    batched call instead of fanning its instances apart. ``parent`` is
    the parent instance a lifted sub-array's group extracts under
    (None for top-level units)."""

    unit: Unit
    item: Optional[int]
    chunk_ids: list
    text: str
    items: tuple = ()  # (first, last) instance indexes of a ranged run
    parent: Optional[int] = None


def items_of(dest):
    """The item indexes one destination claims: a ranged run (a, b)
    claims a..b, anything else is itself (None or an index).
    The one spelling of that rule — parser validation, assignment
    expansion, executor batching and material pricing all come here."""
    return range(dest[0], dest[1] + 1) if isinstance(dest, tuple) else (dest,)


def _claim_range(a, b, key, seen) -> list:
    """A ranged claim's unclaimed indexes — ``key(k)`` their seen-mark,
    the idempotent re-claim fold the plain and chain parser paths
    share: a repeated destination claims nothing new. The expansion
    half is ``items_of``'s."""
    out = []
    for k in items_of((a, b)):
        mark = key(k)
        if mark not in seen:
            seen.add(mark)
            out.append(k)
    return out


def _undouble(item: str) -> str:
    """One destination's item token with a doubled-code range folded:
    '0-d.2' is how the model naturally spells '0-2' (the right end
    repeating the code). One tolerance, one spelling, shared by every
    reader of the language."""
    if dup := _DOUBLED.fullmatch(item):
        return f'{dup[1]}-{dup[2]}'
    return item


def _flatten(item: str) -> str:
    """One destination's item token with sub-numbering flattened: the
    schema's instances are flat, but the model sub-numbers entries the
    document nests (one employer, two roles) — 'c.0.1' claims instance
    0, its parent. Rejection sent a canary repair loop circling to
    exhaustion: the model never abandons the spelling. The parser's
    chain reading takes the schema-aware cases first; this fold stays
    for the recount splice, where no schema shape can disambiguate."""
    if m := _SUBITEM.fullmatch(item):
        return '-'.join(p for p in m.groups() if p)
    return item


def _fold(item: str, unit: Unit):
    """The flat reading of a dotted item token on ``unit``'s line: the
    leading index keeps the claim, further tails drop — the schema's
    instances are flat. None when the token claims nothing (itemless,
    or a non-array unit whose item is stripped)."""
    if unit.kind != 'array' or not item or not item[0].isdigit():
        return None
    return int(item.split('.')[0])


def _nested_unit(mid: str, unit: Unit, by_code: dict):
    """The code of the lifted sub-array a chain's middle token names
    under ``unit`` — by code or by field-name spelling; None when the
    token names no sub-array of this unit."""
    for c, u in by_code.items():
        if u.parent == unit.path and (c == mid or u.field == mid):
            return c
    return None


def _parent_code(unit: Unit, by_code: dict) -> str:
    """The code of the array unit a lifted sub-array hangs under."""
    return next((c for c, u in by_code.items() if u.path == unit.parent),
                unit.parent)


def _rides(code: str, unit: Unit, by_code: dict) -> str:
    """The named error for a nested unit addressed without its parent —
    the one spelling of the chain reminder."""
    return (f'{code} rides its parent — write '
            f'{_parent_code(unit, by_code)}.<item>.{code}.<sub-item>')


def _assignments(segments: list, by_code: dict) -> list:
    """The per-instance assignment rows one map denotes: a dict per
    (destination, item index) carrying its chunks — the shape the trace
    publishes and ``_material`` prices. A chain destination also names
    the parent instance it hangs under."""
    rows = []
    for start, end, destinations in segments:
        for dest in destinations:
            if dest[0] == NONE:
                continue
            for i in items_of(dest[1]):
                row = {'unit': by_code[dest[0]].path, 'item': i,
                       'chunks': list(range(start, end + 1))}
                if dest[2] is not None:
                    row['parent'] = dest[2]
                rows.append(row)
    return rows


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


def _overall_block(overall: str) -> str:
    """The block an overall instruction renders as, or '' — the templates
    glue `{overall}` onto the following line and the block carries its
    own trailing newlines, so the prompts read the same either way."""
    return f'Overall Instruction:\n\n{overall}\n\n' if overall else ''


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
                recount_instructions: str = None,
                overall: str = None) -> Routing:
    """Map chunks to units via one agent call, with bounded repairs —
    each a diff against the previous answer — feeding the validation
    errors back as ``feedback``; plus one
    verification round when any repeating unit comes out declared 0
    (see route loop). A replayed map — the same text twice — passes
    its declared-vs-used mismatches through: the extraction's count
    arbitration owns the number. ``payload`` is the precomputed shared
    prefix (see prompting). ``instructions``/``recount_instructions``
    replace the default prompt templates; they must carry their
    placeholders (ROUTE_PLACEHOLDERS / RECOUNT_PLACEHOLDERS — the
    Extractor validates host overrides at construction). ``overall`` is
    the schema's document-level instruction, rendered as the "Overall
    Instruction" block in the default prompts — '' when absent."""
    by_code = _codes(units)
    overall_block = _overall_block(overall)
    instructions = (instructions or _INSTRUCTIONS).format(
        top=len(chunks) - 1, none=NONE, overall=overall_block,
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

    def adopt(text):
        """Commit a code-authored map text: parse, take the state it
        names, leave the text behind for diff_base to anchor on — the
        one adoption that must survive a later model diff. Returns the
        parse errors; the caller owns the fallback."""
        nonlocal counts, nested, derived, segments, budgets, diff_base, valid
        errors, cts, nest, deriv, segs, budg = _parse(text, by_code,
                                                      len(chunks))
        if not errors:
            counts, nested, derived, segments, budgets = \
                cts, nest, deriv, segs, budg
            diff_base = text
            valid = (segs, cts, nest, deriv, budg, text)
        return errors

    async def split_ask(spans):
        # a shared run holding more mapped material than one shared
        # call may absorb — the dense draw (span ≈ declared). The even
        # partition is computed in code, because the model's own
        # boundary arithmetic across a long run's junk headings is what
        # loses rounds (measured: 2 of 5 canary trials drew an
        # overlapping block and fell into per-instance enumeration) —
        # and adopted in code too: the ask round transcribed the block
        # lines verbatim every time, a round spent re-typing what code
        # already wrote. The round survives only for runs code may not
        # redraw — a line shared with another unit or carrying a chain
        rounds = []
        for code, (first, last, lines) in _overrun_hints(
                segments, spans, counts, chunks, by_code, budgets,
                derived).items():
            spliced = _adopt_blocks(diff_base, code, first, last, lines)
            if spliced is None or (spliced != diff_base
                                   and adopt(spliced)):
                # None: another unit's destination or a chain rides the
                # run's lines; errors: unverifiable — either way the
                # ask round owns it. == diff_base is a refired pass
                # after a full adoption
                rounds.append(_RouteIssue('segments', 'route_hint',
                                          _overrun_hint(code, first, last,
                                                        lines)))
        if not rounds:
            return None
        asked.add('split')  # every ask marks itself when it spends a round
        return True, rounds

    def splice(zeros, answer):
        """`_splice`'s fold — commit the recount's state on success.
        The caller owns the fallback."""
        nonlocal segments, counts, nested, derived
        if merged := _splice(segments, counts, nested, derived, answer,
                             zeros, by_code, len(chunks)):
            segments, counts, nested, derived = merged
            return True
        return False

    def resplit(shared, parsed):
        """Re-split each shared run at its recounted openings; returns
        the codes whose sections were unusable."""
        nonlocal segments, counts
        pending = []
        for code in shared:
            if merged := _resplit(segments, counts, nested, code,
                                  parsed.get(code),
                                  by_code, derived, chunks):
                segments, counts = merged
            else:
                pending.append(code)
        return pending

    async def shared_ask(shared):
        nonlocal segments, counts, nested
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
        # simply stays shared). Fired alone only when no zero
        # pends beside it — shared_and_zeros owns that round, and
        # the caller passes a non-empty hint dict either way
        answer = await _recount_shared(runner, payload, {c: c for c in shared},
                                       by_code, chunks, overall_block=overall_block)
        pending = resplit(shared, _parse_shared_answer(answer, shared))
        if len(pending) == len(shared):
            return [_RouteIssue(
                'segments', 'route_hint',
                _split_hint(code, *shared[code]) + _DIFF_REPLY)
                for code in pending]
        return None  # a full or partial adoption stands

    async def zeros_ask(zeros):
        # a zero is never trusted on the map's own say-so: the
        # model commits to its finished map and will not revisit
        # NONE'd material — not in the same pass, and not in a
        # feedback round that shows it that map (measured: brief
        # sections and summary units rubber-stamped as 0). A
        # fresh conversation without the map re-counts them, and
        # code splices the answer in — the anchored conversation
        # would re-emit its own map verbatim. Its disagreement is
        # resolved, never declined: a botched fix keeps the repair
        # loop, because the alternative is trusting the suspect
        # zero. Fired alone only when no shared run pends beside
        # it — shared_and_zeros owns that round, and the caller
        # passes a non-empty zero list either way
        answer = await _recount(runner, payload, by_code, zeros,
                                chunks,
                                instructions=recount_instructions,
                                overall_block=overall_block)
        if splice(zeros, answer):
            return None  # the recount's adoption stands
        return [_RouteIssue('segments', 'route_invalid',
                            zeros_note(zeros) + _DIFF_REPLY)]

    def zeros_note(zeros):
        note = (f'{", ".join(zeros)}: a separate recount of '
                'the document disagreed with this map but '
                'could not be merged — for each, recheck the '
                'chunk list yourself: give every instance '
                'its map lines with a matching count, or '
                'declare "= <source>" if it only summarizes '
                'another repeating unit; keep 0 only if '
                'truly absent')
        history.append([note])
        return note

    async def shared_and_zeros(shared, zeros):
        # both kinds pend: one fresh conversation recounts them
        # together — the chunks listing is the costly part, and
        # the two recount prompts differ only in framing. Zeros
        # fold first, and nothing is adopted before the splice
        # lands: its disagreement is a resolution, and the diff
        # round a failed splice forces re-parses the model's
        # answer text over any split adopted here. Once the
        # splice stands there is no diff round at all — a
        # pending unit simply stays shared, the same rule the
        # shared-only ask applies after a partial adoption
        answer = await _recount_both(runner, payload, {c: c for c in shared},
                                     zeros, by_code, chunks,
                                     overall_block=overall_block)
        if not splice(zeros, answer):
            return False, [_RouteIssue('segments', 'route_invalid',
                                       zeros_note(zeros) + _DIFF_REPLY)]
        parsed = _parse_shared_answer(answer, list(shared) + zeros)
        # the zero units' count lines delimit sections too — a count
        # line outside the shared codes would otherwise ride the
        # preceding unit's quote list and fail its count check
        resplit(shared, parsed)
        return None  # a full or partial adoption stands

    async def fanout_ask(shared_spans):
        # both fan-out kinds in one round when they pend together;
        # a host's recount_instructions is tuned for the zero
        # case, so an override keeps the two asks apart. The asks
        # mark themselves here — a combined round spends both
        # halves at once, and a shared round's feedback must
        # leave zeros pending for the next visit
        shared = _shared_hints(shared_spans, counts)
        zeros = _pending_zeros(by_code, counts, derived)
        if shared and zeros and 'shared' not in asked \
                and 'zeros' not in asked and not recount_instructions:
            asked.update({'shared', 'zeros'})
            return await shared_and_zeros(shared, zeros)
        if shared and 'shared' not in asked:
            asked.add('shared')
            if fb := await shared_ask(shared):
                return True, fb
        if zeros and 'zeros' not in asked:
            asked.add('zeros')
            if fb := await zeros_ask(zeros):
                return False, fb
        return None

    async def hint_pass():
        """The hint passes in firing order — split first (its
        adoption shifts the spans the fan-out round reads), then
        the fan-out round. Each ask marks itself in ``asked`` when
        it spends a round and returns its own classification —
        (declinable, feedback) or None — because a combined round's
        grade is whichever disposition its halves force; the split
        ask is a suggestion, always. Returns None when no hint
        applied or one adopted its fix in code and the map may be
        final."""
        spans, merged = _shared_spans(segments, counts)
        if 'split' not in asked \
                and (fb := await split_ask(spans)) is not None:
            return fb
        return await fanout_ask(merged)

    def finalize():
        assignments = _assignments(segments, by_code)
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
        nested_counts = {}
        for (cc, p), k in nested.items():
            nested_counts.setdefault(by_code[cc].path, {})[p] = k
        return Routing(_groups(segments, by_code, chunks, derived),
                       {'counts': {by_code[c].path: k for c, k in counts.items()}
                                  | {by_code[d].path: counts[s]
                                     for d, s in derived.items()},
                        'nested_counts': nested_counts,
                        'assignments': assignments,
                        'budgets': {p: _budget_text(b)
                                    for p, b in budgets_by_path.items()},
                        'maps': maps}, by_path)

    feedback, last, history = None, None, []
    asked = set()  # hint passes already spent — each ask fires at most
    # once per routing, however the map shifts afterwards
    maps = []  # every round's applied map text — chunk ids, codes and
    # item indexes only, no document content; the eval trace keeps them
    diff_base = None  # the previous round's map text — armed once below,
    # after every answer, so a future hint round cannot forget it
    valid = None  # the last valid round: parsed state plus the map text,
    # because the diff base must agree with what code holds
    suggestion = None  # whether the outstanding feedback is a
    # suggestion round (declinable) or a resolution (repairable)
    # error lists: a repaired error that later reappears means the model
    # is rewriting fixed lines away — name it
    stuck_text = None  # the previous repair round's applied map text —
    # two identical rounds in a row are a replay, not a repair
    lenient = set()  # unit codes whose declared-vs-used mismatch passed
    # through after a replay (the count arbitration owns the number)
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
            elif (patched := _overlay_text(text, base)) is not None:
                text = patched  # a marker-less reply patches, not replaces
        errors, counts, nested, derived, segments, budgets = \
            _parse(text, by_code, len(chunks))
        # every answer leaves a map text behind — a full re-emission is
        # itself, a diff leaves its applied map, an empty reply the
        # standing base — and the next round diffs against it, hint or
        # repair alike
        diff_base = text
        maps.append(text)
        if errors and not suggestion:
            replayed = text == stuck_text
            if replayed:
                # the model replayed its map verbatim (measured: a count
                # it cannot localize comes back as a no-op diff until
                # the budget burns — every line re-quoted as -x/+x).
                # Re-asking cannot move it; downgrade the declared-vs-
                # used mismatches to a pass-through and let the
                # extraction's count arbitration own the number — the
                # router owns chunk allocation, and a 200 with a
                # settled count beats a 500
                lenient |= _passable(errors)
            stuck_text = text
            if lenient:
                errors, counts, nested, derived, segments, budgets = \
                    _parse(text, by_code, len(chunks), lenient)
        if errors and not suggestion and 'undrawn' not in asked \
                and all(_PASSABLE.match(e) for e in errors):
            # declared-undrawn family: a count declared, no lines drawn.
            # Repair rounds fail it — the anchored conversation re-emits
            # its own map, and the chain/covering arithmetic its fixes
            # need fumbles shared seams (measured: the 81-item b-[] 1-3
            # rounds, the block-layout d-[] half the draws, the
            # same-company multi-stint d-[] burns on the under-merged
            # run). The fresh recount is the zeros' discipline applied
            # to the drawing: count each unit plus its instances'
            # opening text, and code anchors the quotes and lays the
            # lines itself — the declaration is right (the first pass
            # reads the document; only the drawing is skipped), so the
            # recount usually just confirms it. Fires only when every
            # error is count-family — a map with geometry flaws stays
            # on the diff path, and the standing parse's state serves
            # as is (its coherence: _parse's return)
            passable = _passable(errors)
            if pool := _undrawn_pool(segments, counts, nested):
                asked.add('undrawn')
                zeros = ([] if recount_instructions else
                         _pending_zeros(by_code, counts, derived))
                answer = await _recount_undrawn(runner, payload, pool,
                                                zeros, by_code, chunks,
                                                overall_block=overall_block)
                fixed = _fold_undrawn(segments, counts, nested, derived,
                                      pool, zeros, answer, by_code,
                                      len(chunks), chunks, lenient | passable)
                if fixed is not None:
                    (segments, counts, nested, derived), zeros_ok = fixed
                    text = _map_text(segments, counts, nested, derived,
                                     budgets, by_code)
                    diff_base = text
                    maps.append(text)
                    stuck_text = None  # adoption resets the replay ledger
                    replayed = False  # the standing map is code's now
                    if zeros_ok:
                        asked.add('zeros')  # the merged ask covered zeros
                    errors = _map_errors(segments, counts, nested,
                                         derived, by_code, len(chunks))
        if not errors:
            valid = (segments, counts, nested, derived, budgets, text)
        elif not suggestion:
            past = set().union(*history[:-1]) if len(history) > 1 else set()
            removed = []
            if any(_MISSING.match(e) for e in errors):
                edits = _diff_edits(result.data)
                if edits:
                    removed = _net_edits(*edits)[0]
            marked = []
            for raw in errors:
                e = raw
                if m := _MISSING.match(e):
                    missing = [int(x) for x in m[1].split(',')]
                    blame = _uncovered_removal(removed, missing)
                    e += (f' — your removal of "{blame}" uncovered these; '
                          f're-add it unless the chunks belong elsewhere'
                          if blame else
                          _coverage_hint(segments, missing, by_code))
                if raw in past and raw not in history[-1]:
                    e += ' — this error was already fixed in an earlier ' \
                         'round; restore that fix while addressing the others'
                marked.append(e)
            if replayed:
                # the reply was a no-op rewrite: say so — a verbatim
                # "-x/+x" pair cancels out, and the full map is the
                # escape hatch when quoting keeps missing
                marked = [f'{m} — and your previous reply changed '
                          'nothing: re-quoting a line as a "-x/+x" '
                          'pair cancels out; quote only the lines you '
                          'are changing, or re-emit the corrected map '
                          'in full'
                          for m in marked]
            feedback = [_RouteIssue('segments', 'route_invalid', m + _DIFF_FIX)
                        for m in marked]
            history.append(errors)
            continue
        else:
            # a fan-out round whose reply failed validation declines:
            # the standing map was valid before the ask, so a botched
            # redraw is worth less than it — restore it, the diff base
            # with it (a later hint's diff must anchor on the map code
            # actually holds), spend no model round on the wreckage,
            # and let the remaining asks run
            segments, counts, nested, derived, budgets, diff_base = valid
        if hint_fb := await hint_pass():
            suggestion, feedback = hint_fb
            continue
        return finalize()
    # exhausted: the model had its bounded repairs, and the flaws that
    # remain are the ones it could not localize (measured: oscillation
    # between two wrong redraws, or junk). Settle, on the replay
    # downgrade's own trade — a settled 200 beats a 500: the last
    # valid round stands, and count-family mismatches on the final
    # text pass through to the extraction's count arbitration. A
    # geometry flaw (overlap, coverage) has no downstream arbiter —
    # passing it would double-claim chunks silently — so that stays a
    # RouterError, and the caller's fresh draw is the cure
    if valid is not None:
        segments, counts, nested, derived, budgets, _ = valid
        return finalize()
    lenient |= _passable(errors)
    if lenient:
        errors, counts, nested, derived, segments, budgets = \
            _parse(diff_base, by_code, len(chunks), lenient)
        if not errors:
            return finalize()
    raise RouterError(f'router segment map invalid after repair: {errors}')


async def _recount(runner: AgentRunner, payload: str, by_code: dict,
                   zeros: list, chunks: list[str],
                   instructions: str = None, overall_block: str = '') -> str:
    """Fresh-attention recount of the units a valid map left at zero.
    Deliberately NOT a repair round: no history, so the model never
    sees the map it is re-examining — anchoring on its own finished
    map is exactly the failure being corrected (a repair round shown
    the map re-emits it verbatim, measured). Returns the raw answer
    for `_splice` to fold in."""
    instructions = (instructions or _CHECK).format(
        # only the recounted units appear — the other codes' mere
        # presence invited a full re-answer of every array (their
        # "final" counts re-derived from the chunks, measured: 137
        # output tokens for a one-line answer). A summarizing unit's
        # derivation needs no source code in the legend: _splice's
        # crossed-count adoption reads it off the misplaced claims
        legend=_recount_legend({c: c for c in zeros}, by_code),
        chunks=_listing(chunks), overall=overall_block)
    return await fresh_check(
        runner, payload, instructions, RECOUNT_DESCRIPTION)


_SHARED_CHECK = '''A document's chunks are listed below. Some repeating units' instances
were left merged as one; recount them.

{overall}For each unit marked RECOUNT answer:
1. a count line "<code>: <n>" — how many instances the DOCUMENT holds;
2. then exactly n lines, one per instance, each
   quoting VERBATIM the opening text of that instance as it stands in
   the chunk listing — enough text to locate its chunk, no commentary.

Units:
{units}

Chunks:

{chunks}
'''


async def _recount_shared(runner: AgentRunner, payload: str,
                          shared: dict, by_code: dict, chunks: list[str],
                          overall_block: str = '') -> str:
    """Fresh-attention recount of the units left merged into shared
    runs. Every shared unit is recounted in the one conversation — the
    chunks listing, the costly part, is shared. The answer carries, per
    unit, the count and each instance's opening text: quotes anchor to
    chunks by content, the one coordinate the model quotes reliably —
    index arithmetic echoes the prompt's own examples instead of the
    chunks (measured: a stable count over hallucinated indexes).
    Returns the raw answer; `_parse_shared_answer` reads the sections,
    its keys the labels the legend printed (plain codes here — the
    declared-undrawn ask passes chain spellings through this same
    prompt)."""
    instructions = _SHARED_CHECK.format(
        units=_recount_legend(shared, by_code),
        chunks=_listing(chunks), overall=overall_block)
    return await fresh_check(
        runner, payload, instructions, SHARED_RECOUNT_DESCRIPTION)


def _parse_shared_answer(result, codes: dict | set) -> dict:
    """``{code: (count, quotes)}`` from a _SHARED_CHECK reply — a count
    line then quote lines, per unit; units with a malformed or missing
    section are absent, a zero count means the document truly lacks it."""
    sections = sectioned(
        result, lambda line: (m.group(1), int(m.group(2) or 0))
        if (m := _COUNT.match(line)) and m.group(1) in codes else None)
    return {code: (count, [q for q in quotes if q])
            for (code, count), quotes in sections.items() if count}


# The quote rule below is _SHARED_CHECK's, the zero-unit rules are
# _CHECK's — reword one firing mode's copy and the others must follow,
# or the adoption rates drift by path with no test signal.
_BOTH_CHECK = '''A document's chunks are listed below. Some repeating units were
left miscounted: the MERGED ones had their instances folded into one
shared run, the ZERO ones were written off as absent. Recount them in
the DOCUMENT — read the whole chunk list; a brief mention inside an
otherwise irrelevant run is still an instance.

{overall}For each unit marked RECOUNT answer a count line "<code>: <n>" — how
many instances the DOCUMENT holds — then, per its tag:

- a MERGED unit: exactly n lines, one per instance, each quoting
  VERBATIM the opening text of that instance as
  it stands in the chunk listing — enough text to locate its chunk,
  no commentary;
- a ZERO unit with its own text: its map lines in the routing DSL
  ("4 <code>.0", "5 <code>.1"), one item per instance, covering
  exactly its material;
- a ZERO unit the document truly does not contain: the count line
  "<code>: 0" alone.

Units:
{units}

Chunks:

{chunks}
'''


async def _recount_both(runner: AgentRunner, payload: str, shared: dict,
                        zeros: list, by_code: dict, chunks: list[str],
                        overall_block: str = '') -> str:
    """Fresh-attention recount of the merged and the zero units in the
    one conversation — the chunks listing, the costly part, is shared.
    Each unit answers a count line; a merged unit's instances quote
    their openings, a zero unit claims map lines. Returns the raw
    answer: `_splice` reads the zero units' lines, `_parse_shared_
    answer` the merged units' sections — each skips the other's
    codes."""
    instructions = _BOTH_CHECK.format(
        units=_recount_legend({**shared, **{z: z for z in zeros}}, by_code,
                              lambda c: ' MERGED' if c in shared else ' ZERO'),
        chunks=_listing(chunks), overall=overall_block)
    return await fresh_check(runner, payload, instructions,
                             _BOTH_RECOUNT_DESCRIPTION)


def _recount_legend(units, by_code, tag=None) -> str:
    """The Units listing every recount prompt carries — one line per
    recounted unit, ``<label> = <unit header> RECOUNT``. ``units`` maps
    the label to the code (a plain code maps to itself; the declared-
    undrawn ask passes chain spellings as labels); ``tag`` names the
    unit's firing mode (``' MERGED'`` / ``' ZERO'``) when the one
    conversation mixes them."""
    return '\n'.join(
        f'{label} = {by_code[code].header} RECOUNT'
        f'{tag(label) if tag else ""}' for label, code in units.items())


def _pending_zeros(by_code, counts, derived) -> list:
    """Array units a map left at zero. Nested units never appear in
    ``counts`` — their numbers live in ``nested`` — so the flat
    enumeration already excludes them; derived units mirror a source."""
    return [c for c, u in by_code.items()
            if u.kind == 'array' and c not in derived
            and counts.get(c) == 0]


def _undrawn_pool(segments, counts, nested) -> list:
    """The declared-undrawn units of a parsed map — counts declared but
    no lines drawn: ``[(code, None)]`` top-level, ``[(sub-code, parent
    index)]`` chained — the same state `_map_errors` reads when it
    writes the assigns-none errors, read from the source, not the
    prose."""
    used = set()
    for _, _, dests in segments:
        for dd in dests:
            used.add((dd[0], None) if dd[2] is None else (dd[0], dd[2]))
    return [(code, None) for code, declared in counts.items()
            if declared and (code, None) not in used] + \
        [(code, parent) for (code, parent), declared in nested.items()
         if declared and (code, parent) not in used]


def _chain_label(code: str, parent: int | None, by_code: dict) -> str:
    """The chain spelling of a nested unit — the plain code, or
    ``<parent>.<item>.<code>`` naming the parent instance it hangs
    under. The one source of the label grammar: the undrawn recount's
    legend and the rebuilt map's tokens both come here."""
    if parent is None:
        return code
    return f'{_parent_code(by_code[code], by_code)}.{parent}.{code}'


def _line_of(segments, code: str, item: int, parent: int | None):
    """The first line claiming one nested item under one parent
    instance — (start, end), or None. The over-draw remedy's fold
    endpoints come from here."""
    for s, e, dests in segments:
        for dd in dests:
            if dd[0] == code and dd[2] == parent and item in items_of(dd[1]):
                return s, e
    return None


def _runs(nums) -> list:
    """Consecutive ints folded to inclusive (a, b) runs — the grouping
    half of ``items_of``'s expansion."""
    runs = []
    for c in nums:
        if runs and c == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], c)
        else:
            runs.append((c, c))
    return runs


def _dest_token(code: str, item, parent, by_code: dict) -> str:
    """One destination's map token — the bare code, ``code.item``,
    the ranged ``code.a-b``, or the chain spelling. The one renderer
    of the token grammar: rebuilt maps and repair hints both come
    here, so what the parser reads and what the feedback names stay
    the same language."""
    if item is None:
        return code
    if parent is not None:
        return f'{_chain_label(code, parent, by_code)}.{item}'
    if isinstance(item, tuple):
        return f'{code}.{item[0]}-{item[1]}'
    return f'{code}.{item}'


def _dest_tokens(dests, by_code: dict) -> str:
    """A line's right-hand side, comma-joined. A parent whose own
    chains ride the line is dropped — the chain token re-adds its
    parent at parse, so an explicit parent beside them is redundancy.
    The one spelling a map line's destinations take: the rebuilt map
    and the repair hints both come here, so a removal the hint quotes
    matches the line the base text holds."""
    chained = {(_parent_code(by_code[scode], by_code), p)
               for scode, _, p in dests if p is not None}
    return ','.join(_dest_token(code, item, parent, by_code)
                    for code, item, parent in dests
                    if (code, item) not in chained)


async def _recount_undrawn(runner: AgentRunner, payload: str,
                           units: list, zeros: list, by_code: dict,
                           chunks: list[str],
                           overall_block: str = '') -> str:
    """Fresh-attention recount of the units a map declared but never
    drew — the merged recount's prompt verbatim (count plus each
    instance's opening text), the units labeled by their chain
    spellings. Zero-declared units recount in the same conversation
    when any pend (the chunks listing is the costly part). Returns the
    raw answer: `_fold_undrawn` owns the adoption."""
    shared = {_chain_label(code, parent, by_code): code
              for code, parent in units}
    if zeros:
        return await _recount_both(runner, payload, shared, zeros,
                                   by_code, chunks,
                                   overall_block=overall_block)
    return await _recount_shared(runner, payload, shared, by_code, chunks,
                                 overall_block=overall_block)


def _anchors(quotes: list, domain: list, chunks: list[str]) -> list | None:
    """Each quote's chunk in ``domain`` — content seek over the
    domain's normalized texts, normed once per chunk, not once per
    quote. A quote that misses (or lands where another already
    anchored: two instances sharing one chunk) fails the whole set —
    the shared-run recount's own rule, the one coordinate the model
    quotes reliably. None on any miss; the caller sorts before
    partitioning (quotes may not arrive in document order)."""
    norms = {c: norm(chunks[c]) for c in domain}
    hits = []
    for q in quotes:
        n = norm(q)
        hit = next((c for c in domain if c not in hits and n in norms[c]),
                   None) if n else None
        if hit is None:
            return None
        hits.append(hit)
    return hits


def _resplit(segments: list, counts: dict, nested: dict, code: str, answer,
             by_code: dict, derived: dict, chunks: list[str]):
    """Replace a merged unit's map lines with per-instance lines built
    from a fresh recount: each quoted opening anchors to its chunk
    (content seek, whitespace-normalized) and the unit's owned chunks
    partition at those anchors — quotes may not arrive in document
    order, so anchors sort rather than assume a direction; a quote
    whose chunk is already anchored reads as
    two instances sharing one chunk and fails the seek. Other units'
    destinations on the same chunks are preserved. Adoption is code's;
    any unusable piece returns None for the caller's fallback. Returns
    the rebuilt ``(segments, counts)``."""
    if not answer or not (count := answer[0]) or count != len(answer[1]):
        return None
    cover = _cover(segments)
    owned = sorted(c for c, dests in cover.items()
                   if code in {dd[0] for dd in dests})
    if not owned or (hits := _anchors(answer[1], owned, chunks)) is None:
        return None
    hits = sorted(hits)
    # before the first anchor the material rides instance 0
    item_of = {c: max(0, bisect_right(hits, c) - 1) for c in owned}
    per_chunk = {}
    for c, dests in cover.items():
        stripped = tuple(dd for dd in dests if dd[0] != code)
        per_chunk[c] = (stripped + ((code, item_of[c], None),)
                        if c in item_of else stripped)
    counts = {**counts, code: count}
    rebuilt = _resegment(per_chunk, counts, nested, derived, by_code,
                         len(chunks))
    if rebuilt is None:
        return None
    return rebuilt, counts


def _lines(text) -> list:
    """The stripped, non-blank lines every applier and the parser must
    agree on."""
    return [l.strip() for l in str(text).strip().splitlines() if l.strip()]


def _dests(dests: str) -> list:
    """A _LINE destination list's comma-split tokens."""
    return [t.strip() for t in dests.split(',')]


def _diff_edits(reply) -> tuple | None:
    """A diff reply's removals and additions — None when the reply is
    no diff at all. The one parse of the marker grammar: the applier
    and the feedback's removal blame both read it, so a removal the
    hint names is a removal the grammar agrees was asked. Once a real
    marker line shows the reply is a diff, a bare line that speaks the
    map's grammar rides along as an addition — the contract is never
    to re-emit an unchanged line, so a bare line is a lazy "+": asked
    to attach items, the model drew the line with no prefix, and
    dropped as commentary it silently zeroed the unit while the
    removal beside it landed (measured: the b-[] repair burned to
    exhaustion on the phantom)."""
    removed, added, saw_marker = [], [], False
    for l in _lines(reply):
        if l.startswith(('---', '+++')):
            continue
        if l[0] in '+-':
            saw_marker = True
            if content := l[1:].strip():
                (removed if l[0] == '-' else added).append(content)
        elif _LINE.match(l) or _COUNT.match(l):
            added.append(l)
    if not saw_marker or not removed and not added:
        return None
    return removed, added


def _diff_text(reply, base_text) -> str | None:
    """A unified-diff reply applied to the model's own previous map.
    None when the reply is no diff at all (a full re-emission replaces
    the base instead). A bare line the map already holds dedupes to a
    no-op; one it cannot hold surfaces as the overlap error it is."""
    edits = _diff_edits(reply)
    if edits is None:
        return None
    return _apply_edits(base_text, *edits)


def _net_edits(removed: list, added: list) -> tuple:
    """A "-x/+x" pair verbatim cancels: a line removed and re-added
    would only re-append (reorder), and the applier owes every caller
    the byte-stable base the replay downgrade keys on. The removals
    that survive are what the base actually lost — the removal blame
    in the feedback reads the same list."""
    common = Counter(removed) & Counter(added)
    return (list((Counter(removed) - common).elements()),
            list((Counter(added) - common).elements()))


def _apply_edits(base_text, removed: list, added: list) -> str:
    """Order-free line algebra, the apply step every applier shares:
    removed lines drop by content (content-anchored — the model quotes
    its own answer, so a quote that misses costs nothing), added lines
    append unless the map already holds them (a no-op — the model
    rewrites "-x/+x" pairs for lines it means to keep). Line order
    carries no meaning (ranges are explicit), so no positions are
    tracked: the result is exactly the listed edits — the map text the
    round leaves behind. A reply that rewrites every map line as
    "-x/+x" pairs leaves the count declarations as the only surviving
    base lines — appending the additions behind them built a
    counts-first text the parser then blamed the model for. The
    grammar's order is the applier's invariant: map lines first,
    everything after."""
    added = list(added)  # an applier may pass a dict.fromkeys dedupe map
    removed, added = _net_edits(removed, added)
    if not removed and not added:
        return base_text  # every edit cancelled — the base stands
    out = [l for l in _lines(base_text) if l not in removed]
    kept = set(out)
    out += [a for a in dict.fromkeys(added) if a not in kept]
    out.sort(key=lambda l: 0 if _LINE.match(l) else 1)
    return '\n'.join(out)


def _overlay_text(reply, base_text) -> str | None:
    """A marker-less reply of chain lines alone, read as the patch it
    means — the one marker-less fragment the batteries measure: asked
    to add the chain lines, the model answers with the chain lines
    alone, and full-replacement dropped the map's head, spent the next
    round re-typing lines that were never wrong, and left a legal map
    to boot. The chains patch in (dedup included), the count lines
    replace their code's declaration, and every other base line
    stands. Any covering line keeps the replacement reading —
    omission-to-delete is the other marker-less intent, and one
    reading cannot serve both. Chain-ness stays textual and
    over-approximate on purpose: two or more dots in a destination
    token (``b.0.d.0``, the ranged ``b.0.sub.0-2``, the bare-numeric
    ``b.0.1`` whether chain or flat fold) — the parser's
    re-validation owns the borderline forms. None when there is
    nothing to patch."""
    chains, rcount = [], []
    for l in _lines(reply):
        if _LINE.match(l):
            if any(t.count('.') >= 2 for t in _dests(_LINE.match(l)[3])):
                chains.append(l)
            else:
                return None  # a covering line: the map is replaced
        elif _COUNT.match(l):
            rcount.append(l)
    if not chains:
        return None
    rcodes = {_COUNT.match(l)[1] for l in rcount}
    removed = [bl for bl in _lines(base_text)
               if (bm := _COUNT.match(bl)) and bm[1] in rcodes]
    return _apply_edits(base_text, removed, dict.fromkeys(chains + rcount))


def _pure_chain(dests: tuple) -> bool:
    """A segment of nothing but chain destinations — the settled form of
    a chain line nested in its parent's run."""
    return bool(dests) and all(dd[2] is not None for dd in dests)


def _cover(segments: list) -> dict:
    """Per chunk id: the destinations owning it — the expanded form both
    adoption paths (recount splice, shared-run resplit) edit before
    re-segmenting. A pure-chain segment nesting in a parent run keeps
    the parent's destinations underneath it: containment is segment
    geometry, which the per-chunk form cannot express, and dropping the
    parent would leave the chains riding no run — the rebuild would
    reject the very map it started from, and a zeros recount that
    merely CONFIRMED its zeros would burn a repair round on nothing
    (measured: every confirm on a chain map did). The round-trip is
    not shape-stable — a hosted parent run comes back as the chains'
    own slices with the parent riding each — but coverage is identical
    and the group layer re-coalesces adjacency."""
    plain, chains = {}, {}
    for s, e, dests in segments:
        into = chains if _pure_chain(dests) else plain
        for c in range(s, e + 1):
            into[c] = dests
    return {c: plain.get(c, ()) + chains.get(c, ())
            for c in chains.keys() | plain.keys()}


def _resegment(per_chunk: dict, counts: dict, nested: dict, derived: dict,
               by_code: dict, n: int, lenient: set | None = None):
    """The adoption tail shared by both paths that edit the expanded
    map: a per-chunk destination map → coalesced segments (adjacent
    equal destinations merge into one run; the result is a disjoint
    partition, so containment needs no re-settling — ``_cover`` already
    carried the parents beneath the chains), validated against the
    declared counts — None when the result is not a valid map. ``lenient``
    passes count-family mismatches through (the undrawn fold validates
    its claims while unrelated declared-vs-used errors still stand)."""
    rebuilt = []
    for c in range(n):
        dests = per_chunk.get(c, ())
        if rebuilt and rebuilt[-1][2] == dests:
            rebuilt[-1] = (rebuilt[-1][0], c, dests)
        else:
            rebuilt.append((c, c, dests))
    if _map_errors(rebuilt, counts, nested, derived, by_code, n, lenient):
        return None
    return rebuilt


def _splice(segments, counts, nested, derived, answer, zeros, by_code: dict,
            n: int, lenient: set | None = None):
    """Fold a recount answer into a validated routing — the adoption is
    code's, not the model's: a repair round shown the map re-emits it
    verbatim (measured), so nothing is asked of the anchored
    conversation. Claims over chunks the map marked irrelevant splice
    in directly. A unit whose claims land on already-mapped material
    reads as a summarizing unit that answered a count — when exactly
    one other repeating unit shares that count, the derivation is
    adopted (its mirroring beats the model's line placement); anything
    else is unusable (caller falls back to a repair round). Returns the
    merged ``(segments, counts, nested, derived)`` or None. ``lenient``
    passes count-family mismatches through the rebuild's validation —
    the declared-undrawn fold splices zeros while its own units' errors
    still stand."""
    cover = _cover(segments)
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
        for d in _dests(m.group(3)):
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
            item = _undouble(item)
            item = _flatten(item)
            if m3 := _ITEM_RANGE.fullmatch(item):
                a, b = int(m3[1]), int(m3[2])
                if b < a:
                    return None  # unordered — the same rejection _parse names
                # a ranged claim is the compact shared form's own
                # syntax: the claimed chunks carry instances a..b,
                # one destination per index
                index = items_of((a, b))
            else:
                index = [int(item)]
            for i in index:
                if (code, i) in seen:
                    return None
                seen.add((code, i))
                dests.append((code, i, None))
        if not dests:
            continue  # noise about units this recount does not own
        if end >= n or end < start:
            return None
        for c in range(start, end + 1):
            if cover[c] == ((NONE, None, None),):
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
                         counts, nested, derived, by_code, n, lenient)
    if rebuilt is None:
        return None
    return rebuilt, counts, nested, derived


def _budget_text(b):
    """One parsed budget's list-or-scalar spelling — the one shared by
    the trace's budget rendering and the state-to-text rebuild."""
    return [str(v) for v in b] if isinstance(b, list) else str(b)


def _budget_suffix(b) -> str:
    """A parsed budget's round-trip suffix (``@...``) for the rebuilt
    map text — the state-to-text rebuild must not lose the executor's
    estimates."""
    if b is None:
        return ''
    text = _budget_text(b)
    return '@' + ','.join(text) if isinstance(text, list) else f'@{text}'


def _quote_anchor(quote: str, domain: list, norms: dict) -> int | None:
    """The first chunk of ``domain`` holding the quote — ``norms`` the
    domain's normalized texts, precomputed once (one norm per chunk,
    not one per quote). The recount quotes the chunk listing verbatim,
    so content seek is the one reliable coordinate (index arithmetic
    echoes the prompt's own examples, measured)."""
    n = norm(quote)
    if not n:
        return None
    return next((c for c in domain if n in norms[c]), None)


def _map_text(segments, counts, nested, derived, budgets, by_code) -> str:
    """A routing rebuilt in code, rendered back to map text — an
    adoption that continues the round loop must leave a text the
    anchored conversation can keep diffing against. Every count line
    round-trips: declared and zero-filled counts alike, per-parent
    nested counts, derivations, and budget suffixes (a non-repeating
    unit's estimate rides the noise count line the parser ignores for
    the count but reads for the budget)."""
    lines = []
    for start, end, dests in segments:
        if dests == ((NONE, None, None),):
            lines.append(f'{start} -' if start == end else f'{start}-{end} -')
            continue
        tokens = _dest_tokens(dests, by_code)
        lines.append(f'{start}-{end} {tokens}'
                     if start != end else f'{start} {tokens}')
    for code, unit in by_code.items():
        suffix = _budget_suffix(budgets.get(code))
        if code in derived:
            lines.append(f'{code} = {derived[code]}{suffix}')
        elif unit.kind == 'array' and unit.parent is None:
            lines.append(f'{code}: {counts[code]}{suffix}')
        elif suffix and unit.kind != 'array':
            lines.append(f'{code}: 1{suffix}')
    for (code, parent), k in sorted(nested.items()):
        lines.append(f'{_chain_label(code, parent, by_code)}: {k}'
                     f'{_budget_suffix(budgets.get(code))}')
    return '\n'.join(lines)


def _fold_undrawn(segments, counts, nested, derived, pool, zeros, answer,
                  by_code, n, chunks, lenient):
    """Fold the declared-undrawn recount into a validated map — the
    adoption is code's, the anchored conversation never redraws its
    own lines again. Each undrawn unit's answer — count plus per-
    instance opening quotes — anchors to chunks: a top-level unit's
    instances claim the chunks holding their openings (several items
    may share one chunk when the material is inseparable); a chained
    unit's sub-entries partition the parent's run from the first to
    the last anchor, each keeping the chunks up to the next opening,
    the run's head and tail staying the parent's own. A top-level
    recount that merges its instances' openings into one quote — the
    dense list one chunk holds inseparably — claims that chunk for all
    of them. Claimed chunks keep every other unit's lines — a chunk
    the map wrote off as irrelevant yields its NONE claim to the
    recount's material. Counts update to the recount's number, the
    fresh read authoritative under the zeros' own rule.
    A unit whose section is unusable — missing, quotes unanchored,
    count and quotes disagreeing — is skipped and keeps its error for
    the diff path. Returns ``(rebuilt state, zero half spliced)``, or
    None when nothing was adoptable."""
    labels = {_chain_label(code, parent, by_code) for code, parent in pool}
    parsed = _parse_shared_answer(answer, labels | set(zeros))
    changed, zeros_ok = False, False
    if zeros:
        if spliced := _splice(segments, counts, nested, derived, answer,
                              zeros, by_code, n, lenient):
            segments, counts, nested, derived = spliced
            changed = zeros_ok = True
    per_chunk = {c: list(ds) for c, ds in _cover(segments).items()}
    norms = None
    for code, parent in pool:
        section = parsed.get(_chain_label(code, parent, by_code))
        if not section:
            continue  # absent or zero — the map keeps its error, diff repairs
        count, quotes = section
        if parent is None:
            if norms is None:
                norms = {c: norm(chunks[c]) for c in range(n)}
            hits = [_quote_anchor(q, range(n), norms) for q in quotes]
            if any(c is None for c in hits):
                continue
            if len(hits) == 1 and count != len(hits):
                # the recount merged the instances' openings into one
                # quote — the dense list one chunk holds inseparably
                # (measured: 2 of 3 answer the b-[] ask this way);
                # every item claims the chunk
                hits = [hits[0]] * count
            if len(hits) != count:
                continue  # under-quoted across chunks — ambiguous, diff
            for i, c in enumerate(sorted(hits)):
                ds = per_chunk[c]
                if ds == [(NONE, None, None)]:
                    ds.clear()  # material the map wrote off as irrelevant
                ds.append((code, i, None))
            counts = {**counts, code: count}
        else:
            if count != len(quotes):
                continue  # unusable — the map keeps its error, diff repairs
            pcode = _parent_code(by_code[code], by_code)
            owned = sorted(c for c, ds in per_chunk.items()
                           if (pcode, parent, None) in ds)
            if (hits := _anchors(quotes, owned, chunks)) is None:
                continue
            anchors = sorted(hits)  # quotes may not arrive in document order
            for c in owned:
                if anchors[0] <= c <= anchors[-1]:
                    per_chunk[c].append((code, bisect_right(anchors, c) - 1,
                                         parent))
            nested = {**nested, (code, parent): count}
        changed = True
    if not changed:
        return None
    rebuilt = _resegment({c: tuple(ds) for c, ds in per_chunk.items()},
                         counts, nested, derived, by_code, n, lenient)
    if rebuilt is None:
        return None
    return (rebuilt, counts, nested, derived), zeros_ok


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
    splits across the items. Rows sharing an identical chunk list (a
    ranged run's instances) are one run priced once, its material split
    evenly — not one full-chunk-list dict per instance. Derived units
    mirror their source's material — their own map lines (a
    contradictory but tolerated combo) are ignored."""
    base = [a for a in assignments if a['unit'] not in derived]
    runs, lens = {}, {}  # (unit path, chunk tuple) → [item indexes]
    for a in base:
        runs.setdefault((a['unit'], tuple(a['chunks'])), []).append(a['item'])
        for c in a['chunks']:
            lens.setdefault(c, _content_len(chunks[c]))
    claims = {}  # (unit path, chunk) → rows claiming it, across runs —
    # a chunk two differently-bounded instances touch splits between them
    for (unit, cs), items in runs.items():
        for c in cs:
            claims[unit, c] = claims.get((unit, c), 0) + len(items)
    material = {}
    for (unit, cs), items in runs.items():
        amount = sum(lens[c] / claims[unit, c] for c in cs)
        by_item = material.setdefault(unit, {})
        for item in items:
            by_item[item] = by_item.get(item, 0) + amount
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


def _declared_budget(line: str):
    """The ``@`` suffix's parsed entries, or None — tolerated, never
    checked; the one read shared by the top-level and chain count
    branches."""
    if b := _BUDGET.search(line):
        values = [v for v in (_budget(t) for t in
                              re.split(r'[,\s]+', b.group(1).strip())) if v]
        if values:
            return values[0] if len(values) == 1 else values
    return None


def _parse(text, by_code: dict, n: int, lenient: set | None = None):
    """Validate count declarations plus a segment map. Returns (errors,
    counts, nested, derived, segments, budgets); segments are ``(start,
    end, destinations)`` with destinations a uniform ``(code,
    item|None, parent|None)`` — the third element names the parent
    instance of a chain through a lifted sub-array — derived a
    ``{code: source}`` map
    of summarizing units, counts a ``{code: int}`` map of top-level
    declarations, nested a ``{(sub-code, parent index): int}`` map of
    the per-parent ones, budgets a ``{code: Budget}`` map of the
    (tolerated, never checked) ``@`` declarations. ``lenient`` names
    codes whose declared-vs-used mismatch is passed through unsaid —
    a replayed map's count arbitration owns the number."""
    if not isinstance(text, str) or not text.strip():
        return (['output must be the count lines and segment map, nothing else'],
                {}, {}, {}, [], {})
    errors, counts, nested, derived, segments, budgets = [], {}, {}, {}, [], {}
    sub_decls = {}
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
            if chain := _CHAIN.fullmatch(code):
                punit, sunit = by_code.get(chain[1]), by_code.get(chain[3])
                if sunit is None or punit is None \
                        or sunit.parent != punit.path:
                    errors.append(f'line {i + 1}: {code!r} is not a nested '
                                  f'unit chain — {chain[3]!r} must be the '
                                  f'unit the legend nests under {chain[1]}')
                elif chain[4] is not None:
                    # a per-sub-entry declaration — the model's natural
                    # mirror of item counting; tolerated, and folded: the
                    # highest declared sub-entry numbers the parent's
                    # count when no explicit one stands (a budget suffix
                    # on it is ignored — pricing is the parent
                    # declaration's business)
                    if source is not None:
                        errors.append(f'line {i + 1}: {code} takes a '
                                      f'count, no source')
                    else:
                        sub_decls.setdefault((chain[3], int(chain[2])), {})[
                            int(chain[4])] = int(declared or 0)
                elif (chain[3], int(chain[2])) in nested:
                    errors.append(f'line {i + 1}: {code} declared twice')
                elif source is not None:
                    errors.append(f'line {i + 1}: {code} counts per parent — '
                                  f'declare a count ("{code}: <n>"), no '
                                  f'source')
                elif declared is None:
                    errors.append(f'line {i + 1}: {code} must declare a '
                                  f'count ("{code}: <n>")')
                else:
                    nested[(chain[3], int(chain[2]))] = int(declared)
                if chain[4] is None and sunit is not None and punit is not None \
                        and (b := _declared_budget(line)):
                    # the sub-array's arrangement is unit-level: one
                    # entry covers its sub-items under every parent
                    budgets[chain[3]] = b
                continue
            unit = by_code.get(code)
            if unit is None:
                errors.append(f'line {i + 1}: unknown unit code {code!r}')
            elif unit.parent is not None and declared == '0':
                pass  # a nested unit's bare zero says what silence says —
                # noise, ignored (the prompt's declaration example pulls
                # models into writing it; measured on live draws)
            elif unit.parent is not None:
                errors.append(f'line {i + 1}: {code} counts per parent — '
                              f'declare "{_parent_code(unit, by_code)}.'
                              f'<item>.{code}: <n>", no count of its own')
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
            if unit is not None and (b := _declared_budget(line)):
                budgets[code] = b
            continue
        in_map = True
        if not (m := _LINE.match(line)):
            hint = (' — NONE may not be comma-joined with other destinations'
                    if re.search(r',\s*-', line) else
                    ' — write the item range as "5-9 d.0-2": the second '
                    'index never repeats the code'
                    if re.search(r'[a-z]+\.\d+\s*-\s*[a-z]+\.\d+', line) else
                    ' — write "c.0": the item is the code\'s first index, '
                    'further dotted tails are dropped'
                    if re.search(r'[a-z]+\.\d+\.\d+', line) else '')
            tail = (' — a bare "-" maps nothing, drop the line entirely'
                    if re.search(r'(?:^|[\s,])-(?:$|[\s,])', line) else '')
            errors.append(f'line {i + 1}: {line!r} is not '
                          f'"<start>-<end> <code>[.<item>][,...]"{hint}{tail}')
            continue
        start, end = int(m.group(1)), int(m.group(2) or m.group(1))
        if start >= n or end >= n or end < start:
            errors.append(f'line {i + 1}: range {start}-{end} is unordered '
                          f'or outside 0..{n - 1}')
            continue
        destinations, seen = [], set()
        tokens = _dests(m.group(3))
        for d in tokens:
            code, _, rest = d.partition('.')
            unit = by_code.get(code)
            if code == NONE and rest:
                errors.append(f'line {i + 1}: {NONE} takes no item index')
            elif code == NONE:
                destinations.append((NONE, None, None))
            elif unit is None:
                errors.append(f'line {i + 1}: unknown unit code {code!r}')
            elif unit.parent is not None:
                # a lifted sub-array has no address of its own: its items
                # are per-parent, only the chain reaches them — spelled
                # directly or folded from a comma-joined parent
                out, err = _attach_chain(destinations, d, code, rest,
                                         unit, by_code)
                if err:
                    errors.append(f'line {i + 1}: {err}')
            else:
                out, err = _destination(d, code, rest, unit, by_code, seen,
                                        len(tokens) > 1)
                if err:
                    errors.append(f'line {i + 1}: {err}')
                else:
                    destinations += out
        if (seg := _merge_line(segments, start, end,
                               tuple(destinations))) is not None:
            segments.append(seg)
    for (scode, pidx), subs in sub_decls.items():
        # the fold: per-sub-entry declarations stand in for the parent's
        # count when the model didn't write one — map-first, the highest
        # numbered sub-entry is what the document holds
        if (scode, pidx) not in nested:
            nested[(scode, pidx)] = max(subs) + 1
    segments.sort(key=lambda s: s[0])  # line order carries no meaning —
    # ranges are explicit; a diff reply's "+" insert lands at its own
    # editing position, so out-of-order lines are normal, not an error
    _settle_chains(segments, by_code)
    errors += _map_errors(segments, counts, nested, derived, by_code, n,
                          lenient)
    # exact duplicates collapse — one repeated destination must not
    # flood the repair feedback with the same message
    errors = list(dict.fromkeys(errors))
    # segments return alongside errors: the callers gate on the error
    # list, and the declared-undrawn path reads the standing state
    # when every error is count-family (its segments are coherent —
    # syntax and geometry errors are never passable)
    return errors, counts, nested, derived, segments, budgets


def _chain_parents(dests: tuple, by_code: dict) -> set:
    """The parent destinations a line's chain destinations hang under —
    {(parent-code, parent-instance)}; empty when the line has none."""
    return {(_parent_code(by_code[dd[0]], by_code), dd[2])
            for dd in dests if dd[2] is not None}


def _hosted_chain(start: int, end: int, parents: set, candidates) -> bool:
    """The one home of the chain-hosting rule: whether some candidate
    segment spans [start, end] and carries every parent destination in
    ``parents`` among its plain destinations."""
    return any(s <= start and end <= e
               and parents <= {(x[0], x[1]) for x in ds if x[2] is None}
               for s, e, ds in candidates)


def _settle_chains(segments: list, by_code: dict) -> None:
    """A chain-borne line — nothing but chains and their implicit parent
    destinations — that nests in a run of the parent's own line drops
    the implicit parents: the fine partition stands, each sub-entry
    keeping its slice as its own segment (the executor fans them out
    separately; a kept implicit parent would re-extract the run the
    host line already covers). One no run contains stays as it is —
    covering, the parent's coverage there, exactly what the prompt
    promises."""

    def implicit_only(ds):
        chains = [dd for dd in ds if dd[2] is not None]
        parents = _chain_parents(ds, by_code)
        return bool(chains) and all(
            dd[2] is not None or (dd[0], dd[1]) in parents for dd in ds)

    bases = [(s, e, ds) for s, e, ds in segments if not implicit_only(ds)]
    for k, (s, e, ds) in enumerate(segments):
        if not implicit_only(ds):
            continue
        if _hosted_chain(s, e, _chain_parents(ds, by_code), bases):
            segments[k] = (s, e, tuple(dd for dd in ds
                                       if dd[2] is not None))


def _merge_line(segments: list, start: int, end: int, dests: tuple):
    """One parsed line folded into the map — None when an existing line
    absorbed it. Same-range repeats union their destinations (an exact
    re-emission is plain redundancy): the model lines a parent's run
    and its chains up as separate same-range lines, and one chunk set
    is one line's semantics — rejecting that sent the repair loop
    oscillating to exhaustion (measured). Lines at DIFFERENT ranges
    stay apart: a chain line nesting in its parent's run is legal on
    its own (the fine partition fans the sub-entries out separately,
    merging it would collapse them into one whole call), anything else
    overlapping is an overlap error."""

    for k, (s, e, ds) in enumerate(segments):
        if (s, e) == (start, end):
            if ds != dests:
                theirs = () if ds == ((NONE, None, None),) else ds
                mine = () if dests == ((NONE, None, None),) else dests
                theirs_set = set(theirs)  # linear membership, not |mine|·|theirs|
                segments[k] = (s, e, theirs + tuple(
                    dd for dd in mine if dd not in theirs_set))
            return None
    return start, end, dests


def _destination(d, code, rest, unit, by_code, seen, multi):
    """One destination token parsed against the schema — ``(destinations,
    error)``. Top-level destinations are ``(code, item)``; a chain
    through a lifted sub-array adds ``(sub-code, sub-item, parent)`` to
    the same line, the parent instance riding with its sub-entry. Two
    spellings normalize to the chain — the field name
    ('c.0.sub_experiences.0') and, when the unit holds exactly one
    sub-array, the bare numeric tail ('c.0.0': the schema's shape says
    what the tail numbers). Anything else dotted folds to the leading
    index, the flat reading. ``seen`` accumulates this line's
    destinations for duplicate detection."""
    if not (m := _REST.fullmatch(rest)):
        return [(code, _fold(rest, unit), None)], None
    item = _undouble(m.group(1)) if m.group(1) else m.group(1)
    mid, sub = m.group(2), m.group(3)
    if mid is None:
        return _plain(d, code, item, unit, seen, multi)
    dotted = f'{item}.{mid}' + (f'.{sub}' if sub else '')
    if mid.isdigit():
        # the bare numeric tail: a sub-entry of the unit's one lifted
        # sub-array when the schema shapes it so, else the flat fold
        children = [c for c, u in by_code.items() if u.parent == unit.path]
        if (sub is None and unit.kind == 'array' and item and item.isdigit()
                and len(children) == 1):
            parent_dest = [] if (code, item) in seen \
                else [(code, int(item), None)]
            if (children[0], mid, item) in seen:
                return parent_dest, None  # re-claimed: idempotent
            seen.add((children[0], mid, item))
            seen.add((code, item))
            return [*parent_dest,
                    (children[0], int(mid), int(item))], None
        return [(code, _fold(dotted, unit), None)], None
    sub_code = _nested_unit(mid, unit, by_code)
    if sub_code is None:
        if any(u.parent is not None and (c == mid or u.field == mid)
               for c, u in by_code.items()):
            return [], (f'{mid} does not nest under {code} — chain through '
                        f'the unit it really hangs under')
        return [(code, _fold(dotted, unit), None)], None
    if item is None:
        return [], _rides(sub_code, by_code[sub_code], by_code)
    if not item.isdigit():
        return [], (f'{code}.{item}.{sub_code}: the chain numbers one '
                    f'parent instance — no range on the parent index')
    if sub is None:
        return [], (f'{code}.{item}.{sub_code} needs its sub-item — write '
                    f'{code}.{item}.{sub_code}.<sub-item>')
    parent = int(item)
    if m2 := _ITEM_RANGE.fullmatch(_undouble(sub)):
        a, b = int(m2.group(1)), int(m2.group(2))
        if b < a:
            return [], f'{d!r} is unordered'
        # a second chain of the same parent on one line rides the
        # parent destination already claimed — the co-chunked shared
        # form ('c.0.d.0,c.0.d.1') is exactly that spelling; a
        # re-claimed destination is idempotent, never an error
        ks = _claim_range(a, b, lambda k: (sub_code, str(k), item), seen)
        parent_dest = [] if (code, item) in seen \
            else [(code, parent, None)]
        seen.add((code, item))
        return [*parent_dest, *[(sub_code, k, parent) for k in ks]], None
    parent_dest = [] if (code, item) in seen else [(code, parent, None)]
    seen.add((code, item))
    if (sub_code, sub, item) in seen:
        return parent_dest, None  # re-claimed: idempotent
    seen.add((sub_code, sub, item))
    return [*parent_dest, (sub_code, int(sub), parent)], None


def _attach_chain(destinations, d, code, rest, unit, by_code):
    """A bare nested-unit destination ('c.0,d.0-3' — the model
    comma-joins a parent instance and its ranged sub-entries) folds
    into the parent destination the line already claims: one more
    spelling of the chain, measured on a live draw. Unambiguous
    only — several parent instances on the line keep the named error."""
    parents = [dd for dd in destinations
               if dd[2] is None and isinstance(dd[1], int)
               and by_code[dd[0]].path == unit.parent]
    if len(parents) != 1:
        return [], _rides(code, unit, by_code)
    parent = parents[0][1]
    if m2 := _ITEM_RANGE.fullmatch(_undouble(rest)):
        a, b = int(m2.group(1)), int(m2.group(2))
        if b < a:
            return [], f'{d!r} is unordered'
        destinations.extend((code, k, parent)
                            for k in items_of((a, b))
                            if (code, k, parent) not in destinations)
        return [], None
    if not rest.isdigit():
        return [], _rides(code, unit, by_code)
    if (code, int(rest), parent) not in destinations:
        destinations.append((code, int(rest), parent))
    return [], None


def _plain(d, code, item, unit, seen, multi):
    """The no-chain destinations — itemless whole, plain item, ranged:
    the grammar before chains. A re-claimed destination is idempotent,
    never an error: the model repeats a parent instance when its chains
    share the line ('c.0.d.0,c.0'), and a repair round over a duplicate
    that claims nothing new is a wasted round."""
    if item is None:
        if unit.kind == 'array':
            # bare repeating code: instances unsplit, extracted whole
            if (code, '') in seen:
                return [], None
            seen.add((code, ''))
            return [(code, 0, None)], None
        return [(code, None, None)], None
    if unit.kind == 'array' and (m2 := _ITEM_RANGE.fullmatch(item)):
        a, b = int(m2.group(1)), int(m2.group(2))
        if b < a:
            return [], f'{d!r} is unordered'
        if multi:
            # a ranged claim sharing its line with another destination
            # is the co-chunked shared form spelled compactly —
            # expanded per index; alone on its line it stays the
            # inseparable batched run it means
            return [(code, k, None) for k in
                    _claim_range(a, b, lambda k: (code, str(k)), seen)], None
        if (code, item) in seen:
            return [], None
        seen.add((code, item))
        # ranged run: compact shared form — these chunks carry exactly
        # instances a..b, inseparably (the comma-join's syntax, one
        # destination instead of b-a+1)
        return [(code, (a, b), None)], None
    if (code, item) in seen:
        return [], None
    seen.add((code, item))
    return [(code, _fold(item, unit), None)], None


def _block_lines(code: str, first: int, last: int, declared: int,
                 blocks: int) -> str:
    """The even partition the split ask hands the model to transcribe:
    chunks and instance indexes cut by the same cumulative rule, so the
    lines cover the run's chunks and the declared items exactly — the
    map they patch in validates by construction. Computing the
    boundaries here is the point: the model's own even split of a long
    run fumbled the junk headings between entries (measured: 2 of 5
    canary trials drew an overlapping block, then recovered by
    enumerating every instance). Never more blocks than instances or
    chunks — either way a block would come out empty."""
    blocks = min(blocks, declared, last - first + 1)
    span = last - first + 1
    lines = []
    for j in range(blocks):
        lo = first + j * span // blocks
        end = min(first + (j + 1) * span // blocks - 1, last)
        i0 = j * declared // blocks
        i1 = min((j + 1) * declared // blocks - 1, declared - 1)
        lines.append(f'{lo}-{end} {code}.{i0}' if i0 == i1
                     else f'{lo}-{end} {code}.{i0}-{i1}')
    return '\n'.join(lines)


def _overrun_hint(code: str, first: int, last: int, lines: str) -> str:
    """The material-overflow ask — ``_adopt_blocks``'s fallback, for a
    run whose lines code may not redraw — pure geometry: replace the
    run with
    the ranged block lines computed in code, as a diff against
    the standing map — the replaced line is the short ranged form its
    quote cannot miss (the several-hundred-char comma-join that once
    made diffs unreliable is gone; the base prompt demands the ranged
    form instead). The model ratifies or declines — reading the chunks,
    it knows what code cannot: whether a block boundary would cut an
    entry in half. No rationale — the executor's economics mean nothing
    to the router (measured: 'parallel calls' phrasing was ignored).
    Silence declines."""
    return (f'{code}: rewrite its line(s) into these blocks — one ranged '
            f'line per block, exactly as written:\n{lines}\n'
            f'If the instances truly share their chunks inseparably, '
            f'{_DECLINE}') + _DIFF_REPLY


def _adopt_blocks(text: str, code: str, first: int, last: int,
                  lines: str) -> str | None:
    """The overrun ask's adoption: the computed block lines splice into
    the standing map text in place of the run's own lines. The ask
    round transcribed them verbatim every time (measured) — the
    adoption re-validates by construction, the blocks covering the
    run's chunks and declared items exactly. None when code may not
    own the rewrite: a line inside the run carrying another unit's
    destination or a chain would lose it, and that run's ask keeps
    its round. The input back means there was nothing to rewrite — a
    refired pass after a full adoption."""
    own = re.compile(rf'{code}(?:\.\d+(?:-\d+)?)?')
    out, at = [], None
    for line in text.split('\n'):
        m = _LINE.match(line.strip())
        if not (m and int(m[1]) <= last and int(m[2] or m[1]) >= first):
            out.append(line)
            continue
        if m[3] != '-' and not all(own.fullmatch(t) for t in _dests(m[3])):
            return None  # another unit's destination or a chain rides it
        if at is None:
            at = len(out)  # the blocks take the run's first line's slot
        # else dropped: a later line of the run yields to the blocks
    if at is None:
        return text
    out.insert(at, lines)
    return '\n'.join(out)


def _shared_spans(segments: list, counts: dict) -> tuple:
    """Shared-form array units — ``({code: chunk span}, {code: span})``.
    The first maps every unit riding shared lines (comma-joined
    destinations, a lone item spanning the whole run, or a ranged run —
    the material-overflow ask's candidate pool). The second is the
    subset whose recount is worth a round: a ranged run is the
    whole-run-decode case, always in; a line already carrying one
    destination per instance is decomposed, its recount pure
    decode-shaving, in only past RESHARE_LOAD chunks per instance —
    a unit declared once obeys the same arithmetic (a lone item over a
    handful of chunks reads as one long entry as plausibly as a merge;
    measured: its recount confirmed every time and never adopted).
    Single-chunk runs excluded. A unit riding a line that carries a
    chain (a lifted sub-array's destination) is excluded with them —
    redrawing that line would swallow the chain, and the recount sees
    none of the sub-entries."""
    chained = {dd[0] for _, _, dests in segments
               if any(x[2] is not None for x in dests)
               for dd in dests if dd[2] is None}
    shared, merged = {}, {}
    for start, end, dests in segments:
        if start == end:
            continue
        span = end - start + 1
        per, ranged = {}, set()
        for dd in dests:
            if dd[0] != NONE and dd[2] is None and dd[0] not in chained:
                per[dd[0]] = per.get(dd[0], 0) + 1
                if isinstance(dd[1], tuple):
                    ranged.add(dd[0])
        for d, n in per.items():
            if counts.get(d) == 1 or d in ranged or n > 1:
                shared[d] = shared.get(d, 0) + span
                if d in ranged or span >= n * RESHARE_LOAD:
                    merged[d] = merged.get(d, 0) + span
    return shared, merged


def _lazy_shape(span: int, declared: int) -> bool:
    """The lazy shared shape: instances merged into one run far longer
    than one entry plausibly reads alone. The shared recount's case —
    and the overrun ask's exclusion, so one unit can never draw both
    (the exclusivity is this one predicate, not parallel arithmetic)."""
    return span >= declared * 4 and span - declared >= HINT_MIN


def _shared_hints(spans: dict, counts: dict) -> dict:
    """Array units left in the LAZY shared shape — instances merged into
    one long run, a unit declared once whose lone item spans far more
    chunks than one entry plausibly reads alone (the recount-worthy
    subset of ``_shared_spans``; already-decomposed thin lines are
    filtered there). Several chunks per
    instance may still separate at chunk boundaries (measured lazy
    draw: a whole section mapped as one shared item). A genuinely huge
    single entry is indistinguishable from the merged form — the
    recount lets the model confirm it; a declined suggestion costs one
    round. Already-separated maps (each instance its own line) never
    fire; small excesses don't pay for the round. Returns ``{code:
    (chunk span, declared)}``."""
    return {code: (span, declared)
            for code, span in spans.items()
            if (declared := counts.get(code) or 0)
            and _lazy_shape(span, declared)}


def _split_hint(code: str, span: int, declared: int) -> str:
    """The lazy shared-run fallback ask: separate one line per instance
    where chunk boundaries can, decline by staying silent otherwise."""
    return (f'{code}: {span} chunks carry {declared} instance(s). If the '
            f'chunk boundaries can separate the instances, re-emit with one '
            f'line per instance and a matching count ("5-9 {code}.0", '
            f'"10-11 {code}.1"). If the instances truly share their chunks, '
            f'{_DECLINE}')


def _overrun_hints(segments: list, spans: dict, counts: dict,
                   chunks: list[str], by_code: dict, budgets: dict,
                   derived: dict) -> dict:
    """Array units whose shared run holds more mapped material than one
    shared call may absorb — the dense draw (every instance riding its
    section's range) that the lazy ratio cannot see: there span ≈
    declared, while the lazy shape needs span >= declared * 4. The ask
    sizes the split from the unit's own arrangement: total arranged
    budget over one call's capacity gives the block count (a unit with
    no arrangement falls back to its material chars — the ``@`` suffix
    is tolerated, never checked); the lines are computed here and
    adopted in code (``_adopt_blocks``) — the ask round only survives
    where the adoption declines the redraw.
    Lazy-shaped units stay the recount's (the anchored model will not
    unmerge those, measured). Returns ``{code: (first chunk, last
    chunk, block lines)}``."""
    wanted = [code for code, span in spans.items()
              if (declared := counts.get(code) or 0) >= HINT_MIN
              and not _lazy_shape(span, declared)]
    if not wanted:
        return {}
    bounds = {}
    for start, end, dests in segments:
        for c in {dd[0] for dd in dests if dd[2] is None and dd[0] != NONE}:
            lo, hi = bounds.get(c, (start, end))
            bounds[c] = (min(lo, start), max(hi, end))
    paths = {code: by_code[code].path for code in wanted}
    material = _material([a for a in _assignments(segments, by_code)
                          if a['unit'] in paths.values()],
                         chunks, {by_code[d].path: by_code[s].path
                                  for d, s in derived.items()})
    resolved = _resolved({by_code[c].path: b for c, b in budgets.items()},
                         material)
    hints = {}
    for code in wanted:
        chars = sum(material.get(paths[code], {}).values())
        if chars < SPLIT_MATERIAL_CAP:
            continue
        total = resolved.get(paths[code])
        total = sum(total) if isinstance(total, list) else (total or 0)
        if not total:
            total = chars
        if (blocks := round(total / BATCH_BUDGET_CAP)) >= 2:
            first, last = bounds[code]
            hints[code] = (first, last,
                           _block_lines(code, first, last, counts[code],
                                        blocks))
    return hints


_PASSABLE = re.compile(r'^([a-z]+): declared \d+ items ')


def _passable(errors: list) -> set:
    """Codes whose declared-vs-used mismatch a replayed map may pass
    through — top-level, per-parent, and assigns-none alike. The
    extraction's count arbitration owns the number (it settles
    declared-vs-actual by diffing entry openings); the router owns
    chunk allocation."""
    return {m[1] for e in errors if (m := _PASSABLE.match(e))}


def _chain_remedy(start, end, dests, prev_start, prev_end, prev_dests,
                  segments, by_code):
    """The concrete remedy for a chain-involved overlap, when one
    exists: a slice re-drawing a destination a wider line holds names
    the removal that lets the slices replace it; a parent line
    covering its own chain lines names the chunks the parent may keep
    and the count it must then match; two sub-entry slices of one
    parent instance overlapping name the earlier slice's end — the
    chunk before the later one begins. None when the overlap is
    ordinary. The containment check reads ranged parent claims
    deliberately — hosting (`_hosted_chain`) stays strict: a ranged
    run is the inseparable batched form, and only the remedy names
    its chains' chunks."""
    if not any(dd[2] is not None for dd in dests):
        return None
    if prev_start <= start and end <= prev_end and any(
            dd[2] is not None and dd in prev_dests for dd in dests):
        # a slice re-drawing a destination the wide line already
        # holds is a split whose whole the model forgot to remove
        # (measured: the split added both halves, the original line
        # stayed, and the rounds no-op'd out) — name the removal
        return (f'add "- {prev_start}-{prev_end} '
                f'{_dest_tokens(prev_dests, by_code)}" and the slices '
                f'replace it')
    prev_plain = [(dd[0], dd[1]) for dd in prev_dests if dd[2] is None]
    prev_parents = _chain_parents(prev_dests, by_code)
    for dd in dests:
        if dd[2] is None:
            continue
        pcode, pitem = _parent_code(by_code[dd[0]], by_code), dd[2]
        if not any(cc == pcode and pitem in items_of(item)
                   for cc, item in prev_plain):
            continue
        if prev_start < start and (pcode, pitem) in prev_parents:
            return (f'the two chain lines are sub-entries of one instance '
                    f'— each takes its own slice: end the earlier one at '
                    f'{start - 1}, where '
                    f'{_chain_label(dd[0], pitem, by_code)}.{dd[1]} begins')
        spans = [(s2, e2) for s2, e2, ds2 in segments for dd2 in ds2
                 if dd2[2] is not None
                 and _parent_code(by_code[dd2[0]], by_code) == pcode
                 and dd2[2] == pitem]
        outside = [c for c in range(prev_start, prev_end + 1)
                   if not any(s2 <= c <= e2 for s2, e2 in spans)]
        named = ', '.join(str(a) if a == b else f'{a}-{b}'
                          for a, b in _runs(outside))
        if not named:
            return (f'the parent line\'s chunks are all its chains\' — '
                    f'drop the parent line, the chains feed the parent')
        return (f'the parent line covers chunks its own chain lines carry '
                f'— the parent keeps only the chunks outside its chains '
                f'({named}); then the count line must match the '
                f'{pcode}-items the map draws')
    return None


def _uncovered_removal(removed: list, missing: list) -> str | None:
    """The round's own removal that uncovered the missing chunks, if
    one did — the blame the feedback names for a re-add. When this
    fires it takes precedence over the extend-the-neighbour fold: the
    removal is the cause, and the neighbour may hold a foreign unit."""
    for l in removed:
        if (m := _LINE.match(l)) and any(
                int(m[1]) <= c <= int(m[2] or m[1]) for c in missing):
            return l
    return None


def _map_errors(segments, counts, nested, derived, by_code: dict,
                n: int, lenient: set | None = None) -> list:
    """Whole-map consistency: line-range overlap, declared-vs-used
    items per destination — top level and, per parent instance, lifted
    sub-arrays — derivation sanity, and full coverage (reported even
    alongside line errors, so the model gets the complete picture in
    one repair). A pure chain line claims no coverage of its own: it
    must sit inside a line that carries its parent destination, where
    it reads as the fine partition of that run — each sub-entry keeps
    its own slice and fans out on its own; overlapping it with anything
    else is an error."""
    covering, chains = [], []
    for seg in segments:
        if _pure_chain(seg[2]):
            chains.append(seg)
        else:
            covering.append(seg)
    errors, prev_start, prev_end, prev_dests = [], -1, -1, ()
    for start, end, dests in covering:
        if start <= prev_end:
            codes = ','.join(sorted({dd[0] for dd in prev_dests}))
            tail = _chain_remedy(start, end, dests, prev_start, prev_end,
                                 prev_dests, segments, by_code)
            if tail is None and start == end == prev_end:
                shared = sorted(
                    {dd[0] for dd in dests} & {dd[0] for dd in prev_dests}
                    & {c for c, u in by_code.items()
                       if u.kind == 'array' and u.parent is None})
                tail = (f'two {shared[0]} instances on one chunk ride one '
                        f'line, e.g. "{prev_end} {shared[0]}.0,{shared[0]}.1"'
                        if shared else
                        'share the chunk on one line, e.g. '
                        f'"{start} x.0,y.0", or shorten or drop one of '
                        'the two lines')
            if tail is None:
                # a multi-chunk overlap is two claims on the same
                # material — the ride hints would point the wrong way
                tail = ('draw each chunk on exactly one line — shorten or '
                        'drop whichever line covers material another line '
                        'already carries')
            errors.append(f'segment {start}-{end} overlaps line '
                          f'{prev_start}-{prev_end} ({codes}) — every '
                          f'chunk sits on exactly one line; {tail}')
        if end > prev_end:
            prev_start, prev_end, prev_dests = start, end, dests
    for start, end, dests in chains:
        if not _hosted_chain(start, end, _chain_parents(dests, by_code),
                             covering):
            pcode = _parent_code(by_code[dests[0][0]], by_code)
            errors.append(
                f'{dests[0][0]}: chain line {start}-{end} rides no run of '
                f'its parent — draw {pcode}.<item>\'s own line over the '
                f'chain lines, or put the chains on the parent\'s line '
                f'("{start}-{end} {pcode}.<item>.{dests[0][0]}.<sub>")')
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
        if unit.parent is not None:
            errors += _nested_errors(code, unit, segments, counts, nested,
                                     by_code, lenient)
            continue
        items, ranged_seen, ranged_warned = set(), {}, False
        for _, _, dests in segments:
            for dd in dests:
                if dd[0] != code:
                    continue
                item = dd[1]
                if isinstance(item, tuple):
                    for i in items_of(item):
                        if i in ranged_seen and not ranged_warned:
                            # one named overlap per code — a messy draw
                            # must not flood the repair prompt with a
                            # sentence per instance
                            ranged_warned = True
                            errors.append(
                                f'{code}: instance {i} is ranged twice '
                                f'({ranged_seen[i]} and {item}) — a ranged '
                                f'run takes each instance once')
                        ranged_seen.setdefault(i, item)
                        items.add(i)
                else:
                    items.add(item)
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
        if items != set(range(declared)) and code not in (lenient or ()):
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
        # bare on purpose: the extend-the-neighbour fold composes at
        # the feedback layer, where the round's own removals are known
        # and a removal-caused hole blames the removal instead
        errors.append(f'chunks not covered: {missing}')
    return errors


def _coverage_hint(segments, missing: list, by_code: dict) -> str:
    """The concrete fold for a hole that continues a line: extend the
    preceding line over it, or mark the chunks '-'. Bare when the hole
    does not follow one line — the cascade this names (delete a chain
    line, its chunks go homeless) burned a prod exhaustion."""
    start, end = _runs(missing)[0]
    prev = next((seg for seg in segments if seg[1] == start - 1), None)
    if prev is None or len(prev[2]) != 1:
        return ''
    dd = prev[2][0]
    if dd[1] is None or by_code.get(dd[0]) is None:
        return ''  # an itemless or unknown-code dest names no extension
    return (f' — extend the preceding line over them '
            f'("{prev[0]}-{end} '
            f'{_dest_token(dd[0], dd[1], dd[2], by_code)}"), '
            f'or mark them \'-\'')


def _overdraw_remedy(segments, code: str, parent: int, declared: int,
                     extra: int, by_code: dict) -> str:
    """The concrete remedy for a sub-item index beyond the declared
    count: the previous sub-entry's line extended over the extra
    line's chunks — bare deletion orphans them and the repair
    whipsaws — or the count itself is wrong."""
    prev = _line_of(segments, code, declared - 1, parent)
    mine = _line_of(segments, code, extra, parent)
    if prev and prev[0] <= mine[0]:
        return (f'— extend the previous sub-entry\'s line over its chunks '
                f'("{prev[0]}-{mine[1]} '
                f'{_dest_token(code, declared - 1, parent, by_code)}"), '
                f'or the count is wrong — declare what the document holds')
    return ('— its line\'s chunks ride another sub-entry\'s line, or the '
            'count is wrong — declare what the document holds')


def _nested_errors(code: str, unit: Unit, segments, counts, nested,
                   by_code: dict, lenient: set | None = None) -> list:
    """Per-parent consistency for one lifted sub-array: the chain's
    parent instance must exist under the parent unit's declared count,
    and every parent naming a count sees its sub-items 0..n-1 — the
    declared-vs-used contract one level down. A parent holding no
    sub-entries declares nothing, so a missing declaration is only an
    error where the map already claims sub-entries."""
    pcode = next(c for c, u in by_code.items() if u.path == unit.parent)
    used = {}
    for _, _, dests in segments:
        for dd in dests:
            if dd[0] == code:
                for i in items_of(dd[1]):
                    used.setdefault(dd[2], set()).add(i)
    declared = {p: v for (c, p), v in nested.items() if c == code}
    errors = []
    for p in sorted(used.keys() | declared.keys()):
        chain = _chain_label(code, p, by_code)
        if code in (lenient or ()) and p in declared:
            continue  # passed through after a replay — the count
            # arbitration owns it (used-but-undeclared still blocks:
            # with no declaration there is nothing to arbitrate)
        if p in used and p in declared:
            if used[p] != set(range(declared[p])):
                if declared[p] and (extra := min(
                        (i for i in used[p] if i >= declared[p]),
                        default=None)):
                    remedy = _overdraw_remedy(segments, code, p,
                                              declared[p], extra, by_code)
                    errors.append(
                        f'{code}: sub-item {extra} under {pcode}.{p} is beyond '
                        f'the declared {declared[p]} '
                        f'(0..{declared[p] - 1}) {remedy}')
                else:
                    errors.append(
                        f'{code}: declared {declared[p]} items under {pcode}.{p} '
                        f'but the map uses {sorted(used[p])} — give each '
                        f'sub-entry its own line where chunk boundaries can '
                        f'separate them ("5-9 {chain}.0", "10-11 {chain}.1"); '
                        f'instances that share one chunk ride one line '
                        f'("5 {chain}.0,{chain}.1"); or the count is wrong — '
                        f'declare what the document holds')
        elif p in used:
            errors.append(
                f'{code}: sub-entries used under {pcode}.{p} but their '
                f'count is undeclared — declare "{chain}: <n>"')
        else:
            # single remedy, stated once: the declaration stands (it was
            # the map that skipped the sub-entries) — offering "drop the
            # declaration" here read as an equal branch and the model
            # took it after drawing the chains, then got whipsawed by
            # this same check one round later (measured). Every missing
            # index named: a draw that runs one chain short is where
            # the split cascades start
            names = [f'"{chain}.{i}"'
                     for i in range(max(1, min(declared[p], 4)))]
            ellipsis = ', …' if len(names) < declared[p] else ''
            errors.append(
                f'{code}: declared {declared[p]} items under {pcode}.{p} '
                f'but the map assigns none — add the chain lines '
                f'({", ".join(names)}{ellipsis}) as "+" diff lines, '
                f'everything else stays; '
                f'keep the declaration: it is correct when the document '
                f'holds the sub-entries the lines should cover')
    if isinstance(have := counts.get(pcode), int):
        for p in sorted(p for p in used if p >= have):
            errors.append(
                f'{code}: chained under {pcode}.{p} but {pcode} declares '
                f'{have} items — the parent instance does not exist')
    return errors


def _groups(segments: list, by_code: dict, chunks: list[str],
            derived: dict = None) -> list:
    """Validated line ranges are disjoint by construction; each
    destination gets its own group, chunk-adjacent same-destination
    pieces coalesce (a shared unit's items may interleave with the
    detailed unit's; a parent instance's chunks coalesce across the
    chain lines that ride them). Derived units mirror their source's
    groups."""
    groups, last = [], {}
    for start, end, destinations in segments:
        ids, text = list(range(start, end + 1)), '\n'.join(
            chunks[i] for i in range(start, end + 1))
        for dest in destinations:
            if dest[0] == NONE:
                continue
            key = (dest[0], dest[1], dest[2])
            if g := last.get(key):
                if g.chunk_ids[-1] == start - 1:  # adjacent piece, same destination
                    g.chunk_ids.extend(ids)
                    g.text += '\n' + text
                    continue
            g = Group(by_code[dest[0]],
                      None if isinstance(dest[1], tuple) else dest[1],
                      list(ids), text,
                      items=tuple(dest[1]) if isinstance(dest[1], tuple) else (),
                      parent=key[2])
            last[key] = g
            groups.append(g)
    for code, source in (derived or {}).items():
        groups += [Group(by_code[code], g.item, list(g.chunk_ids), g.text,
                         items=g.items)
                   for g in groups if g.unit is by_code[source]]
    return groups
