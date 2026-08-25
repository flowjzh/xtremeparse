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
exactly that shape. The declared counts make the map self-consistent:
item indexes must run exactly 0..declared-1 and every declared item
must receive chunks. Coverage and order hold per line range,
uniqueness per destination; violations get bounded repairs with
precise feedback, then RouterError — a validated map's line ranges are
disjoint by construction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from xtremeparse.contracts import AgentRunner, JSONSchema
from xtremeparse.prompting import json_len
from xtremeparse.units import Unit

NONE = '-'
# placeholders every prompt template must carry (host overrides are
# validated against these; see docs/prompting.md)
ROUTE_PLACEHOLDERS = frozenset({'top', 'none', 'legend', 'chunks'})
RECOUNT_PLACEHOLDERS = frozenset({'legend', 'chunks'})
_LINE = re.compile(r'^(\d+)(?:-(\d+))?\s+('
                   r'[a-z]+(?:\.\d+)?(?:\s*,\s*[a-z]+(?:\.\d+)*)*|-(?:\.\d+)?)$')
_COUNT = re.compile(r'^([a-z]+)(?::\s*(\d+))?(?:\s*=\s*([a-z]+))?(?:\s*@.*)?$')
_BUDGET = re.compile(r'@\s*([0-9][0-9xX%,\s]*)')


def _budget(token: str):
    """One declared suffix entry: 'n%' a ratio of the item's mapped
    material, 'nxm' an average keyword length times a keyword count,
    'n' an absolute cap (the scaled forms add the item schema's
    skeleton upstream). None on garbage — the suffix is tolerated,
    never checked."""
    if m := re.fullmatch(r'(\d+)%', token):
        return Budget('ratio', int(m.group(1)))
    if m := re.fullmatch(r'(\d+)[xX](\d+)', token):
        return Budget('kw', int(m.group(1)), int(m.group(2)))
    if token.isdigit():
        return Budget('abs', int(token))
    return None

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
  reverse chronological order'), follow the card's declared order
  instead — the card's rule overrides document order; the count line
  after the map reports the highest index plus one, 0 for a unit the
  document does not contain at all. A unit with no text of its own
  that only summarizes another unit declares just its source and
  takes NO map lines:
  "x = y @30%" — its items (and its count) mirror the source's
  material, but its budget is its own: set it for the lines the
  summary will emit — a ratio scales against the source material it
  summarizes, so a terse summary sits well below 100%.
- A declaration line ends with an output-budget suffix — one entry per
  item in item-index order ("x: 3 @100%,80%,50%", or "x: 3 @100%" when
  items share one): an estimate of the characters that item's output
  JSON will run to. A ratio, "@<n>%", is the DEFAULT form — the
  output as that percentage of the item's mapped material. It fits
  every item whose output mirrors the material field by field: the
  same facts, carried into the schema's fields. Set the ratio from
  the field descriptions in the JSON Schema (the shared context):
  fields told to keep every listed entry or every figure stay near
  100%, fields told to summarize or compress sit well below (say
  30-50%); a verbatim copy is "@100%". Two shapes take another form
  instead:
  - a list of short, same-shaped entries (certificates, skills,
    tags): the average length of one entry times the number of
    entries the document holds, "@<avg>x<count>" — a list of 3
    certificates averaging 20 characters is "@20x3";
  - a fixed-length summary — the card or schema pins the output size
    ("one line", "about 100 characters") — or an item whose output
    does not mirror the material at all: a plain character estimate,
    "@<n>" ("x: 3 @300"). Scalar fields cost their key name plus
    four punctuation characters and the value: a date ≈7, a name
    ≈10, a one-line title ≈20.
  Every declaration line carries one.
- Map lines ascend, never overlap, and together cover EVERY chunk id in 0..{top}.
- A run may feed several DIFFERENT units at once: comma-join their
  codes ("5 x.0,y.0") — a summary or cross-cutting unit rides the lines
  of the unit whose text it shares, item by item:
      5 x.0,y.0
      6 x.1,y.1
- Several items of the SAME unit may share one line when the run holds
  several instances that chunk boundaries cannot separate
  ("3 x.0,x.1,x.2"): that material is then extracted once, whole.
  Prefer one item per line whenever boundaries do separate instances.
- Every declared item receives at least one chunk. A unit sharing
  another's run may repeat one item across several lines (one instance
  spanning what another unit splits into many).
- Each item is ONE instance of the repeating unit — a complete entry.
  Start a new item whenever the text moves to the next instance; never
  merge distinct instances into one, and never split one instance
  across items.
- Code {none} on its own (no item) marks chunks irrelevant to every unit.
- Before answering, verify: every repeating unit has a declaration line
  (its count "x: <n>" or its source "x = y"), the map covers 0..{top}
  exactly once in ascending non-overlapping lines, item indexes of
  each unit run 0..used-1 with no gaps, and every count line matches
  the map.

Unit codes:
{legend}

Chunks:

{chunks}'''


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
  then add its map lines in the routing DSL ("4-5 x.0,x.1"), one item
  per instance, covering exactly its material.
0 only for a unit the document truly does not contain.

Repeating units (code = unit card; RECOUNT = left at zero):
{legend}

Chunks:

{chunks}'''


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
    budgets: dict = None  # unit path → resolved per-item output estimate in
    # absolute chars (the declared forms ride in raw['budgets'])


@dataclass(frozen=True)
class Budget:
    """One declared output estimate: 'abs' a fixed character cap,
    'ratio' a percentage of the item's mapped material, 'kw' an
    average keyword length times a keyword count — both scaled forms
    sit on top of the item schema's skeleton."""

    kind: str
    value: int
    count: int = 1  # kw only — the document's entry count

    def __post_init__(self):
        assert self.kind in ('abs', 'ratio', 'kw'), self.kind

    def chars(self, material: int, skeleton: int) -> int:
        """One item's absolute estimate — a ratio scales against the
        mapped material (whitespace stripped) it is declared over, a
        keyword total adds the same structure overhead; key names and
        punctuation are code-known, not the model's to estimate."""
        if self.kind == 'ratio':
            return skeleton + round(self.value / 100 * material)
        if self.kind == 'kw':
            return self.value * self.count + skeleton
        return self.value

    def per_item(self, material: dict, skeleton: int) -> list:
        """The estimate for every item this declaration covers: a
        shared ratio scales against each item's own material, a shared
        keyword total splits across the items it covers, anything else
        is one number covering all (the executor repeats a lone
        value)."""
        if self.kind == 'ratio' and material:
            return [self.chars(material[i], skeleton)
                    for i in sorted(material, key=lambda k: (k is None, k))]
        if self.kind == 'kw' and material:
            share = round(self.value * self.count / len(material))
            return [share + skeleton] * len(material)
        return [self.chars(0, skeleton)]

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
    and the model's index arithmetic with it."""
    return '\n'.join(f'[{i}] {c.replace(chr(10), " ¶ ")}'
                     for i, c in enumerate(chunks))


async def route(runner: AgentRunner, *, payload: str,
                units: list[Unit], chunks: list[str],
                instructions: str = None,
                recount_instructions: str = None) -> Routing:
    """Map chunks to units via one agent call, with bounded repairs
    feeding the validation errors back as ``feedback`` — plus one
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
                         for code, unit in by_code.items()),
        chunks=_listing(chunks))
    schema: JSONSchema = {'type': 'string',
                          'description': 'Segment map and count declarations '
                                         'only — no prose, no JSON.'}
    feedback, last, history, verified = None, None, [], False  # per-round
    # error lists: a repaired error that later reappears means the model
    # is rewriting fixed lines away — name it
    for _ in range(5):  # initial call + four bounded repairs
        result = await runner.run(instructions='' if last else instructions,
                                  result_schema=schema, content=payload,
                                  feedback=feedback,
                                  history=last.history if last else None)
        last = result
        errors, counts, derived, segments, budgets = _parse(
            result.data, by_code, len(chunks))
        if not errors:
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
                    feedback = [_RouteIssue('segments', 'route_invalid',
                                f'{", ".join(zeros)}: a separate recount of '
                                'the document disagreed with this map but '
                                'could not be merged — for each, recheck the '
                                'chunk list yourself: give every instance '
                                'its map lines with a matching count, or '
                                'declare "= <source>" if it only summarizes '
                                'another repeating unit; keep 0 only if '
                                'truly absent')]
                    history.append([feedback[0].message])
                    continue
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
                           for d, s in derived.items()}) if budgets_by_path else {},
                {by_code[c].path: _skeleton(by_code[c].sub_schema)
                 for c in budgets})
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
        feedback = [_RouteIssue('segments', 'route_invalid', m) for m in marked]
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
        # one with its own text (answers a count)
        legend='\n'.join(f'{c} = {by_code[c].header}'
                         + (' RECOUNT' if c in zeros else '')
                         for c, u in by_code.items() if u.kind == 'array'),
        chunks=_listing(chunks))
    result = await runner.run(instructions=instructions,
                              result_schema={'type': 'string',
                                             'description': 'Declarations and '
                                                            'map lines for the '
                                                            'RECOUNT units only'},
                              content=payload, feedback=None)
    return str(result.data)


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
    rebuilt = []
    for c in range(n):  # re-segment, extending runs of equal destinations
        dests = claimed.get(c) or cover[c]
        if rebuilt and rebuilt[-1][2] == dests:
            rebuilt[-1] = (rebuilt[-1][0], c, dests)
        else:
            rebuilt.append((c, c, dests))
    if _map_errors(rebuilt, counts, derived, by_code, n):
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


def _empty(schema) -> dict | list | str:
    """The compact JSON of an empty value of this schema — the
    recursive sibling of json_len: nested objects cost their keys,
    an array counts one empty entry, the rest of the entries are
    content the ratio covers."""
    if props := schema.get('properties'):
        return {k: _empty(v) for k, v in props.items()}
    if schema.get('type') == 'array':
        return [_empty(schema.get('items') or {})]
    return ''


def _skeleton(schema) -> int:
    """The compact JSON length of an empty item — key names and
    punctuation every ratio's content estimate must add on top of."""
    return json_len(_empty(schema))


def _resolved(budgets, material, skeletons) -> dict:
    """Absolute per-item estimates the executor compares against its
    batch capacity. A lone Budget expands to every item it covers; a
    list carries each item's own declaration."""
    out = {}
    for path, declared in budgets.items():
        skeleton = skeletons[path]
        if isinstance(declared, Budget):
            values = declared.per_item(material.get(path, {}), skeleton)
        else:
            by_item = material.get(path, {})
            values = [b.chars(by_item.get(i, 0), skeleton)
                      for i, b in enumerate(declared)]
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
    destinations)`` with destinations a tuple of ``(code, item|None)``,
    derived a ``{code: source}`` map of summarizing units, budgets a
    ``{code: Budget}`` map of the (tolerated, never checked) ``@``
    declarations."""
    if not isinstance(text, str) or not text.strip():
        return (['output must be the count lines and segment map, nothing else'],
                {}, {}, [], {})
    errors, counts, derived, segments, budgets = [], {}, {}, [], {}
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
                          f'"<start>-<end> <code>[.<item>][,...]" or "-"{hint}')
            continue
        start, end = int(m.group(1)), int(m.group(2) or m.group(1))
        if start >= n or end >= n or end < start:
            errors.append(f'line {i + 1}: range {start}-{end} is unordered '
                          f'or outside 0..{n - 1}')
            continue
        destinations, seen = [], set()
        for d in (t.strip() for t in m.group(3).split(',')):
            code, _, item = d.partition('.')
            unit = by_code.get(code)
            if code == NONE and item:
                errors.append(f'line {i + 1}: {NONE} takes no item index')
            elif code == NONE:
                destinations.append((NONE, None))
            elif unit is None:
                errors.append(f'line {i + 1}: unknown unit code {code!r}')
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
        if segment not in segments:  # an exact re-emitted line is redundancy
            segments.append(segment)
    errors += _map_errors(segments, counts, derived, by_code, n)
    # exact duplicates collapse — one repeated destination must not
    # flood the repair feedback with the same message
    errors = list(dict.fromkeys(errors))
    return errors, counts, derived, [] if errors else segments, budgets


def _map_errors(segments, counts, derived, by_code: dict, n: int) -> list:
    """Whole-map consistency: line-range order/overlap, declared-vs-used
    items per destination, derivation sanity, and full coverage
    (reported even alongside line errors, so the model gets the
    complete picture in one repair)."""
    errors, prev_end = [], -1
    for start, end, _ in segments:
        if start <= prev_end:
            errors.append(f'segment {start}-{end} overlaps or breaks order '
                          f'after chunk {prev_end}')
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
                    f' — several instances may share one run: comma-join them '
                    f'on that line, e.g. "5 {code}.0,{code}.1"')
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
