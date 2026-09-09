"""Fresh-conversation checks: the anchoring discipline, and the count
arbitration built on it.

A fresh call carries no history and no feedback — the model never sees
the map or the extraction it is re-examining (a round shown its own
answer re-emits it verbatim, measured). The discipline lives here so a
future "give it more context" edit cannot silently restore the anchor;
the router's recount paths are its first consumers, the count
arbitration its second: when a unit's declared count and its extracted
actual disagree, the arbiter diffs the extraction's entry openings
against the document — what is missing from the list, what on the list
the document does not contain — and the count revises by the verdict.
"""

from __future__ import annotations

import re

from xtremeparse.contracts import AgentRunner


def norm(s: str) -> str:
    """Whitespace-collapsed text for content seeks — inverts router
    _listing's ' ¶ ' newline marker and eats the blanks layouts pad
    their lines with, so a model quote matches the chunk it came
    from."""
    return ' '.join(s.replace(' ¶ ', ' ').split())


async def fresh_check(runner: AgentRunner, payload: str,
                      instructions: str, description: str) -> str:
    """One fresh-conversation call — no history (the model never sees
    the answer it would anchor on and re-emit verbatim, measured), no
    feedback. Shared by every check prompt this module and the router
    run."""
    result = await runner.run(instructions=instructions,
                              result_schema={'type': 'string',
                                             'description': description},
                              content=payload, feedback=None)
    return str(result.data)


def unmarked(line: str) -> str:
    """A reply line minus its enumeration marker — the list forms
    models add around quoted lines ('1. x', '- x', '* x'); both reply
    parsers strip it before reading the quote."""
    return re.sub(r'^(?:\d+\s*[.、)]|[-*•])\s*', '', line).strip()


ARBITRATION_LIST_CAP = 120  # entry openings one arbitration carries —
# above this the diff question loses the model; rare (whole-array
# units that large rarely dispute)


def _openings(items: list) -> list:
    """Each extracted item's opening text — its longest string value,
    capped — the form an arbitration diff quotes the list by."""
    out = []
    for item in items:
        if isinstance(item, dict):
            texts = [v for v in item.values() if isinstance(v, str) and v]
            out.append(max(texts, key=len)[:80] if texts else '')
        elif item is not None:
            out.append(str(item)[:80])
    return [o for o in out if o]


_ARBITRATION_CHECK = '''The context carries a document. Below is the list of
instances of one repeating unit extracted from it, each quoted by its
opening text:

{list}

'''
# the answer skeleton carries no holes — a list spliced into a verdict
# section is unrepresentable, so the pre-filled-verdict failure mode
# (measured on a live trace) cannot come back through a template edit
_ARBITRATION_SKELETON = '''Re-read the document and check the list: which document
instances are MISSING from the list, and which list entries does the
document NOT contain? Quote each as its opening text, one per line,
verbatim — enough text to identify it, no commentary. When the list is
complete and faithful, answer with both sections empty.

missing:

extra:
'''


def sectioned(result, is_label) -> dict:
    """A fresh-check reply split into ``{label: [quoted lines]}`` — a
    label line opens its section, every other non-blank line appends to
    the open one with its enumeration marker stripped. The one walk for
    both reply grammars (router recount counts, arbitration verdicts):
    the caller supplies what counts as a label."""
    sections, current = {}, None
    for line in (l.strip() for l in str(result or '').splitlines()):
        if not line:
            continue
        stripped = unmarked(line)
        label = is_label(stripped)
        if label is not None:
            current = sections.setdefault(label, [])
        elif current is not None:
            current.append(stripped)
    return sections


def _label(line: str) -> str | None:
    """A verdict section label ('missing:'/'extra:'), else None."""
    label = line.lower().rstrip(':')
    return label if label in ('missing', 'extra') else None


def _parse_arbitration(result, doc: str, entries: list) -> dict:
    """``{'missing': [...], 'extra': [...]}`` from an arbitration reply.
    Void the verdict when: a section label is absent; a missing quote
    fails to anchor to the document; an extra quote appears in the
    document itself — a provably false claim, the entry is real
    wherever the list holds it; an extra quote matches no list entry.
    A partial verdict invites adopting a half-read answer."""
    buckets = sectioned(result, _label)
    if set(buckets) != {'missing', 'extra'}:
        return {}  # not a verdict at all — before any document scan
    # normalize each section once — both guards over it re-read the
    # same quotes, and a blank quote voids below like any unanchored one
    missing = [norm(q) for q in buckets['missing']]
    extra = [norm(q) for q in buckets['extra']]
    doc_n = norm(doc) if missing or extra else ''
    if not all(n and n in doc_n for n in missing):
        return {}
    if not all(n and n not in doc_n for n in extra):
        return {}
    if extra:
        entries_n = [norm(e) for e in entries]
        if not all(any(n in e for e in entries_n) for n in extra):
            return {}
    return {k: list(dict.fromkeys(v)) for k, v in buckets.items()}


CHECK_DESCRIPTION = ("A check verdict: 'missing:' and 'extra:' sections "
                     "of quoted opening texts")


async def arbitrate_extraction(runner: AgentRunner, *, items: list,
                               actual: int, payload: str) -> tuple:
    """Fresh-conversation check of an extraction against the document
    it came from — the arbiter when a unit's declared count and its
    extracted actual disagree (a shortfall repair against a phantom
    slot can only fabricate entries; an over-count merge against an
    under-declared one can only discard real ones; and a fresh
    free-form recount cannot count at all here — asked to enumerate a
    dense eighty-entry run it answers "0", asked for a plain count it
    lands a third of the way off: measured). Counting is the one thing
    the model does worst and diffing the one it does best: hand it the
    extraction's entry openings and ask what is missing from the list
    and what on the list the document does not contain. No history, no
    feedback — the anchoring discipline of fresh_check. Returns
    ``(revised, raw)`` — the count revised by the anchored verdict
    (``actual + missing - extra``), or None when the reply is unusable
    — including a both-empty verdict on an empty extraction; the
    caller falls back by direction then."""
    listed = _openings(items)[:ARBITRATION_LIST_CAP]
    instructions = (_ARBITRATION_CHECK.format(
        list='\n'.join(f'- {e}' for e in listed)) + _ARBITRATION_SKELETON)
    raw = await fresh_check(runner, payload, instructions, CHECK_DESCRIPTION)
    if verdict := _parse_arbitration(raw, payload, listed):
        if not verdict['missing'] and not actual:
            return None, raw  # an empty list cannot bless itself: both
            # sections empty asserts zero instances and no guard can
            # anchor a quoteless verdict — measured live, a collapse to
            # [] got approved and wiped entries a recount had located.
        return actual + len(verdict['missing']) - len(verdict['extra']), raw
    return None, raw
