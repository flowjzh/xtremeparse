"""The verbatim contract: a schema-declared rule the pipeline enforces.

docxcast derives an ``@verbatim[=ratio]`` Tag directive into the JSON
Schema extension keyword ``x-verbatim`` (the tolerance as a float);
hand-written schemas set the keyword directly. One marker, two
enforcements: units carrying marked fields get their output budget
computed from the routed chunk material (a verbatim echo costs what its
source costs — the router's declared estimate is discarded for those
paths), and a deterministic validator re-checks every extracted string
against the normalized document, feeding failures to the correction
loop as patchable issues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from xtremeparse.chunking import content_len
from xtremeparse.units import type_set

VERBATIM_DEFAULT_RATIO = 0.9  # the protocol default; docxcast resolves the bare marker
MARKDOWN_MARKERS = '*_#`'
_MD_MARKS = str.maketrans('', '', MARKDOWN_MARKERS)


def marker_ratio(prop: dict) -> float | None:
    """The field's ``x-verbatim`` tolerance — None when unmarked.
    docxcast emits the resolved ratio; a malformed value on a
    hand-written schema still arms the contract at the default rather
    than silently disarming it."""
    if 'x-verbatim' not in prop:
        return None
    try:
        ratio = float(prop['x-verbatim'])
    except (TypeError, ValueError):
        return VERBATIM_DEFAULT_RATIO
    return ratio if 0 < ratio <= 1 else VERBATIM_DEFAULT_RATIO


def normalize_text(s: str) -> str:
    """Whitespace- and markdown-insensitive form: the source wraps lines
    mid-word and bolds labels, and a faithful extractor heals both."""
    return ''.join(s.translate(_MD_MARKS).split())


def heading_label(prop: dict) -> str | None:
    """A string field's heading label — its description up to the first
    colon; whole descriptions make hopeless prefixes."""
    desc = prop.get('description') or ''
    if prop.get('type') != 'string':
        return None
    return re.split('[：:]', desc, maxsplit=1)[0].strip()


def verbatim_paths(schema: dict) -> list[str]:
    """Dotted paths of fields carrying the ``x-verbatim`` keyword — the
    template declares the contract, consumers only read it. Unmarked
    schemas get an empty list."""
    out = []

    def walk(node: dict, path: str):
        for name, prop in (node.get('properties') or {}).items():
            p = f'{path}.{name}' if path else name
            if marker_ratio(prop) is not None:
                out.append(p)
            walk(prop.get('items') or {}, p)
            walk(prop, p)

    walk(schema, '')
    return out


@dataclass(frozen=True)
class VerbatimFinding:
    """One item's breach: 'heading' (opens with a sibling's heading
    label — the block belongs to that field) or 'verbatim' (not a
    normalized substring; coverage is the best-window char match)."""

    path: str
    code: str
    item: str
    label: str = ''
    coverage: float = 0.0


def noise_window(source_norm: str, norm: str) -> str | None:
    """The source region the item claims to copy — located by its first
    16-char chunk found in the source, later chunks skipping noise that
    sits inside earlier ones. None when no chunk lands at all."""
    for start in range(0, len(norm), 16):
        chunk = norm[start:start + 16]
        if len(chunk) < 8:
            break
        if (pos := source_norm.find(chunk)) >= 0:
            return source_norm[max(0, pos - start - 24):pos - start + len(norm) + 48]
    return None


def check_item(item: str, path: str, labels: frozenset,
               source_norm: str, min_ratio: float) -> VerbatimFinding | None:
    """The item's breach as a finding, or None when the copy holds."""
    norm = normalize_text(item)
    # a heading line is label + colon; a duty line merely opening with
    # the same word ('部门经理…') is description text, not a block
    hit = next((l for l in labels
                if (nl := normalize_text(l)) and norm.startswith((nl + '：', nl + ':'))),
               None)
    if hit:
        return VerbatimFinding(path, 'heading', item, label=hit)
    if norm in source_norm:
        return None
    coverage = 0.0
    if window := noise_window(source_norm, norm):
        coverage = sum(b.size for b in
                       SequenceMatcher(None, norm, window).get_matching_blocks()) / len(norm)
        if coverage >= min_ratio:
            return None
    return VerbatimFinding(path, 'verbatim', item, coverage=coverage)


def find_violations(schema: dict, data: dict, source_norm: str,
                    min_ratio: float | None = None) -> list[VerbatimFinding]:
    """Lockstep schema/data walk over the marked string fields: every
    value (an array's entries, or the field's own value) must be a
    normalized substring of the source and must not open with a
    sibling's heading line (label + colon). ``min_ratio`` overrides
    each field's own marker ratio (a per-source relaxation); without
    it the marker declares the tolerance."""
    findings: list[VerbatimFinding] = []

    def walk(scm: dict, d, where: str, parent_labels: frozenset):
        if not isinstance(d, dict):
            return
        props = scm.get('properties') or {}
        scope_labels = frozenset(
            lbl for lbl in (heading_label(p) for p in props.values()) if lbl)
        for name, prop in props.items():
            value = d.get(name)
            if value is None:
                continue
            path = f'{where}.{name}' if where else name
            items_schema = prop.get('items') or {}
            if (ratio := marker_ratio(prop)) is not None:
                labels = parent_labels | scope_labels
                tolerance = ratio if min_ratio is None else min_ratio
                if isinstance(value, list) and items_schema.get('type') == 'string':
                    for i, item in enumerate(value):
                        if isinstance(item, str) and item.strip():
                            if f := check_item(item, f'{path}[{i}]', labels,
                                               source_norm, tolerance):
                                findings.append(f)
                elif prop.get('type') == 'string' and isinstance(value, str) \
                        and value.strip():
                    if f := check_item(value, path, labels, source_norm, tolerance):
                        findings.append(f)
            if isinstance(value, list) and 'object' in type_set(items_schema):
                for i, entry in enumerate(value):
                    walk(items_schema, entry, f'{path}[{i}]', scope_labels)
            if isinstance(value, dict) and 'object' in type_set(prop):
                walk(prop, value, path, scope_labels)

    walk(schema, data, '', frozenset())
    return findings


@dataclass(frozen=True)
class VerbatimIssue:
    """Issue-protocol carrier for a verbatim breach; repairable — the
    correction loop owns it (the specialist holds the document in its
    transcript, so a patch round can re-copy)."""

    path: str
    message: str
    code: str = 'verbatim'
    expected: object = 'verbatim copy from the document'
    got: object = ''
    report_only: bool = False


def verbatim_issues(schema: dict, data: dict,
                    source_norm: str) -> list[VerbatimIssue]:
    """Pipeline form of find_violations: breach findings as issues whose
    messages instruct the repair."""
    issues = []
    for f in find_violations(schema, data, source_norm):
        if f.code == 'heading':
            message = (f'opens with heading {f.label!r} — that labelled block '
                       'belongs to its own field; replace this item with the '
                       'remaining text of the block')
        else:
            message = (f'not a verbatim copy of the document text '
                       f'({f.coverage:.0%} coverage) — re-copy it exactly, '
                       'no omissions or summaries')
        issues.append(VerbatimIssue(f.path, message, got=f.item))
    return issues


def verbatim_budgets(schema: dict, units, routing) -> dict:
    """Chunk-derived output budgets for units carrying marked fields —
    the routed material's non-blank char count, per item list for array
    items (gaps inherit the previous item), a lone number for
    whole-shaped units. A verbatim echo costs what its source costs, so
    the router's declared estimate — and any host pin — is discarded
    for these paths; the estimate errs high — headings and JSON syntax
    are not echoed — the safe direction for batching. Which units are
    marked is read from the declared schema, not the decomposed units:
    a lift strips the array property's description, where the marker
    lives. {} when nothing is marked."""
    if not (marked := _marked_units(schema, units)):
        return {}
    per_item: dict[str, dict[int, int]] = {}
    whole: dict[str, int] = {}
    for g in routing.groups:
        if (path := g.unit.path) not in marked:
            continue
        chars = content_len(g.text)
        if g.item is None:
            whole[path] = whole.get(path, 0) + chars
        else:
            row = per_item.setdefault(path, {})
            row[g.item] = row.get(g.item, 0) + chars
    out = {}
    for path, vals in per_item.items():
        row, last = [], 0
        for i in range(max(vals) + 1):
            last = vals.get(i, last)
            row.append(last)
        out[path] = row
    out.update(whole)
    return out


def _marked_units(schema: dict, units) -> set[str]:
    """Unit paths whose contract contains a marked field, or that are a
    marked field themselves (a lifted array)."""
    paths = verbatim_paths(schema)
    return {u.path for u in units
            if any(p == u.path or p.startswith(u.path + '.') for p in paths)}
