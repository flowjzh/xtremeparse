"""Deterministic schema decomposition into extraction units.

JSON Schema in (top level + one nested array level), semantic cards
composed from the ``description``s of each sub-tree. Bare schemas
without descriptions degrade to name-only cards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from xtremeparse.contracts import JSONSchema
from xtremeparse.schema import type_set

MISC = '$misc'  # sentinel path for the top-level scalar group; a real
# object/array property named '$misc' would collide

OVERRIDE_KEY = 'x-override'  # the schema extension keyword naming the
# sibling a field yields to at render (docxcast's @override directive);
# a call whose items all read that sibling on their own drops the field


@dataclass(frozen=True)
class Unit:
    """One concurrent extraction target. ``sub_schema`` is the unit's own
    structured-output contract: object schema for object units, the item
    schema for array units (per-item extraction; the executor wraps it
    when an array runs whole). It shares leaf dicts with the caller's
    schema — consumers must wrap (``{**sub_schema, ...}``), never mutate.
    ``parent`` is set only on a lifted array-under-array: the path of the
    array unit whose instances the sub-array hangs under (its ``field``
    names the parent's property it fills); the router addresses such a
    unit through the parent (``c.0.d.0``)."""

    path: str
    kind: Literal['object', 'array', 'scalar']
    sub_schema: JSONSchema
    card: str
    parent: str | None = None

    @property
    def header(self) -> str:
        """The card's first line — ``[path | kind] description``, the
        unit's own semantics without its field list; every later card
        line is a two-space-indented continuation, schema text arrives
        pre-reflowed (``_card_text``)."""
        return self.card.splitlines()[0]

    @property
    def field(self) -> str:
        """The parent's property this unit fills — the path's last
        segment, which for a nested unit is the parent item's field."""
        return self.path.rsplit('.', 1)[-1]


def decompose(schema: JSONSchema) -> list[Unit]:
    """Split a JSON Schema into object units, array units, and one
    $misc scalar group. Same schema always yields the same units.
    Nullable sections (docxcast emits ``type: ["object", "null"]``)
    decompose like plain objects."""
    units, misc = [], {}
    for name, prop in (schema.get('properties') or {}).items():
        if 'object' in (types := type_set(prop)):
            units.extend(_object_units(name, prop))
        elif 'array' in types:
            units.extend(_array_units(name, prop))
        else:
            misc[name] = prop
    if misc:
        units.append(Unit(MISC, 'scalar', {'type': 'object', 'properties': misc},
                          f'[{MISC} | scalar] {_fields_card(misc)}'))
    return units


def _card_text(text: str) -> str:
    # schema text with a line break rides as a two-space-indented
    # continuation line so the card keeps its [path | kind] first line
    # and owns its whole block
    return text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\n  ')


def _card_header(path: str, kind: str, prop: JSONSchema) -> str:
    return f'[{path} | {kind}] {_card_text(prop.get("description", ""))}'.strip()


def _without(schema: JSONSchema, names: set[str]) -> JSONSchema:
    """``schema`` re-wrapped without the named properties, its
    ``required`` list kept in sync — the one home of the
    properties-minus rule the lift and the call-time strip share."""
    props = schema.get('properties') or {}
    rest = {k: v for k, v in props.items() if k not in names}
    out = {**schema, 'properties': rest}
    if 'required' in schema:
        out['required'] = [k for k in schema['required'] if k in rest]
    return out


def _object_units(path: str, prop: JSONSchema) -> list[Unit]:
    props = prop.get('properties') or {}
    arrays = {k for k, v in props.items() if 'array' in type_set(v)}
    rest = {k: v for k, v in props.items() if k not in arrays}
    units = []
    if rest:
        sub = _without(prop, arrays)
        header = _card_header(path, 'object', prop)
        units.append(Unit(path, 'object', sub, f'{header}\n  {_fields_card(rest)}'))
    return units + [u for k in arrays for u in _array_units(f'{path}.{k}', props[k])]


def _array_units(path: str, prop: JSONSchema) -> list[Unit]:
    """The array unit for ``prop``, plus one unit per array nested
    directly in its items: an array-under-array lifts one level into a
    unit of its own — ``parent`` set, the field stripped from the
    parent's item schema — so the router can address each sub-entry
    through its parent instance and the parent's calls never extract it
    twice. Deeper nesting stays where it is: the sub-unit's own calls
    extract it whole."""
    items = prop.get('items') or {}
    props = items.get('properties') or {}
    nested = {k: v for k, v in props.items() if 'array' in type_set(v)}
    if not nested:
        return [_array_unit(path, prop)]
    rest = {k: v for k, v in props.items() if k not in nested}
    sub = _without(items, nested)
    unit = Unit(path, 'array', sub, _array_card(path, prop, sub))
    return [unit, *(_array_unit(f'{path}.{k}', v, parent=path)
                    for k, v in nested.items())]


def _array_unit(path: str, prop: JSONSchema, parent: str = None) -> Unit:
    items = prop.get('items') or {}
    return Unit(path, 'array', items, _array_card(path, prop, items),
                parent=parent)


def _array_card(path: str, prop: JSONSchema, items: JSONSchema) -> str:
    header = _card_header(path, 'array', prop)
    if fields := _fields_card(items.get('properties') or {}):
        header = f'{header}\n  per item: {fields}'
    return header


def _fields_card(props: dict) -> str:
    return ' · '.join(_field_card(name, prop) for name, prop in props.items())


def stripped_unit(unit: Unit, names: set[str]) -> Unit:
    """The call-time variant of a unit whose overridden fields yield:
    ``sub_schema`` re-wrapped without the named properties (the leaf
    dicts stay shared — consumers wrap, never mutate), the card's field
    listing re-rendered to match. The original card ends in exactly
    ``_fields_card`` of the original properties, so the listing is cut
    by that string and the remainder — header plus its ``per item:``
    lead-in — is kept verbatim."""
    props = unit.sub_schema.get('properties') or {}
    head = unit.card.rpartition(_fields_card(props))[0]
    sub = _without(unit.sub_schema, names)
    card = head + fields if (fields := _fields_card(sub.get('properties')
                                                     or {})) else head.rstrip()
    return replace(unit, sub_schema=sub, card=card)


def unit_yields(unit: Unit, lifted: set[str]) -> dict[str, frozenset[str]]:
    """``{lifted sibling: fields that yield to it}`` over one array
    unit's item schema — the merge-plane half of the ``x-override``
    contract: the declaring fields are the sibling's fallback rendering,
    so once that sibling's own reads land on an item they must not
    survive beside them. ``lifted`` names the item's array properties
    the decomposition raised into units of their own — the lift strips
    them from the item schema, so the target is invisible without it."""
    out = {}
    for k, v in (unit.sub_schema.get('properties') or {}).items():
        if (sib := v.get(OVERRIDE_KEY)) in lifted:
            out[sib] = out.get(sib, frozenset()) | {k}
    return out


def _field_card(name: str, prop: JSONSchema) -> str:
    desc = _card_text(prop.get('description', ''))
    if annot := prop.get('format') or _nonstring(prop.get('type')):
        desc = f'{desc}({annot})'
    return f'{name} {desc}'.strip()


def _nonstring(t) -> str:
    return t if t and t != 'string' else ''
