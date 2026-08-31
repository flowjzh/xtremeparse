"""Deterministic schema decomposition into extraction units.

JSON Schema in (top level + one nested array level), semantic cards
composed from the ``description``s of each sub-tree. Bare schemas
without descriptions degrade to name-only cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from xtremeparse.contracts import JSONSchema

MISC = '$misc'  # sentinel path for the top-level scalar group; a real
# object/array property named '$misc' would collide


@dataclass(frozen=True)
class Unit:
    """One concurrent extraction target. ``sub_schema`` is the unit's own
    structured-output contract: object schema for object units, the item
    schema for array units (per-item extraction; the executor wraps it
    when an array runs whole). It shares leaf dicts with the caller's
    schema — consumers must wrap (``{**sub_schema, ...}``), never mutate."""

    path: str
    kind: Literal['object', 'array', 'scalar']
    sub_schema: JSONSchema
    card: str

    @property
    def header(self) -> str:
        """The card's first line — ``[path | kind] description``, the
        unit's own semantics without its field list."""
        return self.card.splitlines()[0]


def type_set(prop: JSONSchema) -> set:
    """The property's JSON type names. docxcast marks an optional
    section with a type list ('object' / 'null'); a plain property
    carries one name."""
    t = prop.get('type')
    return set(t) if isinstance(t, list) else {t}


def value_branches(prop: JSONSchema) -> list:
    """The property's value subschemas. docxcast marks an optional
    field as anyOf over the value and an empty-string alternative;
    plain properties (and type-listed sections) carry the value at
    the top level."""
    if isinstance(prop.get('anyOf'), list):
        return prop['anyOf']
    return [prop]


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
            units.append(_array_unit(name, prop))
        else:
            misc[name] = prop
    if misc:
        units.append(Unit(MISC, 'scalar', {'type': 'object', 'properties': misc},
                          f'[{MISC} | scalar] {_fields_card(misc)}'))
    return units


def _object_units(path: str, prop: JSONSchema) -> list[Unit]:
    props = prop.get('properties') or {}
    arrays = {k for k, v in props.items() if 'array' in type_set(v)}
    rest = {k: v for k, v in props.items() if k not in arrays}
    units = []
    if rest:
        sub = {**prop, 'properties': rest}
        if 'required' in prop:
            sub['required'] = [k for k in prop['required'] if k in rest]
        header = f'[{path} | object] {prop.get("description", "")}'.strip()
        units.append(Unit(path, 'object', sub, f'{header}\n  {_fields_card(rest)}'))
    return units + [_array_unit(f'{path}.{k}', props[k]) for k in arrays]


def _array_unit(path: str, prop: JSONSchema) -> Unit:
    items = prop.get('items') or {}
    header = f'[{path} | array] {prop.get("description", "")}'.strip()
    if fields := _fields_card(items.get('properties') or {}):
        header = f'{header}\n  per item: {fields}'
    return Unit(path, 'array', items, header)


def _fields_card(props: dict) -> str:
    return ' · '.join(_field_card(name, prop) for name, prop in props.items())


def _field_card(name: str, prop: JSONSchema) -> str:
    desc = prop.get('description', '')
    if annot := prop.get('format') or _nonstring(prop.get('type')):
        desc = f'{desc}({annot})'
    return f'{name} {desc}'.strip()


def _nonstring(t) -> str:
    return t if t and t != 'string' else ''
