"""One reader for the schema dialect the pipeline and its hosts share.

docxcast emits a small, fixed vocabulary: type lists for nullable
sections, ``anyOf`` for optional scalars (the value plus an empty-string
sentinel), ``enum`` for choices, ``format`` for value shapes, and
``x-verbatim`` for the copy contract. Everything that inspects a schema
reads those shapes here instead of re-deriving them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from xtremeparse.contracts import JSONSchema


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


def options_of(prop: JSONSchema) -> list[str]:
    """The literal alternatives a value may take: enum values, plus each
    anyOf branch's own options and format, recursively. The empty string
    an optional field carries as its unfilled sentinel is not an option
    a reader should look for."""
    out = [s for v in prop.get('enum') or () if (s := str(v))]
    for branch in prop.get('anyOf') or ():
        out += options_of(branch)
        if fmt := format_of(branch):
            out.append(fmt)
    return list(dict.fromkeys(out))


def format_of(prop: JSONSchema) -> str | None:
    """The field's declared value format (``date-year``, ``date-month``,
    …), None when it declares none."""
    return prop.get('format')


def walk_fields(node: JSONSchema,
                prefix: str = '') -> Iterator[tuple[str, JSONSchema]]:
    """Yield ``(dotted_path, prop)`` for every field under ``node`` in
    pre-order: properties, then array items, then anyOf branches. A
    branch is an alternate shape of the field it sits on, so its fields
    extend the same path — mutually exclusive branches may yield equal
    paths, and callers must treat each yield as one field occurrence."""
    for name, prop in (node.get('properties') or {}).items():
        path = f'{prefix}.{name}' if prefix else name
        yield path, prop
        yield from walk_fields(prop, path)
        yield from walk_fields(prop.get('items') or {}, path)
        for branch in prop.get('anyOf') or ():
            yield from walk_fields(branch, path)


def heading_label(prop: JSONSchema) -> str | None:
    """A string field's heading label — its description up to the first
    colon; whole descriptions make hopeless prefixes."""
    if prop.get('type') != 'string':
        return None
    desc = prop.get('description') or ''
    return re.split('[：:]', desc, maxsplit=1)[0].strip()
