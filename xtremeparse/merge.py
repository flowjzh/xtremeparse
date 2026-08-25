"""Deterministic merge of unit results back into schema-shaped data."""

from __future__ import annotations

from xtremeparse.units import MISC


def merge(values: dict) -> dict:
    """Assemble unit-path values into one nested dict, order-independent.
    ``$misc`` values scatter to top level; dotted paths nest; sibling
    object and array units at overlapping paths dict-merge; None values
    stay absent."""
    data = {}
    for path, value in values.items():
        if value is None:
            continue
        if path == MISC:
            data.update({k: v for k, v in value.items() if v is not None})
            continue
        *parents, leaf = path.split('.')
        node = data
        for p in parents:
            node = node.setdefault(p, {})
        if isinstance(node.get(leaf), dict) and isinstance(value, dict):
            node[leaf] = {**node[leaf], **value}  # sibling units overlap: keep both
        else:
            node[leaf] = value
    return data
