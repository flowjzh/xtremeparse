"""Deterministic merge of unit results back into schema-shaped data."""

from __future__ import annotations

import re

from xtremeparse.paths import resolve_list, split_index
from xtremeparse.units import MISC, values_key

_SCOPED = re.compile(r'\[(\d+)\](\..+)?$')  # the '[k]' or '[k].field'
# remainder after a counts key's unit path


def merge(values: dict, yields: dict = None) -> dict:
    """Assemble unit-path values into one nested dict, order-independent.
    ``$misc`` values scatter to top level; dotted paths nest; sibling
    object and array units at overlapping paths dict-merge; a lifted
    sub-array's ``parent[n].field`` values graft into the parent
    array's n-th instance; None values stay absent. ``yields`` — parent
    path → {lifted sibling: fields that yield to it} — is the
    ``x-override`` contract's merge half: an item whose sibling claim
    landed truthy loses those fields, the render contract's own
    resolution of the both-truthy quadrant (an empty claim means the
    item holds none of the sibling, so the fallback field keeps its
    content and renders)."""
    data, claims = {}, []
    for path, value in values.items():
        if value is None:
            continue
        if path == MISC:
            data.update({k: v for k, v in value.items() if v is not None})
            continue
        parts = path.split('.')
        node = data
        for part in parts[:-1]:
            node = _open(node, part)
        _graft(node, parts[-1], value)
        if len(parts) > 1 and (split := split_index(parts[-2])) \
                and (drops := (yields or {}).get(
                    '.'.join(parts[:-2] + [split[0]]), {}).get(parts[-1])):
            claims.append((parts[:-2], split[0], split[1], parts[-1], drops))
    for parents, key, i, field, drops in claims:
        node = data
        for part in parents:
            node = _open(node, part)
        entries = node.get(key) or []
        item = entries[i] if i < len(entries) else None
        if isinstance(item, dict) and item.get(field):
            for drop in drops:
                item.pop(drop, None)
    return data


def drop_phantom_items(data: dict, counts: dict, units: list) -> list:
    """Remove repeat items that carry no content — every leaf None or
    blank — and re-point ``counts`` (both mutated) at the shortened
    arrays, so a declared instance that came back empty reads one
    instance fewer, not a shortfall. The rule is unconditional on how
    the empty item arose: an all-empty entry is never worth rendering,
    and ``trace.drops`` records what left. Returns ``[{'path', 'index'}]``
    — ``index`` the item's position before the drop, later siblings
    re-indexing up one. Units process in decompose order: a parent array
    drops before its lifted sub-arrays read the compacted indexing."""
    drops = []
    for unit in units:
        if unit.kind != 'array':
            continue
        if unit.parent is None:
            drops += _drop(resolve_list(data, unit.path), unit.path, counts)
            continue
        for i, parent in enumerate(resolve_list(data, unit.parent)):
            if isinstance(parent, dict):
                drops += _drop(resolve_list(parent, unit.field),
                               values_key(unit, i), counts)
    return drops


def _drop(entries: list, path: str, counts: dict) -> list:
    phantom = [i for i, e in enumerate(entries) if _phantom(e)]
    if not phantom:
        return []
    for i in reversed(phantom):
        del entries[i]
    if path in counts:
        counts[path] = len(entries)
    _rekey_scopes(counts, path, phantom)
    return [{'path': path, 'index': i} for i in phantom]


def _rekey_scopes(counts: dict, path: str, removed: list) -> None:
    """Counts scoped ``path[k]…`` under a parent array that dropped
    instances: the dropped scopes go, the later ones shift down. Rebuilt
    aside and merged back — a rekey in place would let a shifted entry
    overwrite a not-yet-shifted one."""
    moved = {}
    for key in list(counts):
        if not key.startswith(f'{path}[') \
                or not (m := _SCOPED.match(key[len(path):])):
            continue
        i, suffix = int(m[1]), m[2]
        value = counts.pop(key)
        if i not in removed:
            moved[f'{path}[{i - sum(r < i for r in removed)}]{suffix}'] = value
    counts.update(moved)


def _phantom(value) -> bool:
    """Every leaf carries nothing — None, blank, an all-empty structure.
    A bool or number is content: False is a legitimate checkbox state."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(_phantom(v) for v in value.values())
    if isinstance(value, list):
        return all(_phantom(v) for v in value)
    return False


def _open(node: dict, part: str) -> dict:
    """The container ``part`` addresses, created on demand — a dict key
    with an optional ``[n]`` list hop (a lifted sub-array's parent
    instance), padded to the index."""
    if (split := split_index(part)) is None:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        return node[part]
    entries = _slot(node, *split)
    if not isinstance(entries[split[1]], dict):
        entries[split[1]] = {}
    return entries[split[1]]


def _graft(node: dict, leaf: str, value) -> None:
    """One value into its leaf slot, fusing with whatever another
    unit's claim already put there."""
    if (split := split_index(leaf)) is not None:
        entries = _slot(node, *split)
        entries[split[1]] = _fuse(entries[split[1]], value)
        return
    node[leaf] = _fuse(node.get(leaf), value)


def _slot(node: dict, key: str, i: int) -> list:
    """The list ``key`` addresses, created and padded to index ``i``
    on demand — the one home of the list-pad semantics."""
    if not isinstance(node.get(key), list):
        node[key] = []
    entries = node[key]
    entries.extend([None] * (i + 1 - len(entries)))
    return entries


def _fuse(old, new):
    """Two units' claims on one slot: objects merge field-wise — an
    empty list in the newer claim keeps the older field (a narrow
    slice's blank never blanks what the wider read copied verbatim);
    lists merge element-wise (a parent unit's items and a lifted
    sub-array's grafts meet here), either side absent yields."""
    if isinstance(old, dict) and isinstance(new, dict):
        return {**old, **{k: _fuse(old.get(k), v) for k, v in new.items()
                          if not (v == [] and isinstance(old.get(k), list)
                                  and old[k])}}
    if isinstance(old, list) and isinstance(new, list):
        pad = old + [None] * (len(new) - len(old))
        return [_fuse(o, n) for o, n in zip(pad, new)]
    return new
