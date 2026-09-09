"""RFC 6902 JSON Patch: correction rounds diff instead of re-emitting.

A correction round shows the specialist its previous result and asks
for a patch — the minimal set of operations that fixes the routed
issues. Untouched entries cannot collapse in a rewrite (the measured
failure: a retry asked for one missing entry re-decoded the whole batch
and returned none), and the decode shrinks to the size of the fix.

The apply is tolerant in exactly two ways, both measured: a pointer may
be rooted at the unit path (``/project_experiences/81`` for a bare
``/81``) — the model reaches for the document root — and a reply that
is not a patch at all reads as the full corrected value (the pre-patch
semantics), so a draw that ignores the protocol degrades gracefully.
An empty operation list is an empty patch — the model's no-fix
declaration (the requested value does not exist in the material), a
no-op on apply, never a wipe.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

_OPS = ('add', 'remove', 'replace', 'move', 'copy', 'test')
_MISSING = object()

PATCH_ARRAY = {
    'type': 'array',
    'items': {
        'type': 'object',
        'properties': {'op': {'type': 'string'}, 'path': {'type': 'string'},
                       'value': {}},
        'required': ['op', 'path'],
    },
    'description': 'RFC 6902 JSON Patch operations',
}
"""The wire marker of a patch round — recognized structurally by
:func:`is_patch_round`, skipped by :func:`value_branch`. The protocol's
prompt prose has two halves kept consistent by convention:
``corrections.PATCH_HOWTO`` rides the feedback tail, and each host
adapter's output block (e.g. docspectra's) states the wire shape; the
no-fix clause they both teach is ``PATCH_NO_FIX``, shared verbatim."""

PATCH_NO_FIX = ('an empty array when a requested value does not exist in '
                'the material — never fabricate one or re-emit it as null')
"""The no-fix clause of the patch protocol's prompt prose — one home,
interpolated verbatim by every site that teaches the protocol
(``corrections.PATCH_HOWTO``, each host adapter's output block)."""


def is_patch_round(schema: dict) -> bool:
    """Whether this reply schema is the patch-or-value anyOf of a
    correction round — structural (the op+path requirement), so it
    survives serialization of the schema."""
    return any(_is_patch_branch(b) for b in schema.get('anyOf') or ())


def value_branch(schema: dict) -> dict:
    """The full-value branch of a patch-or-value anyOf — the shape a
    non-patch reply takes. Other schemas pass through unchanged."""
    for b in schema.get('anyOf') or ():
        if not _is_patch_branch(b):
            return b
    return schema


def _is_patch_branch(branch: dict) -> bool:
    """A patch branch is an array of operations — the op+path
    requirement lives on the item schema; a value branch of any shape
    (object, or an array of item objects without them) never has it."""
    items = branch.get('items') or branch
    return {'op', 'path'} <= set(items.get('required') or ())


def is_patch(data: Any) -> bool:
    """Whether a specialist reply is a patch rather than the value
    itself: a list whose every element names an op and a pointer. An
    empty list is a patch too — no operations, no change."""
    return (isinstance(data, list)
            and all(isinstance(o, dict) and isinstance(o.get('op'), str)
                    and isinstance(o.get('path'), str) for o in data))


def is_no_fix(data: Any) -> bool:
    """Whether a patch reply declares no fix — the empty patch, the
    taught reply (``PATCH_NO_FIX``) for a value the material does not
    carry. The correction loop honors it: the declared paths are never
    asked again."""
    return not data and is_patch(data)


def apply_patch(prev: Any, ops: list, root: str = None) -> tuple:
    """Apply a JSON Patch to ``prev``, returning ``(patched, error)`` —
    error None on success, a human-readable reason otherwise (the
    previous result is then kept and the next round asks for the full
    value). ``root`` is the unit path a pointer may be rooted at."""
    data = copy.deepcopy(prev)
    for i, op in enumerate(ops):
        kind, pointer = op.get('op'), op.get('path')
        if kind not in _OPS:
            return None, f'op {i}: unknown op {kind!r}'
        parts = _tokens(str(pointer or ''), root)
        if parts is None:
            return None, f'op {i}: bad pointer {pointer!r}'
        if not parts:  # the whole document
            if kind not in ('add', 'replace'):
                return None, f'op {i}: {kind} on the root is not supported'
            data = copy.deepcopy(op.get('value'))
            continue
        if kind == 'move' or kind == 'copy':
            src, skey = _walk(data, _tokens(str(op.get('from') or ''), root)
                              or [])
            if src is None or (here := _get(src, skey)) is _MISSING:
                return None, f'op {i}: from-path does not resolve'
            if kind == 'move' and (err := _drop(src, skey, i)) is not None:
                return None, err
            op = {**op, 'value': copy.deepcopy(here)}
        parent, key = _walk(data, parts)
        if parent is None:
            return None, f'op {i}: path {pointer!r} does not resolve'
        if kind == 'test':
            if _get(parent, key) != op.get('value'):
                return None, f'op {i}: test failed'
            continue
        if kind == 'remove':
            if (err := _drop(parent, key, i)) is not None:
                return None, err
            continue
        if kind == 'replace' and _get(parent, key) is _MISSING:
            return None, f'op {i}: path {pointer!r} does not exist'
        if (err := _put(parent, key, op.get('value'), kind, i)) is not None:
            return None, err
    return data, None


def _tokens(pointer: str, root: Optional[str]) -> Optional[list]:
    """Pointer → unescaped tokens, or None when malformed. A first
    token naming ``root`` (with any ``[n]`` index tail the model may
    append) is stripped — the model roots pointers at the document."""
    if not pointer.startswith('/'):
        return None
    parts = [p.replace('~1', '/').replace('~0', '~')
             for p in pointer.split('/')[1:]]
    if root and parts and parts[0].split('[')[0] == root:
        parts = parts[1:]
    return parts


def _walk(data: Any, parts: list):
    """The container holding the final token, and that token."""
    for tok in parts[:-1]:
        if (step := _get(data, tok)) is _MISSING:
            return None, None
        data = step
    return data, parts[-1] if parts else None


def _get(container: Any, key: str):
    if isinstance(container, list):
        if not key.isdigit() or int(key) >= len(container):
            return _MISSING
        return container[int(key)]
    if isinstance(container, dict):
        return container.get(key, _MISSING)
    return _MISSING


def _put(container: Any, key: str, value: Any, kind: str, i: int) -> Optional[str]:
    if isinstance(container, list):
        if key == '-':
            container.append(value)
            return None
        if not key.isdigit():
            return f'op {i}: array index {key!r} is not a number'
        at = int(key)
        if kind == 'add' and at <= len(container):
            container.insert(at, value)
            return None
        if at < len(container):
            container[at] = value
            return None
        return f'op {i}: array index {at} out of range'
    if isinstance(container, dict):
        container[key] = value
        return None
    return f'op {i}: cannot address into {type(container).__name__}'


def _drop(container: Any, key: str, i: int) -> Optional[str]:
    if isinstance(container, list):
        if not key.isdigit() or int(key) >= len(container):
            return f'op {i}: array index {key!r} out of range'
        container.pop(int(key))
        return None
    if isinstance(container, dict) and key in container:
        del container[key]
        return None
    return f'op {i}: path does not exist'
