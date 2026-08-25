"""Plain-data reads shared by the pipeline and hosts.

Dotted-path lookup is pipeline infrastructure (corrections reads its
merged arrays through it) and the public read behind host eval wiring
— one implementation, no forks.
"""

from __future__ import annotations

import re

_INDEX = re.compile(r'(\w+)\[(\d+)\]')


def resolve(data: dict, path: str):
    """Dotted-path lookup into nested dicts, with ``key[n]`` hops into
    lists (the bracket notation corrections already uses for item
    issues); None when any hop is absent or crosses a non-dict."""
    node = data
    for part in path.split('.'):
        if not isinstance(node, dict):
            return None
        if '[' in part and (m := _INDEX.fullmatch(part)):
            node = node.get(m[1])
            index = int(m[2])
            if not isinstance(node, list) or index >= len(node):
                return None
            node = node[index]
        else:
            node = node.get(part)
    return node


def resolve_list(data: dict, path: str) -> list:
    """The list at ``path``, [] when absent or not a list — the
    array-read behind count checks."""
    value = resolve(data, path)
    return value if isinstance(value, list) else []
