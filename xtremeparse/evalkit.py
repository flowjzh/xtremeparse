"""Eval support: plain-data reads over the orchestration trace.

The lib keeps the reads its shipped evaluators (the ``evals`` extra)
need: the executor's calling conventions (per_item_budgets) and the
router's contract (router_overlap). Hosts own their general eval
tooling — digest projections, anchor checks, assertion DSLs are each
host's selection.
"""

from __future__ import annotations


def per_item_budgets(groups: list) -> dict:
    """Trace groups → the per-path budget map ``item_chars`` reads: one
    number or the item-index-ordered list."""
    whole, by_item = {}, {}
    for g in groups:
        if not g.get('budget'):
            continue
        items = g.get('batch') or ([g['item']] if g['item'] is not None else None)
        if items is None:  # whole-array call: its list is document-ordered
            whole[g['unit']] = g['budget']
            continue
        values = g['budget'] if isinstance(g['budget'], list) else None
        for k, i in enumerate(items):
            by_item.setdefault(g['unit'], {})[i] = values[k] if values else g['budget']
    return whole | {unit: [items[i] for i in sorted(items)]
                    for unit, items in by_item.items()}


def router_overlap(router, groups: list) -> tuple:
    """(overlap, items_lost) between the router's assignments and the
    executed calls; only per-item executions are held to item presence
    (whole-strategy coalescing is legitimate)."""
    assignments = router['assignments'] if router else []
    raw = sum(len(a['chunks']) for a in assignments)
    present = {(g['unit'], i) for g in groups
               for i in (g.get('batch') or [g['item']])}
    per_item_units = {g['unit'] for g in groups if g['strategy'] == 'per-item'}
    declared = {(a['unit'], a.get('item')) for a in assignments
                if a['unit'] in per_item_units}
    return (raw - sum(len(g['chunk_ids']) for g in groups),
            len(declared - present))
