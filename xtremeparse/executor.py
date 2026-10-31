"""Executor: fan specialists out under the scheduler, then merge.

Object and scalar units coalesce their routed groups into one specialist
call; per-item arrays coalesce duplicate rows of the same item. Array
units decide strategy after routing, from the actual routed volume:
whole-array when it fits the threshold, per-item fan-out when it does
not — a per-path ``unit_strategy`` override encodes prior knowledge.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, Optional

from xtremeparse.contracts import AgentRunner, AgentResult, BATCH_BUDGET_CAP, JSONSchema
from xtremeparse.evalkit import pair_at
from xtremeparse.merge import merge
from xtremeparse.prompting import (SPECIALIST_INSTRUCTIONS, WHOLE_ARRAY_ADDENDUM,
                                   estimate_tokens, prompt_schema)
from xtremeparse.router import Routing, items_of
from xtremeparse.scheduling import TaskScheduler
from xtremeparse.units import Unit, stripped_unit, unit_yields

Strategy = Literal['whole', 'per-item', 'ranged']


@dataclass
class Call:
    """One specialist call as dispatched (trace material; ``result``
    carries the AgentResult so the correction loop can pass back history)."""

    unit: Unit
    item: Optional[int]
    strategy: Optional[Strategy]  # None for non-array units
    chunk_ids: list
    scope: str
    result: Optional[AgentResult] = None
    budget: Optional[int | list] = None  # the call's arranged estimate: one
    # number, or the per-item list a batch/whole-array call carries
    batch: tuple = ()  # item indexes when small items share this call
    parent: Optional[int] = None  # the parent instance a lifted
    # sub-array's call extracts under (None for top-level units)

    @property
    def array_shaped(self) -> bool:
        """The call's result schema is an array — whole runs and budget
        batches. One formula, shared by every dispatch site: a retry
        must re-open the shape the call was first run under."""
        return self.strategy == 'whole' or bool(self.batch)

    @property
    def value_shape(self) -> JSONSchema:
        """The result schema for the unit's own value — the fallback
        branch a patch round's anyOf offers beside the patch array."""
        return ({'type': 'array', 'items': self.unit.sub_schema}
                if self.array_shaped else self.unit.sub_schema)

    @property
    def slots(self) -> Optional[int]:
        """Entries the call owes: the batch's item count, one for a
        per-item single, None when the owed count is the unit's declared
        whole (whole-array and object calls — compare against that)."""
        if self.batch:
            return len(self.batch)
        return 1 if self.item is not None else None

    @property
    def produced(self) -> int:
        """Entries the raw reply returned — the produced side of
        ``slots``: a list's length, or one entry (a per-item single's
        own object, or an array-shaped reply that came back unwrapped).
        The merge collapses exact repeats in an over-returning ranged
        reply, so ``produced`` can exceed the entries it contributes;
        len() on a single's entry dict counts keys, not instances, and
        would read every healthy single as an over-full batch: phantom
        shortfalls then patch healthy entries into junk-keyed dicts."""
        if self.result is None or self.result.data is None:
            return 0
        data = self.result.data
        return len(data) if self.array_shaped and isinstance(data, list) else 1

    @property
    def value_path(self) -> str:
        """The merged-data path the call's result fills — the unit path,
        or the parent instance's field a lifted sub-array fills."""
        return values_key(self.unit, self.parent)


def values_key(unit: Unit, parent: Optional[int]) -> str:
    """The one spelling of a call's merged-data address: the unit path,
    or ``parent[n].field`` for a lifted sub-array under one parent
    instance (values_from_calls keys by it, corrections routes by it,
    the extractor flattens declared counts into it)."""
    if parent is None:
        return unit.path
    return f'{unit.parent}[{parent}].{unit.field}'


@dataclass
class Execution:
    values: dict
    calls: list
    yields: dict | None = None  # parent path -> {lifted sibling: fields
    # that yield to it} — the x-override map every merge of these values
    # needs, so a truthy sibling claim blanks the overridden field


async def execute(runner: AgentRunner, routing: Routing, *, payload: str,
                  scheduler: TaskScheduler, unit_strategy: dict = None,
                  budgets: dict = None,
                  specialist_instructions: str = None,
                  batch_cap: int = BATCH_BUDGET_CAP) -> Execution:
    """Run specialist calls concurrently and collect values keyed by unit
    path. ``payload`` must be the byte-identical content prefix every
    specialist of one extraction shares — the document text; each
    call's own partial schema rides per call beside its unit card, so
    the full document schema's bytes are never re-sent here."""
    calls, tasks = [], []
    yields: dict = {}
    lifted_fields: dict[str, set[str]] = {}
    covered: dict[str, set[int]] = {}
    for g in routing.groups:
        if g.unit.parent:
            lifted_fields.setdefault(g.unit.parent, set()).add(g.unit.field)
            covered.setdefault(g.unit.path, set()).add(g.parent)
    for unit, groups in _by_unit(routing.groups):
        unit_yields_map = unit_yields(unit, lifted_fields.get(unit.path) or ())
        if unit_yields_map:
            yields[unit.path] = unit_yields_map
        # a lifted sub-array fans out per parent instance: each parent's
        # sub-entries are one array's worth of material, strategy and
        # batching scoped to it — one parent's whole never carries
        # another's entries. Top-level units partition to one None-keyed
        # part, the same loop
        for parent, rows in _by_parent(groups):
            strategy = _strategy(unit, rows, unit_strategy or {}) \
                if unit.kind == 'array' else None
            # specs carry their own truthful strategy label: the model's
            # ranged blocks (one call per ranged run, its instances one
            # array-shaped answer), the executor's budget batches, and
            # whole-array calls are three distinct shapes
            specs = [('ranged', None, g.text, g.chunk_ids,
                      tuple(items_of(g.items)),
                      _block_budget(budgets, unit, g.items))
                     for g in rows if g.items]
            plain = [g for g in rows if not g.items]
            if plain:
                if strategy == 'per-item':
                    specs += [('per-item', *s) for s in
                              _batched(unit, _by_item(plain), budgets,
                                       batch_cap)]
                else:
                    # co-chunked items (one run holding several instances) share
                    # their material — extract it once, not once per item
                    unique = {tuple(g.chunk_ids): g for g in plain}.values()
                    specs += [(strategy, None,
                               '\n\n'.join(g.text for g in unique),
                               [i for g in unique for i in g.chunk_ids], None,
                               _arranged(budgets, unit, None))]
            for strat, item, scope, ids, batch, budget in specs:
                call = Call(_unit_for_call(unit, rows, unit_yields_map,
                                           covered, item, batch),
                            item, strat, ids, scope, budget=budget,
                            batch=batch, parent=parent)
                calls.append(call)
                tasks.append(await dispatch_specialist(
                    runner, call, payload=payload, scheduler=scheduler,
                    specialist_instructions=specialist_instructions))
    for call, result in zip(calls, await asyncio.gather(*tasks)):
        call.result = result
    return Execution(values_from_calls(calls), calls, yields)


def _first_occurrences(items: list) -> list:
    """The first copy of each exact value in one reply — a ranged
    strategy over-returns by design, so a repeated entry is noise."""
    return list({json.dumps(v, sort_keys=True): v for v in items}.values())


def values_from_calls(calls: list) -> dict:
    """Per-unit values from call results; None data stays absent and
    per-item results concatenate in item-index order. A lifted
    sub-array's values key by the parent instance they fill — merge
    grafts the bracket form into the parent's items. An over-returning
    ranged reply (its run's chunk count, not its declared slots, bounds
    what the specialist can see) collapses exact-value repeats; the
    other strategies' extras stay visible to count arbitration."""
    values = {}
    for call in calls:
        data = call.result.data if call.result else None
        if data is None:
            continue
        path = call.value_path
        if call.batch:
            items = data if isinstance(data, list) else [data]
            if call.strategy == 'ranged' and len(items) > call.slots:
                items = _first_occurrences(items)
            values.setdefault(path, []).extend(items)
        elif call.strategy == 'per-item':
            values.setdefault(path, []).append(data)
        else:
            values[path] = data
    return values


def execution_data(execution: Execution) -> dict:
    """The execution's current merged data — every merged view of an
    execution applies its yields map, so a truthy sibling claim blanks
    the overridden field in every stage's view."""
    return merge(values_from_calls(execution.calls), execution.yields)


async def dispatch_specialist(runner: AgentRunner, call: Call, *, payload: str,
                              scheduler: TaskScheduler,
                              specialist_instructions: str = None,
                              history: list = None,
                              feedback: list = None,
                              schema: JSONSchema = None) -> asyncio.Task:
    """Schedule the one specialist invocation (executor and correction
    rounds share it, so the cache-frozen prompt layout stays identical)
    under the shared budget, seeded with the recipe's own token
    estimate. ``schema`` overrides ``call.value_shape`` — correction
    rounds pass the patch-or-value anyOf so the reply may be either a
    JSON Patch or the full corrected value. Returns the dispatched
    task."""
    async def invoke() -> AgentResult:
        reply_schema = prompt_schema(
            schema if schema is not None else call.value_shape)
        instructions = (specialist_instructions or SPECIALIST_INSTRUCTIONS).format(
            card=call.unit.card)
        if call.array_shaped:
            instructions += WHOLE_ARRAY_ADDENDUM
        # a history round already carries the card and scope in the
        # transcript — only the per-round feedback is fresh material
        return await runner.run(instructions='' if history else instructions,
                                result_schema=reply_schema, content=payload,
                                scope='' if history else call.scope,
                                history=history, feedback=feedback)
    return await scheduler.start_task(
        invoke(), estimated_tokens=estimate_tokens(payload, call.scope,
                                                   call.unit.card))


def _arranged(budgets: dict, unit: Unit, item: Optional[int]):
    """The budget a call carries: its own item's number from the
    arranged list (a lone number covers every item; a short list's last
    value covers items beyond it); a whole-array call keeps the list."""
    if (arranged := (budgets or {}).get(unit.path)) is None:
        return None
    values = arranged if isinstance(arranged, list) else [arranged]
    if item is None:
        return values[0] if len(values) == 1 else list(values)
    return pair_at(values, item)


def _block_budget(budgets: dict, unit: Unit, items: tuple):
    """The ranged block's arranged budget: the unit's per-item list cut
    to the block's own slice — one _arranged call per item, so the
    resolution rule (a lone number repeats; a short list's last value
    covers items beyond it) has one home. Budget presence is
    path-level, so None-ness is all-or-nothing across the block."""
    picked = [_arranged(budgets, unit, i) for i in range(items[0], items[1] + 1)]
    if picked[0] is None:
        return None
    return picked[0] if len(picked) == 1 else picked


def _batched(unit: Unit, item_rows, budgets: dict,
             batch_cap: int) -> list:
    """Per-item call specs, budget-batched: consecutive items share one
    array-shaped call while their ACCUMULATED budget fits the per-call
    capacity — the router's arrangement doubles as the scheduling
    signal, whatever the schema makes the unit. An item that alone
    exceeds the capacity (or carries no arrangement) keeps its own call,
    so N capacity-sized items always stay N parallel streams. Specs are
    ``(item, scope, ids, batch, budget)``; batches carry their item
    indexes and per-item budget list, singles their own number."""
    rows_by_item = item_rows
    arranged = {item: _arranged(budgets, unit, item) for item in rows_by_item}
    specs, run, run_total = [], [], 0

    def emit():
        nonlocal run, run_total
        if not run:
            return
        scope = '\n\n'.join(g.text for item in run for g in rows_by_item[item])
        ids = [i for item in run for g in rows_by_item[item] for i in g.chunk_ids]
        if len(run) == 1:
            specs.append((run[0], scope, ids, None, arranged[run[0]]))
        else:
            run_budgets = [arranged[i] for i in run]
            specs.append((None, scope, ids, tuple(run),
                          run_budgets if all(b is not None for b in run_budgets)
                          else None))
        run, run_total = [], 0

    for item in rows_by_item:
        budget = arranged[item]
        if budget is None or budget > batch_cap:
            emit()
            run = [item]
            emit()
            continue
        if run and run_total + budget > batch_cap:
            emit()
        run.append(item)
        run_total += budget
    emit()
    return specs


def _strategy(unit: Unit, groups: list, unit_strategy: dict) -> Strategy:
    """Decode time is output-bound, so separable items fan out — parallel
    small streams beat one long one even when the material is short.
    Whole is for what per-item scoping cannot split: co-chunked
    instances, single items."""
    if override := unit_strategy.get(unit.path):
        if override not in ('whole', 'per-item'):
            raise ValueError(f'unknown unit_strategy for {unit.path}: {override!r}')
        return override
    ids = [i for g in groups for i in g.chunk_ids]
    if len(ids) != len(set(ids)):
        return 'whole'  # co-chunked items: one call must split the instances
    return 'whole' if len({g.item for g in groups}) <= 1 else 'per-item'


def _unit_for_call(unit: Unit, rows: list, yields_entry: dict[str, frozenset[str]],
                   covered: dict[str, set[int]], item: Optional[int],
                   batch: tuple) -> Unit:
    """The call's unit: a field the schema marks ``x-override`` yields to
    the sibling it names — once the lift gives that sibling its own
    reads for every item this call extracts, the field is that item's
    fallback rendering of the same content and only re-extracts (and
    re-teaches) what the sibling owns, so it leaves the contract and the
    card alike. Calls whose coverage is partial keep the field — a whole
    run over unlifted items holds the field as that item's only
    carrier; a ranged block's span is its coverage, so it strips when
    the span is fully lifted."""
    items = (set(batch) if batch else {item} if item is not None
             else _routed_items(rows))
    if not (items and yields_entry):
        return unit
    drops = set()
    for sib, names in yields_entry.items():
        if items <= covered.get(f'{unit.path}.{sib}', ()):
            drops |= names
    return stripped_unit(unit, drops) if drops else unit


def _routed_items(rows: list) -> set:
    """The items a whole-shaped call extracts from this partition — its
    plain groups; a ranged block in the partition carries its own span
    as the batch, so its items are not this call's."""
    return {g.item for g in rows if not g.items}


def _by_unit(groups) -> list:
    by_path = {}
    for g in groups:
        by_path.setdefault(g.unit.path, []).append(g)
    return [(rows[0].unit, rows) for rows in by_path.values()]


def _by_parent(groups) -> list:
    """One lifted sub-array's groups partitioned by the parent instance
    they extract under, in parent order."""
    by = {}
    for g in groups:
        by.setdefault(g.parent, []).append(g)
    return sorted(by.items())


def _by_item(groups) -> list:
    by_item = {}
    for g in groups:
        by_item.setdefault(g.item, []).append(g)
    # the router's item index is the array position contract — groups
    # arrive in document order, calls must leave in index order
    return dict(sorted(by_item.items(),
                       key=lambda kv: kv[0] if kv[0] is not None else -1))
