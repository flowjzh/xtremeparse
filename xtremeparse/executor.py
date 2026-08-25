"""Executor: fan specialists out under the scheduler, then merge.

Object and scalar units coalesce their routed groups into one specialist
call; per-item arrays coalesce duplicate rows of the same item. Array
units decide strategy after routing, from the actual routed volume:
whole-array when it fits the threshold, per-item fan-out when it does
not — a per-path ``unit_strategy`` override encodes prior knowledge.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

from xtremeparse.contracts import AgentRunner, AgentResult
from xtremeparse.prompting import SPECIALIST_INSTRUCTIONS, estimate_tokens
from xtremeparse.router import Routing
from xtremeparse.scheduling import TaskScheduler
from xtremeparse.units import Unit

Strategy = Literal['whole', 'per-item']
BATCH_BUDGET_CAP = 300  # accumulated budget one shared call may absorb


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


@dataclass
class Execution:
    values: dict
    calls: list


async def execute(runner: AgentRunner, routing: Routing, *, payload: str,
                  scheduler: TaskScheduler, unit_strategy: dict = None,
                  budgets: dict = None,
                  specialist_instructions: str = None,
                  batch_cap: int = BATCH_BUDGET_CAP) -> Execution:
    """Run specialist calls concurrently and collect values keyed by unit
    path. ``payload`` must be the byte-identical shared prefix the router
    was called with — that identity is what makes the provider KV cache
    hit for the whole fleet."""
    calls, tasks = [], []
    for unit, groups in _by_unit(routing.groups):
        strategy = _strategy(unit, groups, unit_strategy or {}) \
            if unit.kind == 'array' else None
        if strategy == 'per-item':
            specs = _batched(unit, _by_item(groups), budgets, batch_cap)
        else:
            # co-chunked items (one run holding several instances) share
            # their material — extract it once, not once per item
            unique = {tuple(g.chunk_ids): g for g in groups}.values()
            specs = [(None, '\n\n'.join(g.text for g in unique),
                      [i for g in unique for i in g.chunk_ids], None,
                      _arranged(budgets, unit, None))]
        for item, scope, ids, batch, budget in specs:
            calls.append(Call(unit, item, strategy, ids, scope, budget=budget,
                              batch=batch))
            tasks.append(await dispatch_specialist(
                runner, unit, payload=payload, scope=scope,
                whole=strategy == 'whole' or bool(batch), scheduler=scheduler,
                specialist_instructions=specialist_instructions))
    for call, result in zip(calls, await asyncio.gather(*tasks)):
        call.result = result
    return Execution(values_from_calls(calls), calls)


def values_from_calls(calls: list) -> dict:
    """Per-unit values from call results; None data stays absent and
    per-item results concatenate in item-index order."""
    values = {}
    for call in calls:
        data = call.result.data if call.result else None
        if data is None:
            continue
        if call.batch:
            values.setdefault(call.unit.path, []).extend(
                data if isinstance(data, list) else [data])
        elif call.strategy == 'per-item':
            values.setdefault(call.unit.path, []).append(data)
        else:
            values[call.unit.path] = data
    return values


async def dispatch_specialist(runner: AgentRunner, unit: Unit, *, payload: str,
                              scope: str, whole: bool, scheduler: TaskScheduler,
                              specialist_instructions: str = None,
                              history: list = None,
                              feedback: list = None) -> asyncio.Task:
    """Schedule the one specialist invocation (executor and correction
    rounds share it, so the cache-frozen prompt layout stays identical)
    under the shared budget, seeded with the recipe's own token
    estimate. Returns the dispatched task."""
    async def invoke() -> AgentResult:
        schema = {'type': 'array', 'items': unit.sub_schema} if whole else unit.sub_schema
        instructions = (specialist_instructions or SPECIALIST_INSTRUCTIONS).format(
            card=unit.card)
        # a history round already carries the card and scope in the
        # transcript — only the per-round feedback is fresh material
        return await runner.run(instructions='' if history else instructions,
                                result_schema=schema, content=payload,
                                scope='' if history else scope, history=history,
                                feedback=feedback)
    return await scheduler.start_task(
        invoke(), estimated_tokens=estimate_tokens(payload, scope, unit.card))


def _arranged(budgets: dict, unit: Unit, item: Optional[int]):
    """The budget a call carries: its own item's number from the
    arranged list (a lone number covers every item; a short list's last
    value covers items beyond it); a whole-array call keeps the list."""
    if (arranged := (budgets or {}).get(unit.path)) is None:
        return None
    values = arranged if isinstance(arranged, list) else [arranged]
    if item is None:
        return values[0] if len(values) == 1 else list(values)
    return values[item] if item < len(values) else values[-1]


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
    small streams beat one long one even when the material is short
    (measured 18s -> 8s on the corpus' smallest file). Whole is for what
    per-item scoping cannot split: co-chunked instances, single items."""
    if override := unit_strategy.get(unit.path):
        if override not in ('whole', 'per-item'):
            raise ValueError(f'unknown unit_strategy for {unit.path}: {override!r}')
        return override
    ids = [i for g in groups for i in g.chunk_ids]
    if len(ids) != len(set(ids)):
        return 'whole'  # co-chunked items: one call must split the instances
    return 'whole' if len({g.item for g in groups}) <= 1 else 'per-item'


def _by_unit(groups) -> list:
    by_path = {}
    for g in groups:
        by_path.setdefault(g.unit.path, []).append(g)
    return [(rows[0].unit, rows) for rows in by_path.values()]


def _by_item(groups) -> list:
    by_item = {}
    for g in groups:
        by_item.setdefault(g.item, []).append(g)
    # the router's item index is the array position contract — groups
    # arrive in document order, calls must leave in index order
    return dict(sorted(by_item.items(),
                       key=lambda kv: kv[0] if kv[0] is not None else -1))
