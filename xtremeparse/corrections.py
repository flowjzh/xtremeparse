"""Correction loop: whole-data validation, issue routing, bounded re-runs.

The injected validator defines severity — whatever it feeds is retried.
Issues route back to the specialist call that owns their path (per-item
calls only when the item index is derivable), and re-runs carry the call's
conversation history plus the routed feedback. A re-run with a previous
result asks for a JSON Patch against it (see patching) — untouched
entries cannot collapse in a rewrite, and the decode shrinks to the fix;
a reply that is not a patch applies as the full corrected value, and a
patch that fails to apply keeps the previous result with the next round
asking for the full value. Stops on clean, budget (default 2 rounds), or
no progress (issue paths identical to the previous round). Never raises
on bad data; unresolved issues return with the data.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional

from xtremeparse.contracts import AgentResult, AgentRunner, Validator
from xtremeparse.paths import resolve, resolve_list
from xtremeparse.executor import Call, Execution, dispatch_specialist, values_from_calls
from xtremeparse.merge import merge
from xtremeparse.patching import PATCH_ARRAY, apply_patch, is_patch
from xtremeparse.prompting import value_chars
from xtremeparse.scheduling import TaskScheduler
from xtremeparse.units import MISC

MAX_ROUNDS = 2
SLIVER = 10  # a whole-array call short by ≤SLIVER% of its declaration is
# not retried — the retry re-decodes the unit's full material and re-emits
# the same belief; the shortfall surfaces as issues instead
# the protocol directive rides the last routed issue's message — it must
# be the freshest text before the adapter's output block, which pins the
# wire shape; see patching.PATCH_ARRAY for the shape half of the contract
PATCH_HOWTO = ('\n\nFix by JSON Patch (RFC 6902): reply with an array '
               'of {"op", "path", "value"} operations applied to your '
               'previous result — op "add" (path "/<index>", "-" appends), '
               '"remove" or "replace"; a pointer may be rooted at the '
               'unit path. Emit only what changes; never re-emit entries '
               'that were already correct. If the fix cannot be expressed '
               'as a patch, reply with the full corrected value instead.')


class _Directed:
    """The last routed issue with the patch protocol appended to its
    message. Delegation, not a dataclass copy: validators inject
    arbitrary Issue-protocol objects the loop must not assume."""

    def __init__(self, issue):
        self._issue = issue

    def __getattr__(self, name):
        return getattr(self._issue, name)

    @property
    def message(self):
        return self._issue.message + PATCH_HOWTO


def _patch_schema(call: Call) -> dict:
    """The reply shape of a patch round: a JSON Patch or the unit's
    full value."""
    return {'anyOf': [PATCH_ARRAY, call.value_shape]}


@dataclass
class Round:
    """One correction round's trace material."""

    unit_path: str
    item: Optional[int]
    issue_paths: list


async def correct(runner: AgentRunner, execution: Execution, *, validator: Validator,
                  payload: str, scheduler: TaskScheduler,
                  max_rounds: int = MAX_ROUNDS,
                  specialist_instructions: str = None) -> tuple:
    """Re-run failing calls with feedback until clean, budgeted, or
    stuck. Re-runs with a previous result ask for a JSON Patch against
    it; a patch that fails to apply keeps the previous result and the
    next round for that call re-asks in full. Returns ``(data, issues,
    rounds)`` — always lenient. Call results mutate in place; re-merge
    from ``execution.calls`` rather than the now-stale
    ``execution.values``. ``specialist_instructions`` must be the same
    override the first round ran with — a correction round continues
    that call's conversation history."""
    calls = execution.calls
    data = merge(values_from_calls(calls))
    issues = list(validator(data) or [])
    rounds, seen, full_form = [], None, set()
    for _ in range(max_rounds):
        paths = {i.path for i in issues}
        if not issues or paths == seen:
            break
        seen = paths
        routed = _route(calls, issues)
        if not routed:
            break
        rounds.extend(Round(call.unit.path, call.item, [i.path for i in feedback])
                      for call, feedback in routed)
        # one patch decision per call, shared by the dispatch and the
        # reply interpretation — they must never drift apart
        plan = [(call, feedback, _patching(call, full_form))
                for call, feedback in routed]
        tasks = [await dispatch_specialist(
            runner, call, payload=payload, scheduler=scheduler,
            specialist_instructions=specialist_instructions,
            history=call.result.history if call.result else None,
            feedback=[_Directed(feedback[-1])] if patching else feedback,
            schema=_patch_schema(call) if patching else None)
            for call, feedback, patching in plan]
        for (call, _, patching), result in zip(plan,
                                               await asyncio.gather(*tasks)):
            if not patching or not is_patch(result.data):
                call.result = result  # the full corrected value replaces
                continue
            patched, err = apply_patch(call.result.data, result.data,
                                       root=call.unit.path)
            if err is None:
                call.result = AgentResult(data=patched,
                                          history=result.history)
            else:  # protocol broke: forget the baseline, re-ask in full
                full_form.add(id(call))
                seen = None
        data = merge(values_from_calls(calls))
        issues = list(validator(data) or [])
    return data, issues, rounds


def _patching(call: Call, full_form: set) -> bool:
    """Whether this re-run asks for a patch: the call has a previous
    result to diff against and has not broken the protocol."""
    return call.result is not None and id(call) not in full_form


def _route(calls: list, issues: list) -> list:
    """Map issues to owning calls. Longest unit path wins; per-item calls
    match only their item; $misc is the fallback for unmatched paths.
    Count shortfalls route to the SHORT calls only — a call that
    returned its full slot count holds no missing instance, and re-running
    one under "return every instance" feedback hazards its healthy
    result for nothing (measured: a full batch re-emitted fewer entries
    and the merge replaced them)."""
    grouped = {}
    for issue in issues:
        owners = (_short_calls(calls, issue) if isinstance(issue, _CountIssue)
                  else [_owner(calls, issue.path)])
        for call in filter(None, owners):
            grouped.setdefault(id(call), (call, []))[1].append(issue)
    return list(grouped.values())


def _short_calls(calls: list, issue) -> list:
    """The unit's calls still short of their slots — a full call holds
    no missing instance. Whole-array calls owe the unit's declared
    count (the issue's ``expected``); batches and singles owe their own
    ``slots``. A whole call short by a sliver (≤SLIVER%) is not retried."""
    unit = issue.path.partition('[')[0]
    out = []
    for c in calls:
        if c.unit.path != unit or c.result is None:
            continue
        slots = c.slots if c.slots is not None else \
            (issue.expected if c.array_shaped else None)
        if slots is None:
            continue
        got = len(c.result.data or [])
        if got >= slots:
            continue
        if c.strategy == 'whole' and (slots - got) * 100 <= slots * SLIVER:
            continue
        out.append(c)
    return out


def _owner(calls: list, path: str):
    covering = [c for c in calls if c.unit.path != MISC and _under(path, c.unit.path)]
    exact = [c for c in covering if c.item is None and not c.batch
             or c.item is not None and _under(path, f'{c.unit.path}[{c.item}]')
             or c.batch and (_item_index(path, c.unit.path) or -1) in c.batch]
    pool = exact or [c for c in calls if c.unit.path == MISC]
    return max(pool, key=lambda c: len(c.unit.path), default=None)


def _item_index(path: str, unit_path: str):
    m = re.fullmatch(rf'{re.escape(unit_path)}\[(\d+)\](\..*)?', path)
    return int(m.group(1)) if m else None


def _under(path: str, root: str) -> bool:
    """``path`` is ``root`` itself or a proper child of it."""
    return path == root or path.startswith(f'{root}.') or path.startswith(f'{root}[')


def item_chars(budgets: dict, data: dict) -> dict:
    """Per budgeted path: ``[(key, arranged, chars)]`` — one entry per
    item for lists, one for the whole value otherwise, each carrying
    its own arranged budget (a lone number covers every item; a short
    list's last value covers items beyond it). The counting surface for
    budget reads (trace, eval audits); the unit is the item's
    extracted VALUE characters — the leaf values' own text, keys and
    punctuation never counted, exactly what a budget declares.
    Overruns are ACCEPTED, never retried: a retry costs a full extra
    decode, the very thing budgets exist to save — budgets shape batch
    scheduling only."""
    out = {}
    for path, arranged in budgets.items():
        if (value := resolve(data, path)) is None:
            continue
        values = arranged if isinstance(arranged, list) else [arranged]
        entries = []
        for i, item in (enumerate(value) if isinstance(value, list)
                        else [(None, value)]):
            budget = values[i] if i is not None and i < len(values) else values[-1]
            entries.append((f'{path}[{i}]' if i is not None else path,
                            budget, value_chars(item)))
        out[path] = entries
    return out


@dataclass
class _CountIssue:
    """A routed item count the merged data does not honour — typically a
    whole-array call that collapsed instances into fewer entries."""

    path: str
    code: str = 'item_count'  # trace vocabulary only — _route keys the
    # short-call routing on this type, never on the code
    message: str = ''
    expected: int = None
    got: int = None


def count_issues(counts: dict, data: dict) -> list:
    """The router's declared counts (map-validated ground truth) against
    the merged arrays. A short array means instances were collapsed or
    dropped — silent to schema validation (nothing declares minItems).
    Items concatenate in item-index order, so a short array is missing
    its tail: each missing index becomes its own issue and routes to the
    call that owns it (a batch member or a single). One that survives a
    retry stops via the no-progress rule."""
    issues = []
    for path, declared in counts.items():
        actual = len(resolve_list(data, path))
        if actual < declared:
            issues += [_CountIssue(
                f'{path}[{i}]', expected=declared, got=actual,
                message=f'{path} is missing item {i} of {declared} — the '
                        f'array holds {actual}; return every instance as '
                        'its own entry, without splitting or duplicating')
                       for i in range(actual, declared)]
        elif actual > declared:
            issues.append(_CountIssue(
                path, expected=declared, got=actual,
                message=f'declared {declared} items but the array holds '
                        f'{actual} — merge the duplicates'))
    return issues

