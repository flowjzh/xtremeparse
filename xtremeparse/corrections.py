"""Correction loop: whole-data validation, issue routing, bounded re-runs.

The injected validator defines severity — whatever it feeds is retried;
an issue carrying ``report_only`` (a disputed declaration the host wants
visible) rides out with the data instead: never routed, never progress.
Issues route back to the specialist call that owns their path (per-item
calls only when the item index is derivable), and re-runs carry the call's
conversation history plus the routed feedback. A re-run with a previous
result asks for a JSON Patch against it (see patching) — untouched
entries cannot collapse in a rewrite, and the decode shrinks to the fix;
a reply that is not a patch applies as the full corrected value, and a
patch that fails to apply keeps the previous result with the next round
asking for the full value. Stops on clean, budget (default 2 rounds), or
no progress (a call whose issue paths repeat its previous round is not
re-run; an empty-patch reply — the no-fix declaration — silences its
own paths, the call still routes for a different fixable issue). Never
raises on bad data; unresolved issues return with the data.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional

from xtremeparse.contracts import AgentResult, AgentRunner, Validator
from xtremeparse.evalkit import pair_at
from xtremeparse.paths import resolve, resolve_list
from xtremeparse.executor import Call, Execution, dispatch_specialist, values_from_calls
from xtremeparse.merge import merge
from xtremeparse.patching import (PATCH_ARRAY, PATCH_NO_FIX, apply_patch,
                                  is_no_fix, is_patch)
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
               'that were already correct. Reply with ' + PATCH_NO_FIX + '. '
               'If the fix cannot be expressed as a patch, reply with the '
               'full corrected value instead.')


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
    next round for that call re-asks in full; an empty patch is the
    call's no-fix declaration — the paths it was asked for are never
    asked again (their issues ride out with the data), a different
    fixable issue still reaches the call. Returns ``(data, issues,
    rounds)`` —
    always lenient. Call results mutate in place; re-merge from
    ``execution.calls`` rather than the now-stale ``execution.values``.
    ``specialist_instructions`` must be the same override the first
    round ran with — a correction round continues that call's
    conversation history."""
    calls = execution.calls
    data = merge(values_from_calls(calls))
    issues = list(validator(data) or [])
    rounds, seen, full_form, no_fix = [], {}, set(), {}
    for _ in range(max_rounds):
        # report-only issues carry no repair: they drive neither the
        # no-progress check nor routing, and ride out with the data
        actionable = [i for i in issues if not getattr(i, 'report_only', False)]
        if not actionable:
            break
        # no progress is judged per call — a call whose issue paths
        # repeat its previous round is not re-run (a sibling's progress
        # must not drag a stuck call through another identical ask);
        # paths a call declared absent ride the no-fix declaration, a
        # different fixable issue on the same call still routes
        routed = []
        for call, feedback in _route(calls, actionable):
            if declared := no_fix.get(id(call)):
                feedback = [i for i in feedback if i.path not in declared]
            if not feedback:
                continue
            if (paths := {i.path for i in feedback}) != seen.get(id(call)):
                seen[id(call)] = paths
                routed.append((call, feedback))
        if not routed:
            break
        rounds.extend(Round(call.unit.path, call.item, [i.path for i in feedback])
                      for call, feedback in routed)
        # one patch decision per call, shared by the dispatch and the
        # reply interpretation — they must never drift apart
        plan = [(call, feedback, _patching(call, full_form))
                for call, feedback in routed]
        # every routed issue rides (a patch can fix them all at once —
        # one per round serialized the repair and burned the budget);
        # only the last is wrapped (see the PATCH_HOWTO note)
        tasks = [await dispatch_specialist(
            runner, call, payload=payload, scheduler=scheduler,
            specialist_instructions=specialist_instructions,
            history=call.result.history if call.result else None,
            feedback=([*feedback[:-1], _Directed(feedback[-1])]
                      if patching else feedback),
            schema=_patch_schema(call) if patching else None)
            for call, feedback, patching in plan]
        for (call, feedback, patching), result in zip(plan,
                                                      await asyncio.gather(*tasks)):
            if not patching or not is_patch(result.data):
                call.result = result  # the full corrected value replaces
                continue
            if is_no_fix(result.data):  # empty patch: every asked value
                # declared absent — a re-declaration accumulates
                no_fix.setdefault(id(call), set()).update(
                    i.path for i in feedback)
                continue
            patched, err = apply_patch(call.result.data, result.data,
                                       root=call.unit.path)
            if err is None:
                call.result = AgentResult(data=patched,
                                          history=result.history)
            else:  # protocol broke: forget the baseline, re-ask in full
                full_form.add(id(call))
                seen.pop(id(call), None)
        data = merge(values_from_calls(calls))
        issues = list(validator(data) or [])
    return data, issues, rounds


def _patching(call: Call, full_form: set) -> bool:
    """Whether this re-run asks for a patch: the call has a previous
    result to diff against and has not broken the protocol."""
    return call.result is not None and id(call) not in full_form


def _route(calls: list, issues: list) -> list:
    """Map issues to owning calls. Longest unit path wins; per-item
    calls match only their item; $misc is the fallback for unmatched
    paths. Count issues route by direction: shortfalls to the SHORT
    calls only (a call that returned its full slot count holds no
    missing instance, and re-running one under "return every instance"
    feedback hazards its healthy result for nothing — measured: a full
    batch re-emitted fewer entries and the merge replaced them);
    over-counts to the OVER-FULL calls, whose slices hold the
    duplicates."""
    grouped = {}
    for issue in issues:
        owners = (_count_calls(calls, issue) if isinstance(issue, _CountIssue)
                  else [_owner(calls, issue.path)])
        for call in filter(None, owners):
            grouped.setdefault(id(call), (call, []))[1].append(issue)
    return list(grouped.values())


def _count_calls(calls: list, issue) -> list:
    """The calls owning the mismatch (see _route for the why): what a
    call owes is its own ``slots`` (a batch or a single), while a
    whole-array call owes the array's declared count (the issue's
    ``expected``); shortfalls go to calls short of that, over-counts to
    calls past it. Issues carry the counts key they were built from —
    matched against ``value_path`` by identity, so a lifted sub-array's
    issue never reaches a sibling parent's call."""
    over = (issue.expected or 0) < (issue.got or 0)
    out = []
    for c in calls:
        if c.value_path != issue.key or c.result is None:
            continue
        slots = c.slots if c.slots is not None else \
            (issue.expected if c.array_shaped else None)
        if slots is None:
            continue
        got = c.produced
        if got > slots or (not over and got < slots
                           and not (c.strategy == 'whole'
                                    and (slots - got) * 100
                                    <= slots * SLIVER)):
            out.append(c)
    return out


def _owner(calls: list, path: str):
    covering = [c for c in calls
                if c.unit.path != MISC and _under(path, c.value_path)]
    exact = [c for c in covering if c.item is None and not c.batch
             or c.item is not None
             and _under(path, f'{c.value_path}[{c.item}]')
             or c.batch and (_item_index(path, c.value_path) or -1)
             in c.batch]
    pool = exact or [c for c in calls if c.unit.path == MISC]
    return max(pool, key=lambda c: len(c.unit.path), default=None)


def _item_index(path: str, scope: str):
    """The item index an issue path names inside one call's scope —
    ``unit[2].x`` → 2 for a whole-array call's scope, a lifted
    sub-array's scope included (the bracket spelling matches
    literally)."""
    m = re.fullmatch(rf'{re.escape(scope)}\[(\d+)\](\..*)?', path)
    return int(m.group(1)) if m else None


def _under(path: str, root: str) -> bool:
    """``path`` is ``root`` itself or a proper child of it."""
    return path == root or path.startswith(f'{root}.') or path.startswith(f'{root}[')


def item_chars(budgets: dict, data: dict, *, whole: set) -> dict:
    """Per budgeted path: ``[(key, arranged, chars)]`` — one entry per
    item for lists, one for the whole value otherwise, each carrying
    its own arranged budget (a lone number covers every item; a short
    list's last value covers items beyond it). ``whole`` names unit
    paths whose extraction ran as ONE call over a shared run (the
    trace's whole strategy): the arrangement — a list of material
    slices or one lone number covering the run — shares its material,
    so item-wise attribution is meaningless; one entry judges its
    total against the array's total, the keyword form's shape.
    The counting surface for budget reads (trace, eval audits); the
    unit is the item's extracted VALUE characters — the leaf values'
    own text, keys and punctuation never counted, exactly what a
    budget declares. Overruns are ACCEPTED, never retried: a retry
    costs a full extra decode, the very thing budgets exist to save —
    budgets shape batch scheduling only."""
    out = {}
    for path, arranged in budgets.items():
        if (value := resolve(data, path)) is None:
            continue
        if path in whole and isinstance(value, list):
            total = sum(arranged) if isinstance(arranged, list) else arranged
            out[path] = [(f'{path} (whole total)', total, value_chars(value))]
            continue
        values = arranged if isinstance(arranged, list) else [arranged]
        if isinstance(value, list):
            out[path] = [(f'{path}[{i}]', pair_at(values, i), value_chars(item))
                         for i, item in enumerate(value)]
        else:
            out[path] = [(path, values[-1], value_chars(value))]
    return out


@dataclass
class _CountIssue:
    """A routed item count the merged data does not honour — typically a
    whole-array call that collapsed instances into fewer entries."""

    path: str
    key: str  # the counts key the path was built from — the values
    # address routing matches by identity and a revision writes back to
    code: str = 'item_count'  # trace vocabulary only — _route keys the
    # count routing on this type, never on the code
    message: str = ''
    expected: int = None
    got: int = None
    report_only: bool = False  # a disputed declaration: surfaced, never
    # routed (an arbitration could not settle whose side is right; a
    # forced repair can only fabricate or discard entries)


def count_mismatches(counts: dict, data: dict) -> dict:
    """``{array_key: (declared, actual)}`` over every count mismatch —
    the arbitration view of count_issues: per-index shortfalls fold to
    their issue's ``key``, the values address an arbiter reads its
    items from and a revision writes back to. Folding the bare unit
    path instead handed a lifted sub-array's mismatch to its parent
    unit (measured: a sub-array holding 1 of 4 revised the healthy
    parent's declared 4 down to a phantom 2). Which side of a mismatch
    is wrong is exactly what an arbitration must decide."""
    out = {}
    for i in count_issues(counts, data):
        out.setdefault(i.key, (i.expected, i.got))
    return out


def count_issues(counts: dict, data: dict, soft=()) -> list:
    """The router's declared counts (map-validated ground truth) against
    the merged arrays. A key may carry a lifted sub-array's bracket
    scope (``parent[0].field``) — resolve reads it like any path. A
    short array means instances were collapsed or
    dropped — silent to schema validation (nothing declares minItems).
    Items concatenate in item-index order, so a short array is missing
    its tail: each missing index becomes its own issue and routes to the
    call that owns it (a batch member or a single). One that survives a
    retry stops via the no-progress rule. A unit in ``soft`` (an
    arbitration disputed its count without settling it) reports
    report-only issues — visible in the return, invisible to routing.
    Soft bites over-counts only: a disputed shortfall keeps its mend
    path, the repairable direction."""
    issues = []
    for path, declared in counts.items():
        actual = len(resolve_list(data, path))
        if actual < declared:
            issues += [_CountIssue(
                f'{path}[{i}]', key=path, expected=declared, got=actual,
                message=f'{path} is missing item {i} of {declared} — the '
                        f'array holds {actual}; return every instance as '
                        'its own entry, without splitting or duplicating')
                       for i in range(actual, declared)]
        elif actual > declared:
            soft_hit = path in soft
            tail = ('the check could not settle the dispute, reported '
                    'unrepaired') if soft_hit else 'merge the duplicates'
            issues.append(_CountIssue(
                path, key=path, expected=declared, got=actual,
                report_only=soft_hit,
                message=f'declared {declared} items but the array holds '
                        f'{actual} — {tail}'))
    return issues

