"""pydantic-evals adapters over the trace reads — install the ``evals``
extra (``pip install xtremeparse[evals]``). The evaluators accept any
output carrying ``.trace`` and ``.data`` the way ``ExtractionResult``
does; hosts keep their domain evaluators beside these.
"""

from __future__ import annotations

import json

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

from xtremeparse.corrections import item_chars
from xtremeparse.evalkit import pair_at, per_item_budgets, router_overlap, whole_units
from xtremeparse.judging import JUDGE_PLACEHOLDERS, judge
from xtremeparse.prompting import check_placeholders
from xtremeparse.router import budget_kind


class RouterOverlap(Evaluator):
    """Invariant tripwire: router validation makes overlap and phantom
    items structurally impossible — a red here means the router contract
    regressed, not a flaky model."""

    def evaluate(self, ctx: EvaluatorContext):
        overlap, lost = router_overlap(ctx.output.trace.router,
                                       ctx.output.trace.groups)
        return EvaluationReason(lost == 0, f'overlap={overlap}, items_lost={lost}')


class BudgetFit(Evaluator):
    """Allocation audit: the model ARRANGES each budget — its estimate
    of the value characters an item's extraction will run to, resolved
    by code against the routed material — so actuals (the extracted
    values' own characters, keys never counted) should land near the
    arranged number. Ratio and absolute forms are judged per item: a
    mean would hide spread. A keyword form is judged on its TOTAL —
    the entries' length sum against the declared average-times-count:
    the declaration is the model's own reading of the entries, so the
    same band as every other form applies. A whole-strategy run's
    budget list shares its material across the run's items (each entry
    is a slice, and a merged item can concentrate another unit's
    values), so its item-wise attribution is meaningless — the list is
    judged on its total the same way. Both edges carry an
    absolute slack: an entry's schema fields cost a fixed overhead
    the material cannot predict (dates, a distilled name beside a
    verbatim description) — clearing the band by less than that
    overhead in absolute characters is the same harmless case."""

    BAND = (0.4, 2.0)
    SLACK = 25  # characters either way; small items clear the band
    # by less than their fields' own fixed cost

    def evaluate(self, ctx: EvaluatorContext):
        ratios, wild = {}, []
        budgets = per_item_budgets(ctx.output.trace.groups)
        whole = whole_units(ctx.output.trace.groups)
        declared = dict((ctx.output.trace.router or {}).get('budgets') or {})
        for path, perm in (getattr(ctx.output, 'sort', None) or {}).items():
            if perm == list(range(len(perm))):
                continue  # the sort moved nothing: no realignment needed
            # the host sorted the data after extraction: realign the
            # extraction-order arrangements with the sorted items —
            # sorted position i holds the item perm[i] declared it
            for src in (budgets, declared):
                if isinstance((arr := src.get(path)), list) and arr:
                    src[path] = [pair_at(arr, p) for p in perm]
        for path, entries in item_chars(budgets, ctx.output.data,
                                         whole=whole).items():
            unit = path.rsplit('.', 1)[-1]
            tokens = declared.get(path)
            tokens = [str(t) for t in tokens] if isinstance(tokens, list) else [str(tokens)]
            pairs = [(e, budget_kind(pair_at(tokens, i)) == 'kw')
                     for i, e in enumerate(entries)]
            checks = [e for e, kw in pairs if not kw and e[1] and e[2]]
            if kw := [e for e, is_kw in pairs if is_kw and e[1]]:
                checks.append((f'{path} (kw total)',
                               sum(b for _, b, _ in kw),
                               sum(c for _, _, c in kw)))
            if rs := [round(chars / budget, 2) for _, budget, chars in checks if budget]:
                ratios[unit] = f'{min(rs)}' if len(rs) == 1 else f'{min(rs)}-{max(rs)}'
            wild += [key.rsplit('.', 1)[-1] for key, budget, chars in checks
                     if budget and (not self.BAND[0] <= chars / budget <= self.BAND[1]
                                    and abs(chars - budget) > self.SLACK)]
        return EvaluationReason(
            not wild, f'act/budget {ratios or "no budgets declared"}'
                      + (f', wild: {wild}' if wild else ''))


class RubricJudge(Evaluator):
    """One rubric call through an injected AgentRunner (see
    xtremeparse.judging) — the judge runs on whatever framework the
    host adapted, where pydantic-evals' own LLMJudge ties the harness
    to a pydantic-ai model. ``source`` is the judged material's
    document, held at construction; ``output_of`` projects the case
    output into what the rubric asks about (e.g. a digest projection
    for containment questions) — see docs/prompting.md for the
    measured judge guidance. The judge's own model settings are
    recommendations carried at the adapter."""

    def __init__(self, runner, rubric: str, source: str = '', *,
                 output_of=None, instructions: str = None):
        if instructions is not None:
            check_placeholders(instructions, JUDGE_PLACEHOLDERS, 'instructions')
        self.runner = runner
        self.rubric = rubric
        self.source = source
        self.output_of = output_of
        self.instructions = instructions

    async def evaluate(self, ctx: EvaluatorContext):
        output = _text(self.output_of(ctx.output) if self.output_of
                       else ctx.output)
        verdict = await judge(self.runner, rubric=self.rubric,
                              source=self.source, output=output,
                              instructions=self.instructions)
        return EvaluationReason(verdict.ok, verdict.reason)


def _text(value) -> str:
    """Stable judge-facing text: an ExtractionResult-shaped value is
    judged by its data; a plain string stays verbatim (JSON-escaping a
    document-sized string would bury its structure); the rest
    serializes compactly (``default=str`` absorbs harness objects like
    Path)."""
    if hasattr(value, 'data'):
        value = value.data
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
