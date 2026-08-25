"""pydantic-evals adapters over the trace reads — install the ``evals``
extra (``pip install xtremeparse[evals]``). The evaluators accept any
output carrying ``.trace`` and ``.data`` the way ``ExtractionResult``
does; hosts keep their domain evaluators beside these.
"""

from __future__ import annotations

import json

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

from xtremeparse.corrections import item_chars
from xtremeparse.evalkit import per_item_budgets, router_overlap
from xtremeparse.judging import JUDGE_PLACEHOLDERS, judge
from xtremeparse.prompting import check_placeholders


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
    of the characters an item's output JSON will run to, resolved by
    code against the routed material — so actuals should land near
    the arranged number. The lower edge is loose on purpose: an idle
    estimate on a naturally short item is harmless (output runs at
    natural size either way). The upper edge flags arrangements so
    far off the router clearly wasn't counting. Per item: a mean
    would hide spread."""

    BAND = (0.4, 2.0)

    def evaluate(self, ctx: EvaluatorContext):
        ratios, wild = {}, []
        budgets = per_item_budgets(ctx.output.trace.groups)
        for path, entries in item_chars(budgets, ctx.output.data).items():
            unit = path.rsplit('.', 1)[-1]
            judged = [e for e in entries if e[1] and e[2]]  # budgeted, non-empty
            if rs := [round(chars / budget, 2) for _, budget, chars in judged]:
                ratios[unit] = f'{min(rs)}' if len(rs) == 1 else f'{min(rs)}-{max(rs)}'
            wild += [key.rsplit('.', 1)[-1] for key, budget, chars in judged
                     if not self.BAND[0] <= chars / budget <= self.BAND[1]]
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
