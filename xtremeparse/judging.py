"""Protocol-neutral LLM judging: one rubric call through the injected
AgentRunner — judge with whatever framework the host adapted, the same
neutrality extraction has. The rubric text stays the host's (domain
semantics, schema-description tier); this module is mechanism only.
"""

from __future__ import annotations

from dataclasses import dataclass

from xtremeparse.contracts import AgentRunner
from xtremeparse.prompting import check_placeholders

JUDGE_INSTRUCTIONS = '''You are an impartial judge. Decide whether the assertion below holds
for the output, given the source. The assertion is the complete
standard — its exclusions are as binding as its requirements; weigh
only what it asks, nothing else. Before counting a candidate loss,
verify it against the output and the assertion: drop every candidate
the output contains or the assertion excludes — the verdict follows
only from what survives.

Assertion:

{rubric}'''

JUDGE_PLACEHOLDERS = frozenset({'rubric'})

VERDICT_SCHEMA = {'type': 'object', 'properties': {
    'ok': {'type': 'boolean', 'description': 'the assertion holds'},
    'reason': {'type': 'string',
               'description': 'the evidence, one or two sentences'}},
    'required': ['ok', 'reason'], 'additionalProperties': False}


@dataclass
class Verdict:
    """One rubric decision and its evidence."""

    ok: bool
    reason: str


async def judge(runner: AgentRunner, *, rubric: str, source: str, output: str,
                instructions: str = None) -> Verdict:
    """One rubric call through the injected runner. Judge-side settings
    live at the adapter's model construction — recommendations, not
    requirements (see docs/prompting.md). The call goes straight to
    the runner, outside any scheduler budget — a judge should not
    compete with the fleet for quota. ``instructions`` replaces the
    judge template and must carry the ``rubric`` placeholder."""
    if instructions is not None:
        check_placeholders(instructions, JUDGE_PLACEHOLDERS, 'instructions')
    result = await runner.run(
        instructions=(instructions or JUDGE_INSTRUCTIONS).format(rubric=rubric),
        result_schema=VERDICT_SCHEMA,
        content=f'Source:\n\n{source}\n\n---\nOutput:\n\n{output}')
    verdict = result.data if isinstance(result.data, dict) else {}
    return Verdict(bool(verdict.get('ok')), str(verdict.get('reason') or ''))
