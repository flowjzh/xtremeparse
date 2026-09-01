"""Protocol doubles for hosts testing against xtremeparse."""

from __future__ import annotations

from dataclasses import dataclass

from xtremeparse.contracts import AgentResult
from xtremeparse.patching import is_patch_round, value_branch


@dataclass
class FakeIssue:
    """Issue-protocol-shaped double."""

    path: str
    message: str = 'invalid'
    code: str = 'type_mismatch'
    expected: object = None
    got: object = None


def plain_schema(schema: dict) -> dict:
    """A patch-or-value correction round's full-value branch — the
    shape a scripted runner keys a non-patch reply on. Plain schemas
    pass through unchanged."""

    if is_patch_round(schema):
        return value_branch(schema)
    return schema


class ScriptedRunner:
    """Fake AgentRunner: returns queued AgentResults, records every call."""

    def __init__(self, *results: AgentResult):
        self._results = list(results)
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self._results.pop(0)


def agent_result(data) -> AgentResult:
    return AgentResult(data=data, history=[])
