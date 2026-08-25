"""Contracts stay vendor-neutral and structurally compatible."""

from xtremeparse.contracts import AgentResult, AgentRunner, ExtractionResult, Issue, Trace
from tests.helpers import FakeIssue


class FakeRunner:
    async def run(self, **kwargs):
        return AgentResult(data={}, history=[])


def test_host_issue_conforms_without_adaptation():
    issue = FakeIssue(path='basic_info.name', message='required', code='missing_required',
                      expected=['a', 'b'])
    assert isinstance(issue, Issue)


def test_runner_protocol_is_structural():
    assert isinstance(FakeRunner(), AgentRunner)


def test_result_holds_opaque_parts():
    issues = [FakeIssue(path='career', message='empty', code='missing_required')]
    result = ExtractionResult(data={'basic_info': {}}, issues=issues, trace=Trace())
    assert result.data == {'basic_info': {}}
    assert result.issues[0].path == 'career'
    assert result.trace.chunks == []
