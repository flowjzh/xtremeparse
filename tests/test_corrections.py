"""Boundary probes for the correction loop: routing, rounds, leniency."""

from xtremeflow.scheduler import TaskScheduler

from xtremeparse.contracts import AgentResult
from xtremeparse.corrections import correct, count_issues, item_chars
from xtremeparse.executor import Call, Execution
from xtremeparse.units import MISC, decompose
from tests.helpers import FakeIssue, ScriptedRunner, agent_result

SCHEMA = {
    'type': 'object',
    'properties': {
        'basic_info': {'type': 'object', 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
        'jobs': {'type': 'array', 'description': '经历', 'items': {'type': 'object', 'properties': {
            'company': {'type': 'string', 'description': '公司'},
        }}},
        'created': {'type': 'string', 'description': '创建时间'},
    },
}
UNITS = {u.path: u for u in decompose(SCHEMA)}


def call(path, data, *, item=None, strategy=None, scope='材料'):
    return Call(UNITS[path], item, strategy, [0], scope, agent_result(data))


def execution(*calls):
    return Execution({}, list(calls))


def scheduler():
    return TaskScheduler(4)


async def test_round_two_fixes_the_failure_and_converges():
    validator_calls = []

    def validator(data):
        validator_calls.append(data)
        if data.get('basic_info', {}).get('name') != '张三':
            return [FakeIssue('basic_info.name')]
        return []

    runner = ScriptedRunner(agent_result({'name': '张三'}))
    data, issues, rounds = await correct(
        runner, execution(call('basic_info', {'age': '30'})),
        validator=validator, payload='p', scheduler=scheduler())
    assert data == {'basic_info': {'name': '张三'}} and issues == []
    assert [(r.unit_path, r.issue_paths) for r in rounds] == [('basic_info', ['basic_info.name'])]
    assert len(validator_calls) == 2
    rerun = runner.calls[0]
    assert rerun['history'] == [] and rerun['feedback'][0].path == 'basic_info.name'
    assert rerun['scope'] == '材料'


async def test_persistent_failure_stops_on_no_progress_not_budget():
    runner = ScriptedRunner(agent_result({'still': 'wrong'}),
                            agent_result({'still': 'wrong'}))
    data, issues, rounds = await correct(
        runner, execution(call('basic_info', {'age': '30'})),
        validator=lambda data: [FakeIssue('basic_info.name')],
        payload='p', scheduler=scheduler(), max_rounds=5)
    assert len(runner.calls) == 1  # second identical path set stopped the loop
    assert issues[0].path == 'basic_info.name'  # unresolved, reported, no raise


async def test_clean_first_validation_never_reruns():
    runner = ScriptedRunner()
    _, issues, rounds = await correct(
        runner, execution(call('basic_info', {'name': '张三'})),
        validator=lambda data: [], payload='p', scheduler=scheduler())
    assert runner.calls == [] and issues == [] and rounds == []


async def test_bracketed_item_issue_reruns_only_that_item():
    runner = ScriptedRunner(agent_result({'company': '阿里修复'}))
    calls = [call('jobs', {'company': '腾讯'}, item=0, strategy='per-item', scope='s0'),
             call('jobs', {'company': ''}, item=1, strategy='per-item', scope='s1')]
    _, _, rounds = await correct(
        runner, execution(*calls),
        validator=lambda data: [FakeIssue('jobs[1].company')],
        payload='p', scheduler=scheduler())
    assert runner.calls[0]['scope'] == 's1'
    assert [(r.unit_path, r.item) for r in rounds] == [('jobs', 1)]


async def test_double_digit_item_does_not_match_single_digit_call():
    runner = ScriptedRunner(agent_result({'company': '修复'}))
    calls = [call('jobs', {'company': '腾讯'}, item=1, strategy='per-item', scope='s1')]
    await correct(runner, execution(*calls),
                  validator=lambda data: [FakeIssue('jobs[12].company')],
                  payload='p', scheduler=scheduler())
    assert runner.calls == []  # jobs[12] is not under jobs[1]: nothing routable


async def test_toplevel_scalar_issue_falls_back_to_misc():
    runner = ScriptedRunner(agent_result({'created': '2026-01-01'}))
    calls = [call('basic_info', {'name': '张三'}), call(MISC, {})]
    await correct(runner, execution(*calls),
                  validator=lambda data: [FakeIssue('created')],
                  payload='p', scheduler=scheduler())
    assert runner.calls[0]['scope'] == '材料'  # the misc call was rerun


async def test_unroutable_issue_without_misc_breaks_and_reports():
    runner = ScriptedRunner()
    data, unresolved, rounds = await correct(
        runner, execution(call('basic_info', {'name': '张三'})),
        validator=lambda data: [FakeIssue('nowhere.field')],
        payload='p', scheduler=scheduler())
    assert runner.calls == [] and rounds == []
    assert unresolved[0].path == 'nowhere.field' and data == {'basic_info': {'name': '张三'}}


async def test_zero_budget_validates_once_and_returns():
    runner = ScriptedRunner()
    data, issues, rounds = await correct(
        runner, execution(call('basic_info', {})),
        validator=lambda data: [FakeIssue('basic_info.name')],
        payload='p', scheduler=scheduler(), max_rounds=0)
    assert runner.calls == [] and rounds == []
    assert issues[0].path == 'basic_info.name'


async def test_history_from_original_result_is_passed_back():
    runner = ScriptedRunner(agent_result({'name': '张三'}))
    original = AgentResult(data={'age': '30'}, history=[{'turn': 1}])
    calls = [Call(UNITS['basic_info'], None, None, [0], '材料', original)]
    await correct(runner, Execution({}, calls),
                  validator=lambda data: [FakeIssue('basic_info.name')],
                  payload='p', scheduler=scheduler())
    rerun = runner.calls[0]
    assert rerun['history'] == [{'turn': 1}]
    # the transcript already carries the card and scope — the rerun
    # passes only the per-round feedback (layout contract, contracts.py)
    assert rerun['instructions'] == '' and rerun['scope'] == ''


def test_item_chars_pairs_each_item_with_its_own_budget():
    data = {'career': {'jobs': [{'company': '腾讯'},
                                {'company': '腾讯',
                                 'positions': [{'title': '工程师'}]}]}}
    assert item_chars({'career.jobs': [10, 20]}, data) == {
        'career.jobs': [('career.jobs[0]', 10, 16),
                        ('career.jobs[1]', 20, 46)]}


def test_item_chars_lone_number_covers_every_item():
    assert item_chars({'career.jobs': 5}, {'career': {'jobs': [
        {'company': '腾讯'}, {'company': '阿里'}]}}) == {
        'career.jobs': [('career.jobs[0]', 5, 16), ('career.jobs[1]', 5, 16)]}


def test_item_chars_checks_non_list_values_whole_and_skips_missing():
    assert item_chars({'basic_info': 1}, {'basic_info': {'name': '张三'}}) == {
        'basic_info': [('basic_info', 1, 13)]}
    assert item_chars({'jobs': 5}, {}) == {}


async def test_correction_rerun_carries_the_budget():
    runner = ScriptedRunner(agent_result({'company': '短'}))
    await correct(runner, execution(
        Call(UNITS['jobs'], 0, 'per-item', [0], '材料',
             agent_result({'company': '很长很长的公司名'}), budget=10)),
        validator=lambda d: [FakeIssue('jobs[0].company')],
        payload='p', scheduler=scheduler())
    assert 'Output budget' not in runner.calls[0]['instructions']


def test_count_issues_reconcile_declared_against_data():
    data = {'career': {'jobs': [{'company': '腾讯'}, {'company': '阿里'}]}}
    assert count_issues({'career.jobs': 2, 'education': 0}, data) == []
    collapsed = count_issues({'career.jobs': 3}, data)
    assert [(i.path, i.code, i.expected, i.got) for i in collapsed] == \
        [('career.jobs[2]', 'item_count', 3, 2)]
    assert 'missing item 2' in collapsed[0].message
    assert [i.path for i in count_issues({'jobs': 2}, {})] == ['jobs[0]', 'jobs[1]']
