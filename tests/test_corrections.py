"""Boundary probes for the correction loop: routing, rounds, leniency."""

from xtremeflow.scheduler import TaskScheduler

from xtremeparse.contracts import AgentResult
from xtremeparse.corrections import correct, count_issues, item_chars
from xtremeparse.executor import Call, Execution
from xtremeparse.patching import is_patch_round
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


def call(path, data, *, item=None, strategy=None, scope='材料', batch=()):
    return Call(UNITS[path], item, strategy, [0], scope, agent_result(data),
                batch=batch)


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
        'career.jobs': [('career.jobs[0]', 10, 2),
                        ('career.jobs[1]', 20, 5)]}


def test_item_chars_lone_number_covers_every_item():
    assert item_chars({'career.jobs': 5}, {'career': {'jobs': [
        {'company': '腾讯'}, {'company': '阿里'}]}}) == {
        'career.jobs': [('career.jobs[0]', 5, 2), ('career.jobs[1]', 5, 2)]}


def test_item_chars_checks_non_list_values_whole_and_skips_missing():
    assert item_chars({'basic_info': 1}, {'basic_info': {'name': '张三'}}) == {
        'basic_info': [('basic_info', 1, 2)]}
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


async def test_count_shortfall_retries_only_short_batches():
    # a full batch holds no missing instance — the shortfall retries the
    # short batch alone, and the full batch's healthy result stands
    full = call('jobs', [{'company': '腾讯'}, {'company': '阿里'}],
                strategy='per-item', scope='s01', batch=(0, 1))
    short = call('jobs', [{'company': '美团'}],
                 strategy='per-item', scope='s23', batch=(2, 3))
    runner = ScriptedRunner(agent_result([{'company': '美团'}, {'company': '京东'}]))
    data, issues, rounds = await correct(
        runner, execution(full, short),
        validator=lambda d: count_issues({'jobs': 4}, d),
        payload='p', scheduler=scheduler())
    assert [c['scope'] for c in runner.calls] == ['s23']
    assert [c['company'] for c in data['jobs']] == ['腾讯', '阿里', '美团', '京东']
    assert issues == []


async def test_count_shortfall_with_no_short_call_surfaces_unrouted():
    # every call returned its slots; the extra declared index has no owner
    # — no destructive retries, the shortfall reports
    full = call('jobs', [{'company': '腾讯'}, {'company': '阿里'}],
                strategy='per-item', scope='s01', batch=(0, 1))
    runner = ScriptedRunner()
    data, issues, rounds = await correct(
        runner, execution(full),
        validator=lambda d: count_issues({'jobs': 3}, d),
        payload='p', scheduler=scheduler())
    assert runner.calls == [] and rounds == []
    assert [i.path for i in issues] == ['jobs[2]']


async def test_over_count_retries_only_the_over_full_batch():
    # duplicates push one batch past its slots — the over-count routes to
    # the batch holding the extras, and healthy calls stand
    full = call('jobs', [{'company': '腾讯'}, {'company': '阿里'}],
                strategy='per-item', scope='s01', batch=(0, 1))
    over = call('jobs', [{'company': '美团'}, {'company': '京东'},
                         {'company': '美团'}, {'company': '京东'}],
                strategy='per-item', scope='s23', batch=(2, 3))
    runner = ScriptedRunner(agent_result([{'company': '美团'}, {'company': '京东'}]))
    data, issues, rounds = await correct(
        runner, execution(full, over),
        validator=lambda d: count_issues({'jobs': 4}, d),
        payload='p', scheduler=scheduler())
    assert [c['scope'] for c in runner.calls] == ['s23']
    assert [c['company'] for c in data['jobs']] == ['腾讯', '阿里', '美团', '京东']
    assert issues == []


async def test_over_count_whole_call_holding_the_duplicates_is_rerun():
    # a whole-array call over its declared count owns the merge itself
    whole = call('jobs', [{'company': '腾讯'}, {'company': '阿里'},
                          {'company': '阿里'}], strategy='whole', scope='s')
    runner = ScriptedRunner(agent_result([{'company': '腾讯'}, {'company': '阿里'}]))
    data, issues, rounds = await correct(
        runner, execution(whole),
        validator=lambda d: count_issues({'jobs': 2}, d),
        payload='p', scheduler=scheduler())
    assert [c['scope'] for c in runner.calls] == ['s']
    assert [c['company'] for c in data['jobs']] == ['腾讯', '阿里']
    assert issues == []


def test_soft_count_issues_report_without_routing():
    # a disputed declaration: the over-count stays visible, marked
    # report-only, its message naming the dispute — not a repair order
    data = {'jobs': [{'company': '腾讯'}, {'company': '阿里'},
                     {'company': '美团'}]}
    (soft,) = count_issues({'jobs': 2}, data, soft={'jobs'})
    assert soft.report_only and soft.expected == 2 and soft.got == 3
    assert 'could not settle' in soft.message
    (plain,) = count_issues({'jobs': 2}, data)
    assert not plain.report_only and 'merge the duplicates' in plain.message


async def test_a_disputed_over_count_never_triggers_a_round():
    # a report-only issue rides out with the data — no repair, no merge
    whole = call('jobs', [{'company': '腾讯'}, {'company': '阿里'},
                          {'company': '美团'}], strategy='whole', scope='s')
    runner = ScriptedRunner()
    data, issues, rounds = await correct(
        runner, execution(whole),
        validator=lambda d: count_issues({'jobs': 2}, d, soft={'jobs'}),
        payload='p', scheduler=scheduler())
    assert runner.calls == [] and rounds == []
    assert [i.report_only for i in issues] == [True]
    assert [c['company'] for c in data['jobs']] == ['腾讯', '阿里', '美团']


async def test_a_report_only_issue_does_not_mask_repairable_progress():
    # mixed issues: the actionable one drives the round; the report-only
    # one neither routes nor counts as stuckness
    over = call('jobs', [{'company': '腾讯'}, {'company': '阿里'},
                         {'company': '美团'}], strategy='whole', scope='j')
    misc = call(MISC, {})
    runner = ScriptedRunner(agent_result({'created': '2026-01-01'}))
    def validator(data):
        return (count_issues({'jobs': 2}, data, soft={'jobs'})
                + ([FakeIssue('created')] if not data.get('created') else []))
    data, issues, rounds = await correct(
        runner, execution(over, misc), validator=validator,
        payload='p', scheduler=scheduler())
    assert [c['scope'] for c in runner.calls] == ['材料']  # misc rerun only
    assert data.get('created') == '2026-01-01'
    assert [(i.path, i.report_only) for i in issues] == [('jobs', True)]


def _needs_second_job(data):
    """One missing entry fires one correction round (FakeIssue, not
    count_issues: the generic-issue routing is what reaches the patch
    round under test)."""
    if len(data.get('jobs') or []) < 2:
        return [FakeIssue('jobs[1]')]
    return []


async def test_patch_reply_is_applied_not_replaced():
    # the correction round asks for a JSON Patch against the previous
    # result; untouched entries cannot collapse in a rewrite
    runner = ScriptedRunner(agent_result([
        {'op': 'add', 'path': '/jobs/1', 'value': {'company': '阿里'}}]))
    data, issues, rounds = await correct(
        runner, execution(call('jobs', [{'company': '腾讯'}])),
        validator=_needs_second_job, payload='p', scheduler=scheduler())
    assert data == {'jobs': [{'company': '腾讯'}, {'company': '阿里'}]}
    assert issues == [] and len(rounds) == 1
    rerun = runner.calls[0]
    assert is_patch_round(rerun['result_schema'])
    assert 'RFC 6902' in rerun['feedback'][0].message


async def test_a_failing_patch_keeps_the_result_then_reasks_in_full():
    runner = ScriptedRunner(
        agent_result([{'op': 'add', 'path': '/nope/9', 'value': 1}]),
        agent_result([{'company': '腾讯'}, {'company': '阿里'}]))
    data, issues, rounds = await correct(
        runner, execution(call('jobs', [{'company': '腾讯'}])),
        validator=_needs_second_job, payload='p', scheduler=scheduler())
    assert data == {'jobs': [{'company': '腾讯'}, {'company': '阿里'}]}
    assert issues == []
    # round one's unapplicable patch kept the previous result; round two
    # re-asked in full (no patch schema, no protocol directive)
    assert not runner.calls[1]['result_schema'].get('anyOf')
    assert 'RFC 6902' not in runner.calls[1]['feedback'][0].message


# --- nested arrays: per-parent counts route into the lifted sub-array's calls ---

NESTED_UNITS = {u.path: u for u in decompose({
    'type': 'object',
    'properties': {
        'career': {'type': 'object', 'properties': {
            'jobs': {'type': 'array', 'description': '工作经历',
                     'items': {'type': 'object', 'properties': {
                         'company': {'type': 'string', 'description': '公司'},
                         'roles': {'type': 'array', 'description': '任职经历',
                                   'items': {'type': 'object', 'properties': {
                                       'title': {'type': 'string', 'description': '职位'},
                                   }}},
                     }}},
        }},
    },
})}


def test_count_issues_reconcile_a_lifted_sub_array_per_parent():
    data = {'career': {'jobs': [
        {'company': '甲', 'roles': [{'title': '工程师'}]},
        {'company': '乙'}]}}
    issues = count_issues({'career.jobs': 2, 'career.jobs[0].roles': 2}, data)
    assert [(i.path, i.expected, i.got) for i in issues] == \
        [('career.jobs[0].roles[1]', 2, 1)]


async def test_nested_shortfall_reruns_the_short_whole_call():
    runner = ScriptedRunner(agent_result([{'title': '工程师'}, {'title': '经理'}]))
    calls = [Call(NESTED_UNITS['career.jobs.roles'], None, 'whole', [1], 'r0',
                  agent_result([{'title': '工程师'}]), parent=0)]
    _, issues, rounds = await correct(
        runner, Execution({}, calls),
        validator=lambda d: count_issues({'career.jobs[0].roles': 2}, d),
        payload='p', scheduler=scheduler())
    assert issues == []
    assert [(r.unit_path, r.item) for r in rounds] == [('career.jobs.roles', None)]


async def test_nested_field_issue_reruns_only_that_sub_entry():
    runner = ScriptedRunner(agent_result({'title': '经理修复'}))
    calls = [
        Call(NESTED_UNITS['career.jobs.roles'], 0, 'per-item', [1], 'r0',
             agent_result({'title': '工程师'}), parent=0),
        Call(NESTED_UNITS['career.jobs.roles'], 1, 'per-item', [2], 'r1',
             agent_result({'title': ''}), parent=0),
    ]
    _, _, rounds = await correct(
        runner, Execution({}, calls),
        validator=lambda d: [FakeIssue('career.jobs[0].roles[1].title')],
        payload='p', scheduler=scheduler())
    assert runner.calls[0]['scope'] == 'r1'
    assert [(r.unit_path, r.item) for r in rounds] == [('career.jobs.roles', 1)]
