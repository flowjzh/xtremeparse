"""Boundary probes for the executor and merge: strategies, concurrency, assembly."""

import asyncio

import pytest

from xtremeflow.scheduler import TaskScheduler

from xtremeparse.executor import execute
from xtremeparse.merge import merge
from xtremeparse.prompting import SPECIALIST_INSTRUCTIONS
from xtremeparse.router import Group, Routing
from xtremeparse.units import MISC, decompose
from tests.helpers import ScriptedRunner, agent_result

SCHEMA = {
    'type': 'object',
    'properties': {
        'basic_info': {'type': 'object', 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
        'career': {'type': 'object', 'properties': {
            'jobs': {'type': 'array', 'description': '工作经历', 'items': {
                'type': 'object', 'properties': {
                    'company': {'type': 'string', 'description': '公司'},
                }}},
        }},
        'created': {'type': 'string', 'description': '创建时间'},
    },
}
UNITS = {u.path: u for u in decompose(SCHEMA)}


class ProbeRunner:
    """Dispatches on result_schema shape; tracks peak concurrency."""

    def __init__(self):
        self.calls = []
        self.active = self.max_active = 0

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        schema = kwargs['result_schema']
        if schema.get('type') == 'array':
            return agent_result([{'company': c} for c in kwargs['scope'].split('\n\n')])
        return agent_result({k: kwargs['scope'] for k in schema.get('properties', {})})


def routing(*groups):
    return Routing(list(groups), {'assignments': []})


def group(path, item=None, text='x', ids=(0,)):
    return Group(UNITS[path], item, list(ids), text)


async def test_golden_shape_fans_out_per_item_and_nests_on_merge():
    routing_ = routing(
        group('basic_info', text='姓名张三'),
        group('career.jobs', item=0, text='腾讯', ids=(1,)),
        group('career.jobs', item=1, text='阿里', ids=(2,)),
        group(MISC, text='2026'),
    )
    execution = await execute(ProbeRunner(), routing_, payload='p',
                              scheduler=TaskScheduler(8))
    assert len(execution.calls) == 4  # 1 object + 2 items + 1 misc
    assert execution.values['career.jobs'] == [{'company': '腾讯'}, {'company': '阿里'}]
    assert merge(execution.values) == {
        'basic_info': {'name': '姓名张三'},
        'career': {'jobs': [{'company': '腾讯'}, {'company': '阿里'}]},
        'created': '2026',
    }


async def test_short_array_runs_whole_with_wrapped_schema():
    routing_ = routing(group('career.jobs', item=0, text='短'))
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p', scheduler=TaskScheduler(2))
    assert [c.strategy for c in execution.calls] == ['whole']
    call = runner.calls[0]
    assert call['result_schema'] == {'type': 'array', 'items': UNITS['career.jobs'].sub_schema}
    assert execution.values['career.jobs'] == [{'company': '短'}]


async def test_separable_items_fan_out_even_when_small():
    # decode time is output-bound: parallel small streams beat one long
    # one, so short material no longer buys whole-array
    routing_ = routing(group('career.jobs', item=0, text='短', ids=[0]),
                       group('career.jobs', item=1, text='更短', ids=[1]))
    execution = await execute(ProbeRunner(), routing_, payload='p',
                              scheduler=TaskScheduler(2))
    assert [c.strategy for c in execution.calls] == ['per-item', 'per-item']
    assert [c.item for c in execution.calls] == [0, 1]


async def test_ranged_blocks_and_budget_batches_wear_distinct_labels():
    # the model's ranged block is its own call shape — labeled 'ranged',
    # not borrowed from the strategy; the executor's own budget batching
    # of plain items stays 'per-item'
    ranged = Group(UNITS['career.jobs'], None, [0, 1, 2], 'a\nb', items=(0, 2))
    plain = [Group(UNITS['career.jobs'], i, [i], 't') for i in (3, 4)]
    execution = await execute(ProbeRunner(), routing(ranged, *plain),
                              payload='p', scheduler=TaskScheduler(2),
                              budgets={'career.jobs': 10})
    assert [c.strategy for c in execution.calls] == ['ranged', 'per-item']
    assert execution.calls[0].batch == (0, 1, 2) \
        and execution.calls[0].budget == [10, 10, 10]
    assert execution.calls[1].batch == (3, 4)


async def test_cochunked_items_force_whole_and_deduplicate_material():
    # several items routed to the same chunks: one whole call, the
    # shared material extracted once — not repeated per item
    routing_ = routing(group('career.jobs', item=0, text='一段两个学历', ids=[3]),
                       group('career.jobs', item=1, text='一段两个学历', ids=[3]))
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p',
                              scheduler=TaskScheduler(2))
    assert [c.strategy for c in execution.calls] == ['whole']
    assert len(runner.calls) == 1
    assert runner.calls[0]['scope'] == '一段两个学历'
    assert execution.calls[0].chunk_ids == [3]
    assert execution.values['career.jobs'] == [{'company': '一段两个学历'}]


async def test_unit_strategy_overrides_both_ways():
    long_routing = routing(group('career.jobs', item=0, text='长' * 5000))
    execution = await execute(ProbeRunner(), long_routing, payload='p',
                              scheduler=TaskScheduler(2),
                              unit_strategy={'career.jobs': 'whole'})
    assert [c.strategy for c in execution.calls] == ['whole']

    short_routing = routing(group('career.jobs', item=0, text='短'))
    execution = await execute(ProbeRunner(), short_routing, payload='p',
                              scheduler=TaskScheduler(2),
                              unit_strategy={'career.jobs': 'per-item'})
    assert [c.strategy for c in execution.calls] == ['per-item']


async def test_scheduler_concurrency_cap_is_respected():
    routing_ = routing(*[group('career.jobs', item=i, text=f'经历{i}', ids=[i])
                         for i in range(6)])
    runner = ProbeRunner()
    await execute(runner, routing_, payload='p', scheduler=TaskScheduler(2))
    assert len(runner.calls) == 6
    assert runner.max_active <= 2


async def test_specialist_calls_carry_card_scope_and_shared_payload():
    routing_ = routing(group('basic_info', text='张三'))
    runner = ProbeRunner()
    await execute(runner, routing_, payload='shared', scheduler=TaskScheduler(1))
    call = runner.calls[0]
    assert '[basic_info | object] 基本信息' in call['instructions']
    assert call['scope'] == '张三'
    assert call['content'] == 'shared'
    assert call.get('history') is None and call.get('feedback') is None


async def test_object_unit_groups_coalesce_into_one_call():
    routing_ = routing(group('basic_info', text='a', ids=(0,)),
                       group('basic_info', text='b', ids=(1,)))
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p', scheduler=TaskScheduler(2))
    assert len(runner.calls) == 1
    assert runner.calls[0]['scope'] == 'a\n\nb'
    assert execution.calls[0].chunk_ids == [0, 1]


async def test_empty_routing_makes_no_calls():
    execution = await execute(ProbeRunner(), routing(), payload='p',
                              scheduler=TaskScheduler(2))
    assert execution.values == {}
    assert merge(execution.values) == {}


async def test_none_results_stay_absent_everywhere():
    execution = await execute(ScriptedRunner(agent_result(None)),
                              routing(group('basic_info')),
                              payload='p', scheduler=TaskScheduler(1))
    assert execution.values == {}
    assert merge(execution.values) == {}


def test_merge_scatter_dotted_and_none():
    values = {MISC: {'created': 'today', 'extra': None},
              'career.jobs': [1], 'top': None}
    assert merge(values) == {'created': 'today', 'career': {'jobs': [1]}}


def test_merge_is_order_independent_for_sibling_units():
    array_first = {'career.jobs': [{'company': '腾讯'}], 'career': {'title': '工程师'}}
    object_first = {'career': {'title': '工程师'}, 'career.jobs': [{'company': '腾讯'}]}
    assert merge(array_first) == merge(object_first) \
        == {'career': {'title': '工程师', 'jobs': [{'company': '腾讯'}]}}


async def test_duplicate_item_rows_coalesce_into_one_call():
    routing_ = routing(
        group('career.jobs', item=0, text='腾讯前半', ids=(1,)),
        group('career.jobs', item=0, text='腾讯后半', ids=(2,)),
    )
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p',
                              scheduler=TaskScheduler(2))
    assert len(runner.calls) == 1
    assert runner.calls[0]['scope'] == '腾讯前半\n\n腾讯后半'
    assert execution.calls[0].chunk_ids == [1, 2]


async def test_unknown_strategy_override_raises():
    with pytest.raises(ValueError):
        await execute(ProbeRunner(), routing(group('career.jobs')),
                      payload='p', scheduler=TaskScheduler(1),
                      unit_strategy={'career.jobs': 'chunked'})


async def test_budget_flows_into_every_item_call():
    routing_ = routing(
        group('career.jobs', item=0, text='腾讯', ids=(1,)),
        group('career.jobs', item=1, text='阿里', ids=(2,)))
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p',
                              scheduler=TaskScheduler(8),
                              budgets={'career.jobs': 400})
    assert [c.budget for c in execution.calls] == [400, 400]
    assert all('Output budget' not in c['instructions'] for c in runner.calls)


async def test_per_item_budgets_reach_their_own_calls():
    routing_ = routing(
        group('career.jobs', item=0, text='腾讯', ids=(1,)),
        group('career.jobs', item=1, text='阿里', ids=(2,)))
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p',
                              scheduler=TaskScheduler(8),
                              budgets={'career.jobs': [400, 500]})
    assert [c.budget for c in execution.calls] == [400, 500]
    assert all('Output budget' not in c['instructions'] for c in runner.calls)


async def test_whole_array_budget_says_each_item():
    runner = ProbeRunner()
    await execute(runner, routing(group('career.jobs', item=0, text='短')),
                  payload='p', scheduler=TaskScheduler(8),
                  budgets={'career.jobs': 50})
    assert 'Output budget' not in runner.calls[0]['instructions']


async def test_whole_array_per_item_budgets_list_their_own():
    runner = ProbeRunner()
    await execute(runner, routing(group('career.jobs', item=0, text='短')),
                  payload='p', scheduler=TaskScheduler(8),
                  budgets={'career.jobs': [500, 300]})
    assert 'Output budget' not in runner.calls[0]['instructions']


async def test_budget_reaches_non_array_units_too():
    runner = ProbeRunner()
    await execute(runner, routing(group('basic_info', text='姓名张三')),
                  payload='p', scheduler=TaskScheduler(8),
                  budgets={'basic_info': 80})
    assert 'Output budget' not in runner.calls[0]['instructions']


async def test_no_budget_leaves_instructions_untouched():
    runner = ProbeRunner()
    await execute(runner, routing(group('basic_info', text='短')),
                  payload='p', scheduler=TaskScheduler(8))
    assert runner.calls[0]['instructions'] == SPECIALIST_INSTRUCTIONS.format(
        card=UNITS['basic_info'].card)


def test_default_specialist_prompt_carries_all_placeholders():
    from xtremeparse.prompting import SPECIALIST_PLACEHOLDERS, check_placeholders
    check_placeholders(SPECIALIST_INSTRUCTIONS, SPECIALIST_PLACEHOLDERS,
                       'specialist')


async def test_capacity_is_accumulated_not_per_item():
    # three 300-budget items fill the 300 capacity each — three parallel
    # streams, never one 900-budget call
    routing_ = routing(*[group('career.jobs', item=i, text=f'经历{i}', ids=[i])
                         for i in range(3)])
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p',
                              scheduler=TaskScheduler(8),
                              budgets={'career.jobs': [300, 300, 300]})
    assert [(c.item, c.batch) for c in execution.calls] == [(0, None), (1, None), (2, None)]


async def test_two_small_items_share_the_capacity():
    routing_ = routing(*[group('career.jobs', item=i, text=f'经历{i}', ids=[i])
                         for i in range(2)])
    runner = ProbeRunner()
    execution = await execute(runner, routing_, payload='p',
                              scheduler=TaskScheduler(8),
                              budgets={'career.jobs': [150, 150]})
    calls = [c for c in execution.calls]
    assert len(calls) == 1 and calls[0].batch == (0, 1) and calls[0].budget == [150, 150]


async def test_array_shaped_calls_carry_the_no_merge_addendum():
    # a whole array call (single unsplit group) reads the addendum; an
    # object-shaped call (basic_info, or a multi-item per-item fan-out) does not
    for routings, expect_addendum in [
            ([group('career.jobs', text='腾讯 阿里', ids=(1,))], True),
            ([group('basic_info', text='姓名张三')], False),
            ([group('career.jobs', item=0, text='腾讯', ids=(1,)),
              group('career.jobs', item=1, text='阿里', ids=(2,))], False)]:
        probe = ProbeRunner()
        await execute(probe, routing(*routings), payload='p',
                      scheduler=TaskScheduler(8))
        assert all(('never merge' in c['instructions']) == expect_addendum
                   for c in probe.calls)


# --- nested arrays: a lifted sub-array fans out per parent and grafts ---

NESTED_SCHEMA = {
    'type': 'object',
    'properties': {
        'basic_info': {'type': 'object', 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
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
        'created': {'type': 'string', 'description': '创建时间'},
    },
}
NESTED_UNITS = {u.path: u for u in decompose(NESTED_SCHEMA)}


def nested_group(path, item=None, text='x', ids=(0,), parent=None, items=()):
    return Group(NESTED_UNITS[path], item, list(ids), text, items=items,
                 parent=parent)


async def test_lifted_sub_array_fans_out_per_parent_and_grafts():
    routing_ = routing(
        group('basic_info', text='姓名张三'),
        group('career.jobs', item=0, text='公司甲', ids=(1, 2)),
        group('career.jobs', item=1, text='公司乙', ids=(3,)),
        nested_group('career.jobs.roles', item=0, text='工程师', ids=(1,), parent=0),
        nested_group('career.jobs.roles', item=1, text='经理', ids=(2,), parent=0),
    )
    execution = await execute(ProbeRunner(), routing_, payload='p',
                              scheduler=TaskScheduler(8))
    assert execution.values['career.jobs[0].roles'] == [
        {'title': '工程师'}, {'title': '经理'}]
    data = merge(execution.values)
    assert data['career']['jobs'][0]['roles'] == [
        {'title': '工程师'}, {'title': '经理'}]
    assert data['career']['jobs'][0]['company'] == '公司甲'
    assert data['career']['jobs'][1] == {'company': '公司乙'}


async def test_merge_grafts_a_lifted_sub_array_in_either_arrival_order():
    parent_first = merge({'career.jobs': [{'company': '公司甲'}, {'company': '公司乙'}],
                          'career.jobs[0].roles': [{'title': '工程师'}]})
    child_first = merge({'career.jobs[0].roles': [{'title': '工程师'}],
                         'career.jobs': [{'company': '公司甲'}, {'company': '公司乙'}]})
    expected = {'career': {'jobs': [
        {'company': '公司甲', 'roles': [{'title': '工程师'}]},
        {'company': '公司乙'}]}}
    assert parent_first == child_first == expected
