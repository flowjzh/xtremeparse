"""Rubric judging through the neutral runner protocol."""

import pytest

from tests.helpers import ScriptedRunner, agent_result
from xtremeparse.judging import judge


async def test_judge_calls_the_runner_with_rubric_and_materials():
    runner = ScriptedRunner(agent_result({'ok': True, 'reason': 'all facts kept'}))
    verdict = await judge(runner, rubric='No facts lost.',
                          source='source text', output='{"a": 1}')
    assert (verdict.ok, verdict.reason) == (True, 'all facts kept')
    call = runner.calls[0]
    assert 'No facts lost.' in call['instructions']
    assert 'Source:\n\nsource text' in call['content']
    assert 'Output:\n\n{"a": 1}' in call['content']
    assert call['result_schema']['required'] == ['ok', 'reason']


async def test_judge_tolerates_a_shapeless_answer():
    runner = ScriptedRunner(agent_result(None))
    verdict = await judge(runner, rubric='R', source='', output='')
    assert (verdict.ok, verdict.reason) == (False, '')


async def test_judge_instructions_are_overridable():
    runner = ScriptedRunner(agent_result({'ok': False, 'reason': 'r'}))
    await judge(runner, rubric='R', source='', output='',
                instructions='JUDGE says {rubric}')
    assert runner.calls[0]['instructions'] == 'JUDGE says R'


async def test_judge_override_requires_the_rubric_placeholder():
    with pytest.raises(ValueError, match='instructions'):
        await judge(None, rubric='R', source='', output='',
                    instructions='no slot here')
