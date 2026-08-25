"""pydantic-evals adapters over the trace reads (the ``evals`` extra)."""

from types import SimpleNamespace

from pydantic_evals import Case, Dataset

from tests.helpers import ScriptedRunner, agent_result
from xtremeparse.evals import BudgetFit, RouterOverlap, RubricJudge


def ctx(output):
    return SimpleNamespace(output=output)


def router_result(assignments, groups):
    return SimpleNamespace(trace=SimpleNamespace(router={'assignments': assignments},
                                                 groups=groups))


def budget_result(groups, data):
    return SimpleNamespace(data=data, trace=SimpleNamespace(router={}, groups=groups))


def test_budget_fit_judges_each_item_not_the_mean():
    # one item dead-on, one at 0.2: the mean (0.6) would pass — the item may not
    spread = budget_result([{'unit': 'career.jobs', 'budget': 200, 'item': None}],
                           {'career': {'jobs': [{'company': '十' * 10},
                                                {'company': '腾讯'}]}})
    verdict = BudgetFit().evaluate(ctx(spread))
    assert verdict.value is False
    assert 'jobs[1]' in verdict.reason and '0.08' in verdict.reason


def test_budget_fit_judges_a_per_item_budget_list():
    # item 0 at its own 12 (2.0, the upper edge), item 1 far under its own 200 (0.08)
    result = budget_result([{'unit': 'career.jobs', 'budget': [12, 200], 'item': None}],
                           {'career': {'jobs': [{'company': '十' * 10},
                                                {'company': '腾讯'}]}})
    verdict = BudgetFit().evaluate(ctx(result))
    assert verdict.value is False and 'jobs[1]' in verdict.reason


def test_budget_fit_band_and_clean_pass():
    close = budget_result([{'unit': 'career.jobs', 'budget': 8, 'item': 0}],
                          {'career': {'jobs': [{'company': '腾讯'}]}})  # 16/8
    tight = budget_result([{'unit': 'career.jobs', 'budget': 2, 'item': 0}],
                          {'career': {'jobs': [{'company': '腾讯集团云计算'}]}})  # 21/2
    loose = budget_result([{'unit': 'career.jobs', 'budget': 200, 'item': 0}],
                          {'career': {'jobs': [{'company': '腾讯'}]}})  # 16/200
    unbudgeted = budget_result([{'unit': 'career.jobs', 'budget': None, 'item': 0}], {})
    assert BudgetFit().evaluate(ctx(close)).value is True
    assert BudgetFit().evaluate(ctx(tight)).value is False
    assert BudgetFit().evaluate(ctx(loose)).value is False
    assert 'no budgets declared' in BudgetFit().evaluate(ctx(unbudgeted)).reason


def test_budget_fit_judges_each_item_against_its_own_group_budget():
    # the corpus shape: per-item groups each carrying a different number —
    # a unit-keyed dict would judge every item by the LAST group's budget.
    # Item JSON is {"company":"…"} = 14 + len(value) serialized chars.
    groups = [{'unit': 'career.jobs', 'budget': 80, 'item': 0},
              {'unit': 'career.jobs', 'budget': 600, 'item': 1},
              {'unit': 'career.jobs', 'budget': [70, 100], 'item': None,
               'batch': [2, 3]}]
    data = {'career': {'jobs': [
        {'company': '十' * 20},       # 34/80
        {'company': 'x' * 686},       # 700/600 own (1.17); 700/150 poisoned (4.67)
        {'company': '十' * 20},       # 34/70
        {'company': '十' * 30}]}}     # 44/100
    verdict = BudgetFit().evaluate(ctx(budget_result(groups, data)))
    assert verdict.value is True, verdict.reason
    # the same items judged against one unit-wide number would fail
    poisoned = budget_result([{'unit': 'career.jobs', 'budget': 150, 'item': None}],
                             data)
    assert BudgetFit().evaluate(ctx(poisoned)).value is False


async def test_rubric_judge_judges_the_data_against_the_source():
    runner = ScriptedRunner(agent_result({'ok': True, 'reason': 'facts intact'}))
    output = SimpleNamespace(data={'name': '张三'})
    context = SimpleNamespace(output=output)
    verdict = await RubricJudge(runner, 'Names match.',
                                source='姓名张三').evaluate(context)
    assert verdict.value is True and verdict.reason == 'facts intact'
    content = runner.calls[0]['content']
    assert '姓名张三' in content and '张三' in content
    assert 'Source:' in content and 'Output:' in content


async def test_rubric_judge_projects_the_output_before_judging():
    # a containment rubric judges a flattened digest — the projection
    # hook is what keeps the judge from mirroring field structure
    def flatten(v):
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return '\n'.join(map(flatten, v))
        if isinstance(v, dict):
            return '\n'.join(map(flatten, v.values()))
        return ''

    runner = ScriptedRunner(agent_result({'ok': True, 'reason': 'r'}))
    output = SimpleNamespace(data={'a': {'x': 'one'}, 'b': ['two']})
    await RubricJudge(runner, 'R',
                      output_of=lambda o: flatten(o.data)).evaluate(
        SimpleNamespace(output=output))
    content = runner.calls[0]['content']
    assert 'one\ntwo' in content and "'x'" not in content


def test_rubric_judge_validates_instruction_overrides_at_construction():
    import pytest
    with pytest.raises(ValueError, match='instructions'):
        RubricJudge(None, 'R', instructions='no slot here')


def test_evaluators_satisfy_the_pydantic_evals_contract():
    """The coercion layer real runs go through — the tuple-return bug
    class lives exactly here."""
    dataset = Dataset(name='contract', cases=[
        Case(name='ok', inputs=None, evaluators=(RouterOverlap(),),
             expected_output=None)])
    report = dataset.evaluate_sync(lambda _: router_result(
        [{'unit': 'a', 'chunks': [0]}],
        [{'unit': 'a', 'chunk_ids': [0], 'item': None, 'strategy': None}]))
    (case,) = report.cases
    assert case.evaluator_failures == []
    result = case.assertions['RouterOverlap']
    assert result.value is True and 'items_lost=0' in result.reason
