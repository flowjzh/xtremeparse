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


def kw_result(groups, data, declared):
    return SimpleNamespace(data=data, trace=SimpleNamespace(
        router={'budgets': declared}, groups=groups))


def test_budget_fit_judges_each_item_not_the_mean():
    # one item dead-on, one at 5x: the mean (3.0) reads mild — the item may not
    spread = budget_result([{'unit': 'career.jobs', 'budget': 20, 'item': None}],
                           {'career': {'jobs': [{'company': '十' * 10},
                                                {'company': '十' * 100}]}})
    verdict = BudgetFit().evaluate(ctx(spread))
    assert verdict.value is False
    assert 'jobs[1]' in verdict.reason and '5.0' in verdict.reason


def test_budget_fit_judges_a_per_item_budget_list():
    # item 0 dead-on its own 10, item 1 far over its own 20 (5x)
    result = budget_result([{'unit': 'career.jobs', 'budget': [10, 20], 'item': None}],
                           {'career': {'jobs': [{'company': '十' * 10},
                                                {'company': 'x' * 100}]}})
    verdict = BudgetFit().evaluate(ctx(result))
    assert verdict.value is False and 'jobs[1]' in verdict.reason


def test_budget_fit_judges_a_whole_run_on_its_total():
    # a whole-strategy run's entries are slices of shared material; a
    # merged item concentrating a sub-array's values breaks item-wise
    # pairing (60/25 = 2.4) — the total (25+25) against the array's
    # total (60+10) is the run's real reading
    result = budget_result([{'unit': 'career.jobs', 'budget': [25, 25],
                             'item': None, 'strategy': 'whole'}],
                           {'career': {'jobs': [{'company': 'x' * 10,
                                                 'positions': [{'title': 'y' * 50}]},
                                                {'company': 'z' * 10}]}})
    verdict = BudgetFit().evaluate(ctx(result))
    assert verdict.value is True, verdict.reason


def test_budget_fit_judges_a_whole_runs_lone_number_on_its_total():
    # one shared number covers the whole run — pairing it per item
    # flags every item below the sub-heavy one (0.11-0.64); the run's
    # total (929 vs 824) is the declaration's real reading
    result = budget_result([{'unit': 'career.jobs', 'budget': 929,
                             'item': None, 'strategy': 'whole'}],
                           {'career': {'jobs': [{'company': 'x' * 591},
                                                {'company': 'y' * 133},
                                                {'company': 'z' * 100}]}})
    assert BudgetFit().evaluate(ctx(result)).value is True


def test_budget_fit_realigns_arrangements_with_the_hosts_sort():
    # the host sorted [A, B] into [B, A]: pairing sorted position i
    # with budget entry i reads A's 60 chars against B's 10 and flags
    # both — perm [1, 0] pairs each item with its own arrangement
    out = SimpleNamespace(
        data={'career': {'jobs': [{'company': 'x' * 10},
                                  {'company': 'y' * 60}]}},
        trace=SimpleNamespace(router={}, groups=[
            {'unit': 'career.jobs', 'budget': 60, 'item': 0},
            {'unit': 'career.jobs', 'budget': 10, 'item': 1}]),
        sort={'career.jobs': [1, 0]})
    assert BudgetFit().evaluate(ctx(out)).value is True


def test_budget_fit_band_and_clean_pass():
    close = budget_result([{'unit': 'career.jobs', 'budget': 2, 'item': 0}],
                          {'career': {'jobs': [{'company': '腾讯'}]}})  # 2/2
    tight = budget_result([{'unit': 'career.jobs', 'budget': 3, 'item': 0}],
                          {'career': {'jobs': [{'company': '腾讯集团云计算'}]}})  # 7/3
    loose = budget_result([{'unit': 'career.jobs', 'budget': 200, 'item': 0}],
                          {'career': {'jobs': [{'company': '腾讯'}]}})  # 2/200
    unbudgeted = budget_result([{'unit': 'career.jobs', 'budget': None, 'item': 0}], {})
    assert BudgetFit().evaluate(ctx(close)).value is True
    # the 4-char overshoot is an entry's fixed field cost — slack absorbs it
    assert BudgetFit().evaluate(ctx(tight)).value is True
    # both edges validated: the under-run misses the band by 198 — flagged
    assert BudgetFit().evaluate(ctx(loose)).value is False
    assert 'no budgets declared' in BudgetFit().evaluate(ctx(unbudgeted)).reason


def test_budget_fit_slack_absorbs_field_cost_both_edges():
    small = budget_result([{'unit': 'career.jobs', 'budget': 20, 'item': 0}],
                          {'career': {'jobs': [{'company': '十' * 42}]}})  # 2.1x, diff 22
    assert BudgetFit().evaluate(ctx(small)).value is True  # field-cost slack
    big = budget_result([{'unit': 'career.jobs', 'budget': 20, 'item': 0}],
                        {'career': {'jobs': [{'company': '十' * 100}]}})  # 5x, diff 80
    assert BudgetFit().evaluate(ctx(big)).value is False
    under = budget_result([{'unit': 'career.jobs', 'budget': 200, 'item': 0}],
                          {'career': {'jobs': [{'company': '腾讯'}]}})  # 2/200
    assert BudgetFit().evaluate(ctx(under)).value is False  # lower edge validated


def test_budget_fit_judges_a_keyword_form_on_its_total():
    # uneven keyword lengths are the norm: 45+5+10 = 60 against a
    # declared 20x3 (60) — per-item shares (20 each) would flag two
    # of the three items, the total reads 1.0
    groups = [{'unit': 'certs', 'budget': [20, 20, 20], 'item': None,
               'batch': [0, 1, 2]}]
    data = {'certs': ['x' * 45, 'x' * 5, 'x' * 10]}
    verdict = BudgetFit().evaluate(ctx(kw_result(groups, data, {'certs': '20x3'})))
    assert verdict.value is True, verdict.reason
    # a declared total the entries overrun 4x fails: 240/60 = 4.0,
    # diff 180 — past ratio and slack both; so does a 4x under-run
    # (15/60 = 0.25, diff 45)
    long = {'certs': ['x' * 80, 'x' * 80, 'x' * 80]}
    verdict = BudgetFit().evaluate(ctx(kw_result(groups, long, {'certs': '20x3'})))
    assert verdict.value is False and '4.0' in verdict.reason
    short = {'certs': ['x' * 5, 'x' * 5, 'x' * 5]}
    verdict = BudgetFit().evaluate(ctx(kw_result(groups, short, {'certs': '20x3'})))
    assert verdict.value is False and '0.25' in verdict.reason


def test_budget_fit_judges_each_item_against_its_own_group_budget():
    # the corpus shape: per-item groups each carrying a different number —
    # a unit-keyed dict would judge every item by the LAST group's budget.
    # The unit is value characters: a job's cost is its company's length.
    groups = [{'unit': 'career.jobs', 'budget': 40, 'item': 0},
              {'unit': 'career.jobs', 'budget': 700, 'item': 1},
              {'unit': 'career.jobs', 'budget': [50, 60], 'item': None,
               'batch': [2, 3]}]
    data = {'career': {'jobs': [
        {'company': '十' * 20},       # 20/40
        {'company': 'x' * 686},       # 686/700 own (0.98); 686/40 poisoned
        {'company': '十' * 20},       # 20/50
        {'company': '十' * 30}]}}     # 30/60
    verdict = BudgetFit().evaluate(ctx(budget_result(groups, data)))
    assert verdict.value is True, verdict.reason
    # the same items judged against one unit-wide number would fail
    poisoned = budget_result([{'unit': 'career.jobs', 'budget': 40, 'item': None}],
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
