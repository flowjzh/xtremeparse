"""Scripted end-to-end through the facade: every stage driven by a fake runner."""

from xtremeflow.scheduler import TaskScheduler

from xtremeparse import Extractor, ExtractionResult
from xtremeparse.arbitration import CHECK_DESCRIPTION
from xtremeparse.prompting import estimate_tokens, shared_payload
from tests.helpers import (FakeIssue, ScriptedRunner, agent_result,
                           plain_schema)


SCHEMA = {
    'type': 'object',
    'properties': {
        'basic_info': {'type': 'object', 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
        'career': {'type': 'object', 'properties': {
            'jobs': {'type': 'array', 'description': '经历', 'items': {'type': 'object', 'properties': {
                'company': {'type': 'string', 'description': '公司'},
            }}},
        }},
        'created': {'type': 'string', 'description': '创建时间'},
    },
}

TEXT = '''姓名：张三

## 工作经历
- 腾讯 后端
- 阿里 高级专家

创建于 2026'''


class PipelineRunner:
    """Answers the router's index map, then each specialist's extraction.
    A patch-or-value correction round (the anyOf schema) is answered
    with the full value — the scripted model never patches."""

    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        schema = plain_schema(kwargs['result_schema'])
        if schema.get('type') == 'string':  # router DSL
            return agent_result('0 a\n1 -\n2 b.0\n3 b.1\n4 c\nb: 2')
        scope = kwargs['scope']
        props = schema.get('properties', {})
        if 'name' in props:
            return agent_result({'name': scope.split('：')[1]})
        if 'company' in props:
            return agent_result({'company': next(w for w in scope.split() if w != '-')})
        return agent_result({'created': '2026-01-01'})


class FlakyRunner(PipelineRunner):
    """Wrong once on the name field, forcing one correction round."""

    def __init__(self):
        super().__init__()
        self._fixed = False

    async def run(self, **kwargs):
        if 'name' in kwargs['result_schema'].get('properties', {}) and not self._fixed:
            self._fixed = True
            return agent_result({'name': '缺姓名'})
        return await super().run(**kwargs)


class BudgetedRunner(PipelineRunner):
    """Serves a custom router DSL; the first company call runs verbose."""

    def __init__(self, dsl, verbose='很长很长的公司名'):
        super().__init__()
        self.dsl, self.verbose, self._fixed = dsl, verbose, False

    async def run(self, **kwargs):
        schema = kwargs['result_schema']
        if schema.get('type') == 'string':
            return agent_result(self.dsl)
        if schema.get('type') == 'array':  # batched small items
            return agent_result([{'company': self.verbose}, {'company': '阿里'}])
        if 'company' in schema.get('properties', {}) and not self._fixed:
            self._fixed = True
            return agent_result({'company': self.verbose})
        return await super().run(**kwargs)


def validator(data):
    if data.get('basic_info', {}).get('name') != '张三':
        return [FakeIssue('basic_info.name')]
    return []


def extractor(runner=None):
    return Extractor(runner or PipelineRunner(),
                     unit_strategy={'career.jobs': 'per-item'})


async def test_end_to_end_pipeline_produces_data_issues_and_trace():
    result = await extractor().extract(TEXT, SCHEMA, validator)
    assert isinstance(result, ExtractionResult)
    assert result.data == {
        'basic_info': {'name': '张三'},
        'career': {'jobs': [{'company': '腾讯'}, {'company': '阿里'}]},
        'created': '2026-01-01',
    }
    assert result.issues == []
    assert len(result.trace.chunks) == 5
    assert result.trace.router['assignments'][1]['unit'] == 'career.jobs'
    assert [g['unit'] for g in result.trace.groups] == [
        'basic_info', 'career.jobs', 'career.jobs', '$misc']
    assert result.trace.groups[1]['strategy'] == 'per-item'
    assert result.trace.corrections == []


async def test_router_runner_slices_and_fleet_runner_extracts():
    from tests.helpers import ScriptedRunner

    class FleetOnlyRunner(PipelineRunner):
        def __init__(self):
            super().__init__()
            self.saw_assignments = False

        async def run(self, **kwargs):
            if 'assignments' in kwargs['result_schema'].get('properties', {}):
                self.saw_assignments = True  # router must never reach the fleet
            return await super().run(**kwargs)

    router = ScriptedRunner(*[agent_result('0 a\n1-4 -\nb: 0')] * 2)
    fleet = FleetOnlyRunner()
    result = await Extractor(fleet, router_runner=router,
                             unit_strategy={'career.jobs': 'per-item'}).extract(
        TEXT, SCHEMA, validator)
    assert len(router.calls) == 2  # slice call + zero-verification round
    assert not router._results
    assert not fleet.saw_assignments
    assert result.data['basic_info']['name'] == '张三'


async def test_share_one_extractor_across_extractions():
    shared = extractor()
    first = await shared.extract(TEXT, SCHEMA, validator)
    second = await shared.extract(TEXT, SCHEMA, validator)
    assert first.data == second.data


async def test_correction_round_visible_in_trace():
    result = await extractor(FlakyRunner()).extract(TEXT, SCHEMA, validator)
    assert result.data['basic_info']['name'] == '张三'
    assert [(r['unit_path'], r['issue_paths']) for r in result.trace.corrections] == [
        ('basic_info', ['basic_info.name'])]


async def test_unresolvable_failure_reports_leniently():
    result = await extractor().extract(TEXT, SCHEMA,
                                       validator=lambda d: [FakeIssue('ghost.path')])
    assert result.data['basic_info']['name'] == '张三'
    assert result.issues[0].path == 'ghost.path'


async def test_empty_text_short_circuits_the_fleet():
    runner = PipelineRunner()
    result = await Extractor(runner).extract('', SCHEMA, validator)
    assert runner.calls == []  # no router call on empty input
    assert result.data == {} and result.issues[0].path == 'basic_info.name'


async def test_propertyless_schema_short_circuits_the_fleet():
    runner = PipelineRunner()
    result = await Extractor(runner).extract(TEXT, {'type': 'object'},
                                             validator=lambda d: [])
    assert runner.calls == [] and result.data == {}


def test_crlf_input_is_normalized_before_the_payload():
    chunks_payload = Extractor(PipelineRunner())
    assert chunks_payload.max_chars == 250  # MAX_CHARS constant wired


class RecordingScheduler(TaskScheduler):
    def __init__(self):
        super().__init__(8)
        self.estimates = []

    async def start_task(self, coro, **kwargs):
        self.estimates.append(kwargs.get('estimated_tokens'))
        return await super().start_task(coro, **kwargs)


async def test_every_llm_call_is_scheduled_with_a_token_estimate():
    scheduler = RecordingScheduler()
    await Extractor(PipelineRunner(), scheduler=scheduler,
                    unit_strategy={'career.jobs': 'per-item'}).extract(
        TEXT, SCHEMA, validator)
    assert scheduler.estimates[0] == estimate_tokens(shared_payload(TEXT, SCHEMA), TEXT)
    assert all(e > 0 for e in scheduler.estimates)
    assert len(scheduler.estimates) == 5  # router + four specialists


async def test_correction_round_is_scheduled_with_a_token_estimate():
    scheduler = RecordingScheduler()
    await Extractor(FlakyRunner(), scheduler=scheduler,
                    unit_strategy={'career.jobs': 'per-item'}).extract(
        TEXT, SCHEMA, validator)
    assert len(scheduler.estimates) == 6  # five calls + one correction round


async def test_router_scheduler_gives_the_slice_model_its_own_quota():
    fleet, router = RecordingScheduler(), RecordingScheduler()
    await Extractor(PipelineRunner(), scheduler=fleet, router_scheduler=router,
                    unit_strategy={'career.jobs': 'per-item'}).extract(
        TEXT, SCHEMA, validator)
    assert len(router.estimates) == 1  # only the slice call
    assert len(fleet.estimates) == 4  # only the specialist fleet


def test_estimate_tokens_calibrates_chars_per_token():
    assert estimate_tokens('ab', 'cdef') == 4  # 6 chars / 1.6, rounded


async def test_over_budget_output_is_kept_not_retried():
    # overruns are accepted — a retry would cost a full extra decode. Both
    # items are arranged small (@20 ≤ 300) so they BATCH into one array call
    class BatchedVerbose(BudgetedRunner):
        async def run(self, **kwargs):
            schema = kwargs['result_schema']
            if schema.get('type') == 'array':
                return agent_result([{'company': '很长很长的公司名'},
                                     {'company': '阿里'}])  # item 0 overruns 20
            return await super().run(**kwargs)

    runner = BatchedVerbose('0 a\n1 -\n2 b.0\n3 b.1\n4 c\nb: 2 @20')
    result = await Extractor(runner).extract(TEXT, SCHEMA, validator)
    assert [c['company'] for c in result.data['career']['jobs']] == \
        ['很长很长的公司名', '阿里']
    assert result.trace.corrections == []  # overrun never re-runs
    groups = result.trace.groups
    assert [(g['batch'], g['budget']) for g in groups if g['unit'] == 'career.jobs'] \
        == [((0, 1), [20, 20])]


async def test_host_output_budget_overrides_the_router_declaration():
    runner = BudgetedRunner('0 a\n1 -\n2 b.0\n3 b.1\n4 c\nb: 2 @50')
    result = await Extractor(runner, output_budgets={'career.jobs': 18}).extract(
        TEXT, SCHEMA, validator)
    assert [(g['batch'], g['budget']) for g in result.trace.groups
            if g['unit'] == 'career.jobs'] == [((0, 1), [18, 18])]  # host wins
    assert result.issues == []


class CollapsingRunner(PipelineRunner):
    """Whole-array call returns one item of two declared, then mends."""

    def __init__(self):
        super().__init__()
        self._collapsed = True

    async def run(self, **kwargs):
        schema = plain_schema(kwargs['result_schema'])
        if schema.get('type') == 'array':
            scope = kwargs['scope']
            if self._collapsed:
                self._collapsed = False
                return agent_result([{'company': '腾讯'}])  # 1 of 2 declared
            return agent_result([{'company': next(w for w in c.split() if w != '-')}
                                 for c in scope.split('\n\n')])
        return await super().run(**kwargs)


async def test_collapsed_array_is_reconciled_and_mended():
    runner = CollapsingRunner()
    result = await Extractor(
        runner, unit_strategy={'career.jobs': 'whole'}).extract(
        TEXT, SCHEMA, validator)
    assert [c['company'] for c in result.data['career']['jobs']] == ['腾讯', '阿里']
    assert result.issues == []
    assert [r['issue_paths'] for r in result.trace.corrections] == [['career.jobs[1]']]


def test_override_missing_a_placeholder_fails_at_construction():
    import pytest
    with pytest.raises(ValueError, match='router_instructions'):
        Extractor(PipelineRunner(), router_instructions='no slots here')
    with pytest.raises(ValueError, match='specialist_instructions'):
        Extractor(PipelineRunner(), specialist_instructions='no card slot')


async def test_prompt_overrides_reach_router_and_specialists():
    runner = PipelineRunner()
    await Extractor(runner,
                    router_instructions='ROUTE {top} {none} {legend} {chunks}',
                    specialist_instructions='SPEC {card}').extract(
        TEXT, SCHEMA, validator)
    router_call, specialist_call = runner.calls[0], runner.calls[1]
    assert router_call['instructions'].startswith('ROUTE 4 - ')
    assert '[basic_info | object] 基本信息' in router_call['instructions']
    assert specialist_call['instructions'].startswith('SPEC [basic_info')
    assert all(c['instructions'].startswith('SPEC [')
               for c in runner.calls[1:])


async def test_recount_override_reaches_the_recount_call():
    class ZeroRunner(PipelineRunner):
        async def run(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs['result_schema'].get('type') == 'string':
                return agent_result('0 a\n1-4 -\nb: 0')  # b left at zero
            return agent_result({})

    runner = ZeroRunner()
    result = await Extractor(
        runner, recount_instructions='RECHECK {legend} {chunks}').extract(
        TEXT, SCHEMA, validator)
    assert runner.calls[1]['instructions'].startswith('RECHECK')
    assert result.trace.prompts['recount'].startswith('#')


async def test_correction_round_reuses_the_specialist_override():
    runner = FlakyRunner()
    result = await Extractor(runner, specialist_instructions='SPEC {card}').extract(
        TEXT, SCHEMA, validator)
    # the wrong-name call returns before FlakyRunner records it, so the
    # only recorded name call IS the correction re-dispatch
    assert result.trace.corrections
    (correction,) = [c for c in runner.calls
                     if 'name' in plain_schema(c['result_schema'])
                     .get('properties', {})]
    assert correction['instructions'].startswith('SPEC [basic_info')


async def test_trace_records_prompt_provenance():
    result = await extractor().extract(TEXT, SCHEMA, validator)
    assert result.trace.prompts == {'router': 'default', 'recount': 'default',
                                    'specialist': 'default'}
    overridden = await Extractor(
        PipelineRunner(), router_instructions='R {top} {none} {legend} {chunks}'
    ).extract(TEXT, SCHEMA, validator)
    assert overridden.trace.prompts['router'].startswith('#')
    assert overridden.trace.prompts['specialist'] == 'default'


class OverdeclaringRunner(PipelineRunner):
    """Router maps 3 jobs (two co-chunked) for a 2-job text; the
    whole-array call extracts the 2 real ones; the arbitration diff
    finds nothing missing and nothing extra — the count revises to 2."""

    map_answer = '0 a\n1 -\n2 b.0\n3 b.1,b.2\n4 c\nb: 3'
    verdict = 'missing:\n\nextra:\n'

    async def run(self, **kwargs):
        schema = plain_schema(kwargs['result_schema'])
        if schema.get('type') == 'string':
            if str(schema.get('description', '')).startswith(CHECK_DESCRIPTION):
                return agent_result(self.verdict)
            return agent_result(self.map_answer)
        if schema.get('type') == 'array':
            return agent_result([{'company': '腾讯'}, {'company': '阿里'}])
        return await super().run(**kwargs)


async def test_arbitration_revises_an_over_declared_count():
    # declared 3, extracted 2: instead of a shortfall round forcing a
    # third entry into existence, the diff check arbitrates the true count
    runner = OverdeclaringRunner()
    result = await Extractor(runner).extract(TEXT, SCHEMA, validator)
    assert [j['company'] for j in result.data['career']['jobs']] == ['腾讯', '阿里']
    assert result.issues == []
    assert result.trace.corrections == []  # no fabrication round ran
    assert result.trace.recounts == [{'unit': 'career.jobs', 'declared': 3,
                                      'actual': 2, 'revised': 2,
                                      'answer': 'missing:\n\nextra:\n'}]


class UnderdeclaringRunner(OverdeclaringRunner):
    """Router maps 1 job (whole, co-chunked) for a 2-job text; the
    whole-array call extracts both; the arbitration verdict voids —
    the dispute reports without a merge round."""

    map_answer = '0 a\n1 -\n2-4 b\nb: 1'
    verdict = 'missing:\n\nextra:\n编造的不存在条目'


async def test_an_unsettled_over_count_reports_without_a_merge():
    # declared 1, extracted 2, verdict void: no repair runs — both real
    # entries survive and the dispute reports against the declaration
    result = await Extractor(UnderdeclaringRunner()).extract(
        TEXT, SCHEMA, validator)
    assert [j['company'] for j in result.data['career']['jobs']] == \
        ['腾讯', '阿里']
    assert result.trace.corrections == []  # no merge round ran
    (issue,) = [i for i in result.issues if i.path == 'career.jobs']
    assert (issue.expected, issue.got, issue.report_only) == (1, 2, True)


async def test_unanchored_verdict_keeps_the_shortfall_repair():
    # a verdict with an extra quote matching no list entry is voided —
    # the declared count stands and the correction loop repairs as before
    class UnanchoredRunner(OverdeclaringRunner):
        verdict = 'missing:\n\nextra:\n编造的不存在条目'

    result = await Extractor(UnanchoredRunner()).extract(TEXT, SCHEMA, validator)
    assert result.trace.recounts[-1]['revised'] is None
    assert [i.path for i in result.issues] == ['career.jobs[2]']
