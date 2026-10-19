"""The verbatim contract: keyword detection, item checks, issues, budgets."""

import copy
from types import SimpleNamespace

from xtremeparse.router import Group
from xtremeparse.schema import heading_label
from xtremeparse.units import Unit, decompose
from xtremeparse.verbatim import (VERBATIM_DEFAULT_RATIO, check_item,
                                  find_violations, marker_ratio,
                                  normalize_text, verbatim_budgets,
                                  verbatim_issues, verbatim_paths)

SRC = ('### 汇报对象：总部 HRD\n·负责线下渠道的\n营销活动策划与执行。\n'
       '**关键成果** 达成率 105%。\n汇报对象调整需与总部对齐。')
SRC_NORM = normalize_text(SRC)
LABELS = frozenset(['汇报对象', '关键成果'])

SCHEMA = {
    'type': 'object',
    'properties': {
        'work_experiences': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'company': {'type': 'string', 'description': '公司名'},
                    'report_to': {'type': 'string', 'description': '汇报对象'},
                    'description': {
                        'type': 'array',
                        'description': '描述文字',
                        'x-verbatim': 0.9,
                        'items': {'type': 'string'},
                    },
                },
            },
        },
    },
}


def marked_schema(ratio: float) -> dict:
    schema = copy.deepcopy(SCHEMA)
    schema['properties']['work_experiences']['items']['properties'][
        'description']['x-verbatim'] = ratio
    return schema


def test_marker_ratio_reads_the_extension_keyword():
    assert marker_ratio({'x-verbatim': 0.85}) == 0.85
    assert marker_ratio({'x-verbatim': 0.9}) == VERBATIM_DEFAULT_RATIO
    # only the keyword arms enforcement; rendered @verbatim text is display-only
    assert marker_ratio({'description': '描述 @verbatim'}) is None
    assert marker_ratio({'description': 'plain'}) is None and marker_ratio({}) is None
    # a malformed value on a hand-written schema still arms the contract
    assert marker_ratio({'x-verbatim': 'abc'}) == VERBATIM_DEFAULT_RATIO
    assert marker_ratio({'x-verbatim': 0}) == VERBATIM_DEFAULT_RATIO


def test_normalize_text_strips_whitespace_and_markdown():
    assert normalize_text('**关键成果** 达成率\n105%。') == '关键成果达成率105%。'


def test_heading_label_splits_at_colon():
    assert heading_label({'type': 'string', 'description': '汇报对象：汇报线'}) == '汇报对象'
    assert heading_label({'type': 'object', 'description': 'x：y'}) is None


def test_check_item_accepts_healed_and_bolded_copies():
    assert check_item('负责线下渠道的营销活动策划与执行。', 'd[0]', LABELS,
                      SRC_NORM, 0.9) is None
    assert check_item('关键成果 达成率 105%。', 'd[1]', LABELS,
                      SRC_NORM, 0.9) is None  # bold stripped, label no colon


def test_check_item_flags_heading_and_paraphrase():
    heading = check_item('汇报对象：总部 HRD', 'd[2]', LABELS, SRC_NORM, 0.9)
    assert heading.code == 'heading' and heading.label == '汇报对象'
    para = check_item('负责线下渠道营销活动的统筹规划。', 'd[0]', LABELS, SRC_NORM, 0.9)
    assert para.code == 'verbatim' and para.coverage < 0.9


def test_check_item_noise_window_pards_interleaved_noise():
    noisy = normalize_text(
        SRC + '主要从事资产证券化业务，参与多个资产证 Ø 项目经验 券化项目的方案制作等工作。')
    item = '主要从事资产证券化业务，参与多个资产证券化项目的方案制作等工作。'
    assert normalize_text(item) not in noisy  # no contiguous copy — yet faithful
    assert check_item(item, 'd[0]', LABELS, noisy, 0.9) is None


def test_verbatim_paths_walks_nested_levels():
    assert verbatim_paths(SCHEMA) == ['work_experiences.description']
    assert verbatim_paths({'type': 'object'}) == []


def test_find_violations_walks_nested_arrays_with_scope_labels():
    data = {'work_experiences': [
        {'description': ['汇报对象：总部 HRD', '不是原文的句子。'],
         'report_to': '总部 HRD'},
        {'description': ['汇报对象调整需与总部对齐。']},
    ]}
    findings = find_violations(SCHEMA, data, SRC_NORM)
    assert [(f.path, f.code) for f in findings] == [
        ('work_experiences[0].description[0]', 'heading'),
        ('work_experiences[0].description[1]', 'verbatim'),
    ]


def test_marker_ratio_and_override_layering():
    item = '负责线下渠道的营销活动策划与执行。并统筹跨部门的资源调度安排。'
    data = {'work_experiences': [{'description': [item]}]}
    assert 0.4 < find_violations(SCHEMA, data, SRC_NORM)[0].coverage < 0.7
    assert find_violations(marked_schema(0.5), data, SRC_NORM) == []
    assert find_violations(SCHEMA, data, SRC_NORM, min_ratio=0.5) == []
    assert len(find_violations(marked_schema(0.5), data, SRC_NORM,
                               min_ratio=0.99)) == 1  # override wins


def test_marked_scalar_field_is_checked_too():
    schema = copy.deepcopy(SCHEMA)
    prop = schema['properties']['work_experiences']['items']['properties']
    prop['company'] = {'type': 'string', 'description': '公司名', 'x-verbatim': 0.9}
    ok = find_violations(schema, {'work_experiences': [
        {'company': '总部 HRD'}]}, SRC_NORM)
    bad = find_violations(schema, {'work_experiences': [
        {'company': '编造的公司名'}]}, SRC_NORM)
    assert ok == [] and [(f.path, f.code) for f in bad] == [
        ('work_experiences[0].company', 'verbatim')]


def test_verbatim_issues_carry_repairable_messages():
    data = {'work_experiences': [{'description': ['不是原文的句子。']}]}
    (issue,) = verbatim_issues(SCHEMA, data, SRC_NORM)
    assert issue.path == 'work_experiences[0].description[0]'
    assert issue.code == 'verbatim' and issue.report_only is False
    assert issue.expected == 'verbatim copy from the document'
    assert issue.got == '不是原文的句子。' and 're-copy' in issue.message


BUDGET_SCHEMA = {'type': 'object', 'properties': {
    'career': {'type': 'object', 'properties': {
        'jobs': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'company': {'type': 'string', 'description': '公司'},
            'detail': {'type': 'object', 'properties': {
                'description': {'type': 'array', 'description': '描述',
                                'x-verbatim': 0.9,
                                'items': {'type': 'string'}}}}}}}}},
    'misc': {'type': 'object', 'properties': {
        'things': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'company': {'type': 'string', 'description': '公司'}}}}}}}}


def group(path: str, item, text: str) -> Group:
    return Group(Unit(path, 'array', {}, ''), item, [], text)


def test_verbatim_budgets_derive_from_routed_material():
    units = [Unit('career.jobs', 'array', {}, ''),
             Unit('misc.things', 'array', {}, '')]
    routing = SimpleNamespace(groups=[
        # '- 腾讯 后端\n- 负责渠道营销。' → 1+2+2+1+7 non-blank chars
        group('career.jobs', 0, '- 腾讯 后端\n- 负责渠道营销。'),
        group('career.jobs', 1, '- 阿里'),
        group('misc.things', 0, '- 某某公司'),
    ])
    assert verbatim_budgets(BUDGET_SCHEMA, units, routing) == {'career.jobs': [13, 3]}


def test_verbatim_budgets_cover_whole_units_and_inherit_gaps():
    unit = Unit('career.jobs', 'array', {}, '')
    whole = verbatim_budgets(BUDGET_SCHEMA, [unit], SimpleNamespace(
        groups=[group('career.jobs', None, '整段 材料')]))
    assert whole == {'career.jobs': 4}
    gap = verbatim_budgets(BUDGET_SCHEMA, [unit], SimpleNamespace(groups=[
        group('career.jobs', 0, '甲乙'),
        group('career.jobs', 2, '丙丁')]))
    assert gap == {'career.jobs': [2, 2, 2]}  # item 1 inherits item 0's value


def test_verbatim_budgets_reach_lifted_marked_arrays():
    """decompose lifts a marked array-under-array into a unit of its
    own and strips the array property's description — detection reads
    the declared schema, so the lift cannot lose the mark."""
    lifted = {'type': 'object', 'properties': {
        'work': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'tasklist': {'type': 'array', 'description': '清单',
                         'x-verbatim': 0.9,
                         'items': {'type': 'string'}}}}}}}
    units = decompose(lifted)
    routing = SimpleNamespace(groups=[group('work.tasklist', 0, '材料')])
    assert verbatim_budgets(lifted, units, routing) == {'work.tasklist': [2]}


def test_verbatim_budgets_empty_for_unmarked_schemas():
    unit = Unit('misc.things', 'array', {}, '')
    assert verbatim_budgets(BUDGET_SCHEMA, [unit], SimpleNamespace(
        groups=[group('misc.things', 0, '材料')])) == {}
