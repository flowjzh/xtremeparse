"""Boundary probes for the router: segment DSL, count declarations, repair."""

import pytest

from xtremeparse.router import (NONE, RECOUNT_PLACEHOLDERS, ROUTE_PLACEHOLDERS,
                                RouterError, Group, _name_list, route)
from xtremeparse.units import MISC, decompose
from tests.helpers import ScriptedRunner, agent_result

SCHEMA = {
    'type': 'object',
    'properties': {
        'basic_info': {'type': 'object', 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
        'jobs': {'type': 'array', 'description': '工作经历', 'items': {'type': 'object', 'properties': {
            'company': {'type': 'string', 'description': '公司'},
        }}},
        'created': {'type': 'string', 'description': '创建时间'},
    },
}
UNITS = decompose(SCHEMA)
CHUNKS = ['姓名张三', '第一段：腾讯', '第二段：阿里', '无关页脚']
PAYLOAD = '全文\n\n---\nJSON Schema: ...'


def runner_ok():
    # codes: a=basic_info, b=jobs, c=$misc, - = none
    return ScriptedRunner(agent_result('0 a\n1 b.0\n2 b.1\n3 -\nb: 2'))


async def route_with(text):
    # the DSL is queued twice: a zero-declared array unit gets one
    # verification round that re-emits it unchanged
    return await route(ScriptedRunner(*[agent_result(text)] * 2),
                       payload=PAYLOAD, units=UNITS, chunks=CHUNKS)


async def test_groups_carry_unit_item_and_chunk_text():
    routing = await route(runner_ok(), payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups] == [
        ('basic_info', None, [0]), ('jobs', 0, [1]), ('jobs', 1, [2])]
    assert routing.groups[0].text == '姓名张三'
    assert routing.raw['assignments'][1] == {'unit': 'jobs', 'item': 0, 'chunks': [1]}
    assert routing.raw['counts'] == {'jobs': 2}


async def test_ranges_and_single_chunk_forms_are_equivalent():
    ranged = await route_with('0-1 a\n2-3 -\nb: 0')
    single = await route_with('0 a\n1 a\n2 -\n3 -\nb: 0')
    assert [(g.unit.path, g.chunk_ids) for g in ranged.groups] == \
        [(g.unit.path, g.chunk_ids) for g in single.groups]


async def test_prompt_carries_legend_rules_and_numbered_chunks():
    runner = runner_ok()
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    call = runner.calls[0]
    assert 'a = [basic_info | object] 基本信息' in call['instructions'] \
        and f'c = [{MISC} | scalar] created 创建时间' in call['instructions']
    assert 'cover EVERY chunk id in 0..3' in call['instructions']
    assert 'order you meet them across the WHOLE document' \
        in call['instructions']
    assert 'distinct instances into one' in call['instructions']
    assert '[0] 姓名张三' in call['instructions']
    assert '4-9 x.1' in call['instructions']  # inline format anchor
    assert 'one job' not in call['instructions']  # no domain leakage
    assert call['result_schema']['type'] == 'string'
    assert call['feedback'] is None


def test_default_prompts_carry_all_placeholders():
    # drift guard: an edited default missing a slot would render empty
    # at call time — the same fail-fast hosts get for their overrides
    from xtremeparse.prompting import check_placeholders
    from xtremeparse import router
    check_placeholders(router._INSTRUCTIONS, ROUTE_PLACEHOLDERS, 'router')
    check_placeholders(router._CHECK, RECOUNT_PLACEHOLDERS, 'recount')


async def test_invalid_map_gets_one_repair_with_feedback():
    runner = ScriptedRunner(
        agent_result('0 a\n1 x.0\n2 b.1\nb: 2'),  # unknown code + missing coverage
        agent_result('0 a\n1 b.0\n2 b.1\n3 -\nb: 2'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert "unknown unit code 'x'" in messages and 'not covered' in messages
    assert routing.groups[1].chunk_ids == [1]


async def test_still_invalid_after_repairs_raises():
    bad = agent_result('0 a\n1 b.0\n2 b.1\nb: 2')  # never covers chunk 3
    runner = ScriptedRunner(*[bad] * 5)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)


async def test_overlap_between_segments_is_repaired():
    runner = ScriptedRunner(
        agent_result('0-1 a\n1 b.0\n2-3 -\nb: 1'),  # chunk 1 in two segments
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'overlaps or breaks order' in runner.calls[1]['feedback'][0].message


async def test_bare_array_code_is_coerced_to_whole_item_zero():
    # a bare repeating code means its instances share the run unsplit —
    # coerced to item 0, the executor extracts them whole
    routing = await route_with('0 a\n1 b\n2-3 -\nb: 1')
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups] == [
        ('basic_info', None, [0]), ('jobs', 0, [1])]


async def test_none_takes_no_item_index():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -.0\nb: 1'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'takes no item index' in runner.calls[1]['feedback'][0].message


async def test_item_on_non_array_is_stripped():
    routing = await route_with('0-1 a.7\n2 b.0\n3 -\nb: 1')
    assert routing.groups[0].item is None


async def test_all_none_yields_no_groups():
    routing = await route_with('0-3 -\nb: 0')
    assert routing.groups == [] and routing.raw['assignments'] == []


async def test_single_item_section_is_fine_when_declared_as_one():
    # one instance is legitimate — the map's own count makes it consistent
    routing = await route_with('0 a\n1-3 b.0\nb: 1')
    assert [g.item for g in routing.groups if g.unit.kind == 'array'] == [0]


async def test_declared_count_must_match_mapped_items():
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 b.0\nb: 2'),  # declared 2, mapped only item 0
        agent_result('0 a\n1 b.0\n2-3 b.1\nb: 2'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'declared 2 items but the map uses [0]' \
        in runner.calls[1]['feedback'][0].message
    assert [g.item for g in routing.groups if g.unit.kind == 'array'] == [0, 1]


async def test_missing_count_declaration_is_repaired():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'count is undeclared' in runner.calls[1]['feedback'][0].message


async def test_absent_declaration_reads_as_zero_and_gets_recounted():
    # map-first output declares what the map used; a unit absent from
    # both reads as zero — safe because every zero is re-examined by
    # the fresh recount pass
    runner = ScriptedRunner(
        agent_result('0-3 -'),
        agent_result('0-3 -\nb: 0'))  # recount confirms the zero
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 0} and routing.groups == []


async def test_zero_declaration_gets_one_recount_round():
    # a zero is never trusted on the map's own say-so: a fresh
    # conversation (no map in view) re-counts the unit, and code splices
    # its answer into the validated map — the anchored conversation
    # would re-emit its own map verbatim
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 0'),
        agent_result('b: 1\n1 b.0'))  # recount: one instance, its map line
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2 and runner.calls[1]['feedback'] is None
    assert routing.raw['counts'] == {'jobs': 1}
    assert [g.chunk_ids for g in routing.groups if g.unit.kind == 'array'] == [[1]]


async def test_recount_adopts_a_derivation_for_a_summarizing_unit():
    # the recount may answer with a source instead of a count — the
    # adopted unit then mirrors its source's groups, no map lines
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1\nc: 0'),
        agent_result('c = b'))  # recount: it only summarizes b
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 1, 'summary': 1}
    assert [g.chunk_ids for g in routing.groups if g.unit.path == 'summary'] == [[1]]


async def test_unspliceable_recount_falls_back_to_a_repair_round():
    # the recount claims a chunk the map already routed — code cannot
    # merge it, so the anchored conversation gets one bounded repair
    runner = ScriptedRunner(
        agent_result('0-1 a\n2-3 -\nb: 0'),
        agent_result('b: 1\n1 b.0'),  # claims chunk 1, routed to a
        agent_result('0-1 a\n2 b.0\n3 -\nb: 1'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'could not be merged' in runner.calls[2]['feedback'][0].message
    assert routing.raw['counts'] == {'jobs': 1}


async def test_recount_claims_inside_one_units_items_read_as_derivation():
    # a summarizing unit answers a COUNT and maps its rows onto another
    # unit's items — the misplaced claims betray the summary relation;
    # code adopts the derivation, whose mirroring is exact
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2 b.1\n3 -\nb: 2\nc: 0'),
        agent_result('c: 2\n1 c.0\n2 c.1'))
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 2, 'summary': 2}
    assert [g.chunk_ids for g in routing.groups if g.unit.path == 'summary'] \
        == [[1], [2]]


async def test_recount_noise_about_other_units_is_ignored_not_fatal():
    # the recount re-answers units it was not asked about, with an
    # out-of-range line among them — only RECOUNT-owned lines matter
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1\nc: 0'),
        agent_result('b: 1\n9-9 b.0\nc: 0'),  # b not owned; 9-9 outside
    )
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=CHUNKS)
    assert len(runner.calls) == 2  # spliced, no repair round
    assert routing.raw['counts'] == {'jobs': 1, 'summary': 0}


async def test_unusable_claims_with_a_unique_same_count_still_derive():
    # the recounted count is right but its map lines land on garbage —
    # a unique same-count repeating unit still reads as the source
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2 b.1\n3 -\nb: 2\nc: 0'),
        agent_result('c: 2\n0 c.0,c.1'))  # claims a non-unit's chunk
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 2, 'summary': 2}


async def test_recount_of_a_truly_absent_unit_finishes_there():
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 0'),
        agent_result('b: 0'))  # recount confirms the zero
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 0}
    assert [g.unit.path for g in routing.groups] == ['basic_info']


async def test_count_before_map_is_rejected_and_non_array_count_ignored():
    runner = ScriptedRunner(
        agent_result('a: 2\n0 a\n1 b.0\n2-3 -\nb: 1\nc: 1'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'follow the map' in messages and 'not a repeating unit' not in messages


async def test_non_text_output_is_invalid():
    bad = agent_result({'assignments': []})
    runner = ScriptedRunner(*[bad] * 5)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)


def test_group_is_a_plain_dataclass():
    g = Group(unit=UNITS[0], item=None, chunk_ids=[0], text='x')
    assert g.unit.path == 'basic_info' and NONE == '-'


# --- shared runs: one chunk range feeding several units (comma-joined codes) ---

SHARED_SCHEMA = {
    'type': 'object',
    'properties': {
        'basic_info': {'type': 'object', 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
        'jobs': {'type': 'array', 'description': '工作经历', 'items': {'type': 'object', 'properties': {
            'company': {'type': 'string', 'description': '公司'},
        }}},
        'summary': {'type': 'array', 'description': '经历概述', 'items': {'type': 'object', 'properties': {
            'company': {'type': 'string', 'description': '公司'},
        }}},
    },
}
SHARED_UNITS = decompose(SHARED_SCHEMA)  # a=basic_info, b=jobs, c=summary


async def route_shared(text):
    return await route(ScriptedRunner(*[agent_result(text)] * 2),
                       payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)


async def test_shared_run_feeds_both_units():
    routing = await route_shared('0 a\n1 b.0,c.0\n2 b.1,c.1\n3 -\nb: 2\nc: 2')
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups] == [
        ('basic_info', None, [0]),
        ('jobs', 0, [1]), ('summary', 0, [1]),
        ('jobs', 1, [2]), ('summary', 1, [2])]
    assert routing.groups[1].text == routing.groups[2].text == '第一段：腾讯'
    assert routing.raw['counts'] == {'jobs': 2, 'summary': 2}
    assert {'unit': 'summary', 'item': 0, 'chunks': [1]} in routing.raw['assignments']


async def test_repeated_index_spans_another_units_items():
    # one summary item covering both jobs: c.0 repeats across adjacent lines
    routing = await route_shared('0 a\n1 b.0,c.0\n2 b.1,c.0\n3 -\nb: 2\nc: 1')
    assert [(g.item, g.chunk_ids) for g in routing.groups
            if g.unit.path == 'summary'] == [(0, [1, 2])]


async def test_code_twice_on_a_line_is_repaired():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0,b.0\n2-3 -\nb: 1\nc: 1'),
        agent_result('0 a\n1 b.0,c.0\n2-3 -\nb: 1\nc: 1'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert "'b.0' appears twice" in runner.calls[1]['feedback'][0].message


async def test_cochunked_items_share_a_line():
    # one chunk holding several instances: items share the run, and the
    # executor later extracts that material once, whole
    routing = await route_shared('0 a\n1-2 b.0,b.1\n3 -\nb: 2\nc: 0')
    assert [(g.item, g.chunk_ids, g.text) for g in routing.groups
            if g.unit.path == 'jobs'] == [
        (0, [1, 2], '第一段：腾讯\n第二段：阿里'),
        (1, [1, 2], '第一段：腾讯\n第二段：阿里')]


async def test_none_cannot_be_comma_joined():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0,-\n2-3 -\nb: 1\nc: 1'),
        agent_result('0 a\n1 b.0,c.0\n2-3 -\nb: 1\nc: 1'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert "'1 b.0,-'" in runner.calls[1]['feedback'][0].message  # format-level reject


async def test_shared_unit_count_still_cross_checked():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0,c.0\n2-3 -\nb: 1\nc: 2'),
        agent_result('0 a\n1 b.0,c.0\n2-3 -\nb: 1\nc: 1'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert 'declared 2 items but the map uses [0]' \
        in runner.calls[1]['feedback'][0].message


async def test_prompt_carries_sharing_rule():
    runner = ScriptedRunner(agent_result('0 a\n1 b.0,c.0\n2-3 -\nb: 1\nc: 1'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert 'comma-join' in runner.calls[0]['instructions']
    assert '5 x.0,y.0' in runner.calls[0]['instructions']
    assert '3 x.0,x.1,x.2' in runner.calls[0]['instructions']
    assert '"x = y"' in runner.calls[0]['instructions']


# --- derived units: "c = b" after the map mirrors the source, no map lines ---

async def test_derived_unit_mirrors_its_source():
    routing = await route_shared('0 a\n1 b.0\n2 b.1\n3 -\nb: 2\nc = b')
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups] == [
        ('basic_info', None, [0]),
        ('jobs', 0, [1]), ('jobs', 1, [2]),
        ('summary', 0, [1]), ('summary', 1, [2])]
    assert [a['unit'] for a in routing.raw['assignments']] == \
        ['basic_info', 'jobs', 'jobs', 'summary', 'summary']
    assert routing.raw['counts'] == {'jobs': 2, 'summary': 2}


async def test_derived_count_comes_from_the_source():
    # a number beside "= <source>" is tolerated and ignored — mirroring
    # is definitional, so the count-mismatch failure class cannot exist
    routing = await route_shared('0 a\n1 b.0\n2 b.1\n3 -\nb: 2\nc: 7 = b')
    assert routing.raw['counts'] == {'jobs': 2, 'summary': 2}
    assert [g.chunk_ids for g in routing.groups
            if g.unit.path == 'summary'] == [[1], [2]]


async def test_bare_declaration_without_count_or_source_is_rejected():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1\nc'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1\nc = b'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert 'must declare a count' in runner.calls[1]['feedback'][0].message


async def test_exact_duplicate_map_line_is_tolerated():
    routing = await route_shared('0 a\n1 b.0,c.0\n1 b.0,c.0\n2-3 -\nb: 1\nc: 1')
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups] == [
        ('basic_info', None, [0]), ('jobs', 0, [1]), ('summary', 0, [1])]


async def test_spaces_around_commas_are_tolerated():
    routing = await route_shared('0 a\n1 b.0, c.0\n2-3 -\nb: 1\nc: 1')
    assert [g.unit.path for g in routing.groups] == \
        ['basic_info', 'jobs', 'summary']


async def test_derived_source_must_be_directly_mapped():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1\nc = d'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1\nc = b'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert 'is not a unit code' in runner.calls[1]['feedback'][0].message


async def test_derived_source_needs_its_own_count():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nc = b'),  # b itself undeclared
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1\nc = b'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'needs b\'s own count declared first' in messages


async def test_multiline_chunks_list_one_line_each():
    # embedded newlines would make the listing's line count disagree
    # with the chunk ids and break the model's index arithmetic
    runner = ScriptedRunner(*[agent_result('0-1 -\nb: 0')] * 2)
    await route(runner, payload=PAYLOAD, units=UNITS,
                chunks=['两行\n的块', 'x'])
    assert '[0] 两行 ¶ 的块' in runner.calls[0]['instructions']


async def test_budget_suffix_parses_into_routing():
    routing = await route_with('0 a\n1 b.0\n2 b.1\n3 -\nb: 2 @300')
    assert routing.budgets == {'jobs': 300}
    assert routing.raw['budgets'] == {'jobs': '300'}  # declared form, verbatim


async def test_budget_on_a_derived_line_parses():
    routing = await route_shared('0 a\n1 b.0,c.0\n2-3 -\nb: 1\nc = b @40')
    assert routing.budgets == {'summary': 40}
    assert routing.raw['counts'] == {'jobs': 1, 'summary': 1}


async def test_dotted_item_tail_in_a_comma_list_keeps_its_index():
    # a comma-joined "b.2.0" parses as item 2 (the regex's comma tail
    # admits dotted numbers); it must not crash the int conversion
    routing = await route_shared('0 a\n1 b.0,c.0\n2 b.1,c.1\n3 c.2,b.2.0\nb: 3\nc = b')
    assert [g.item for g in routing.groups if g.unit.path == 'jobs'] == [0, 1, 2]


async def test_none_comma_joined_gets_a_specific_hint():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 c.0,-\nb: 1'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'may not be comma-joined' in runner.calls[1]['feedback'][0].message


async def test_per_item_budget_list_parses():
    routing = await route_with('0 a\n1 b.0\n2 b.1\n3 -\nb: 2 @500,300')
    assert routing.budgets == {'jobs': [500, 300]}
    assert routing.raw['budgets'] == {'jobs': ['500', '300']}


async def test_ratio_budget_resolves_value_chars_against_material():
    # one shared ratio scales per item: chunk 1 is '第一段：腾讯' (6 chars),
    # chunk 1 is '第一段：腾讯' (6 value chars), chunk 2 likewise
    routing = await route_with('0 a\n1 b.0\n2 b.1\n3 -\nb: 2 @100%')
    assert routing.budgets == {'jobs': [6, 6]}
    assert routing.raw['budgets'] == {'jobs': '100%'}


async def test_shared_chunk_material_splits_across_items():
    # co-chunked items share one run ("1-2 b.0,b.1") but each mirrors
    # its own slice: the 12 chars split across the two, 6 apiece
    routing = await route_with('0 a\n1-2 b.0,b.1\n3 -\nb: 2 @100%')
    assert routing.budgets == {'jobs': [6, 6]}


async def test_keyword_budget_splits_its_document_total_across_items():
    # 20x3 is the document's total (60): two mapped items share it,
    routing = await route_with('0 a\n1 b.0\n2 b.1\n3 -\nb: 2 @20x3')
    assert routing.budgets == {'jobs': [30, 30]}
    assert routing.raw['budgets'] == {'jobs': '20x3'}


async def test_mixed_budget_forms_resolve_per_item():
    routing = await route_with('0 a\n1 b.0\n2 b.1\n3 -\nb: 2 @50%,20x3')
    assert routing.budgets == {'jobs': [3, 60]}
    # a listed keyword entry is its item's own total, not split
    assert routing.raw['budgets'] == {'jobs': ['50%', '20x3']}


async def test_ratio_budget_on_a_derived_line_scales_against_the_source():
    routing = await route_shared('0 a\n1 b.0,c.0\n2 b.1,c.1\n3 -\nb: 2\nc = b @50%')
    assert routing.budgets == {'summary': [3, 3]}
    # half of each source item's material, in value characters
    assert routing.raw['budgets'] == {'summary': '50%'}


async def test_malformed_budget_entry_is_dropped_not_fatal():
    routing = await route_with('0 a\n1 b.0\n2 b.1\n3 -\nb: 2 @50x')
    assert routing.budgets == {} and routing.raw['counts'] == {'jobs': 2}


async def test_budget_without_digits_is_dropped_not_fatal():
    routing = await route_with('0 a\n1 b.0\n2 b.1\n3 -\nb: 2 @若干')
    assert routing.budgets == {}
    assert routing.raw['counts'] == {'jobs': 2}


async def test_budget_on_a_map_line_is_rejected():
    runner = ScriptedRunner(*[agent_result('0 a\n1 b.0 @50\n2-3 -\nb: 1')] * 5)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)


def test_name_list_predicate():
    """One string field per item is the mechanical name-list shape;
    optional (anyOf-wrapped) strings count, two fields or a scalar
    $misc do not."""
    def unit(kind, sub_schema):
        from xtremeparse.units import Unit
        return Unit(path='x', kind=kind, sub_schema=sub_schema, card='c')

    assert _name_list(unit('array', {'type': 'object', 'properties': {
        'name': {'type': 'string'}}}))
    assert _name_list(unit('array', {'type': 'object', 'properties': {
        'name': {'anyOf': [{'type': 'string'}, {'enum': ['']}]}}}))
    assert _name_list(unit('array', {'type': 'string'}))  # scalar repeat
    assert not _name_list(unit('array', {'type': 'object', 'properties': {
        'name': {'type': 'string'}, 'date': {'type': 'string'}}}))
    assert not _name_list(unit('array', {'type': 'object', 'properties': {
        'count': {'type': 'integer'}}}))

    assert not _name_list(unit('scalar', {'type': 'object', 'properties': {
        'name': {'type': 'string'}}}))


async def test_prompt_carries_the_budget_rule():
    runner = runner_ok()
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert '"x: 3 @100%,80%,50%"' in runner.calls[0]['instructions']
    assert '"x: 3 @300"' in runner.calls[0]['instructions']  # the abs example
    assert '"@<n>%"' in runner.calls[0]['instructions']
    assert '"@20x3"' not in runner.calls[0]['instructions']  # no numeric anchor
    assert 'never words' in runner.calls[0]['instructions']  # char arithmetic stays


async def test_a_repaired_error_that_reappears_is_called_out():
    # ping-pong repairs: each round fixes one error and resurrects the
    # other — the third round's feedback must name the regression
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2 x.0\nb: 1'),          # unknown code
        agent_result('0 a\n1 b.0\nb: 1'),                  # coverage lost
        agent_result('0 a\n1 b.0\n2 x.0\nb: 1'),          # unknown code back
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    fourth_call = ' '.join(i.message for i in runner.calls[3]['feedback'])
    assert 'already fixed in an earlier round' in fourth_call


async def test_deriving_from_a_non_repeating_unit_is_explained():
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 0\nc = a'),
        agent_result('0 a\n1-3 -\nb: 0\nc = b'),
        agent_result('0 a\n1-3 -\nb: 0\nc = b'))  # verification round
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert 'is not a repeating unit' in runner.calls[1]['feedback'][0].message
    assert routing.raw['counts'] == {'jobs': 0, 'summary': 0}


async def test_starred_run_splits_one_item_per_chunk():
    routing = await route_with('0 a\n1-2 b*\n3 -\nb: 2')
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups] == [
        ('basic_info', None, [0]), ('jobs', 0, [1]), ('jobs', 1, [2])]
    assert routing.raw['counts'] == {'jobs': 2}
    assert routing.raw['assignments'][1:] == [
        {'unit': 'jobs', 'item': 0, 'chunks': [1]},
        {'unit': 'jobs', 'item': 1, 'chunks': [2]}]


async def test_starred_count_is_code_read_not_trusted():
    routing = await route_with('0 a\n1-2 b*\n3 -\nb: 7')
    assert routing.raw['counts'] == {'jobs': 2}


async def test_starred_count_line_may_be_omitted():
    routing = await route_with('0 a\n1-2 b*\n3 -')
    assert routing.raw['counts'] == {'jobs': 2}


async def test_star_and_numbered_cannot_mix():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b*\n2 b.0\n3 -\nb: 1'),
        agent_result('0 a\n1 b*\n2-3 -\nb: 1'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'cannot mix' in messages
    assert [(g.unit.path, g.item) for g in routing.groups if g.unit.path == 'jobs'] \
        == [('jobs', 0)]


async def test_star_takes_its_line_alone():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b*,a\n3 -\nb: 2'),
        agent_result('0 a\n1-2 b*\n3 -\nb: 2'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'takes its line alone' in messages
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups] == [
        ('basic_info', None, [0]), ('jobs', 0, [1]), ('jobs', 1, [2])]


async def test_star_needs_an_array_unit():
    runner = ScriptedRunner(
        agent_result('0-1 a*\n2 b.0\n3 -\nb: 1'),
        agent_result('0 a\n1 a\n2 b.0\n3 -\nb: 1'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'needs a repeating unit' in messages
    assert routing.groups[0].unit.path == 'basic_info'


# the dense-list scenario: 8 instances sharing an 8-chunk run, plus
# one object chunk and one NONE tail — the star-hint's home case
DENSE_CHUNKS = [f'c{i}' for i in range(10)]
DENSE = '0 a\n1-8 b.0,b.1,b.2,b.3,b.4,b.5,b.6,b.7\n9 -\na: 1\nb: 8'


def shared_jobs(routing):
    """The jobs assignments of the shared form: one per instance, each
    co-owning the whole run."""
    proj = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert len(proj) == 8
    assert all(len(a['chunks']) == 8 for a in proj)
    return proj


async def test_dense_shared_run_gets_one_star_hint_then_verbatim_accepted():
    # 8 instances over 8 chunks in shared form (the dense-list soup): the
    # hint asks once; the verbatim re-emission is a declined suggestion —
    # the map is accepted
    runner = ScriptedRunner(agent_result(DENSE), agent_result(DENSE))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=DENSE_CHUNKS)
    hint = runner.calls[1]['feedback'][0]
    assert hint.code == 'route_hint' \
        and 're-emit the map with every b run starred' in hint.message
    assert len(runner.calls) == 2  # asked once, then accepted as declined
    shared_jobs(routing)  # the model's shared form stands


async def test_star_hint_answered_with_star_fans_out():
    runner = ScriptedRunner(
        agent_result('0 a\n1-8 b\n9 -\na: 1\nb: 8'),
        agent_result('0 a\n1-8 b*\n9 -\na: 1\nb: 8'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=DENSE_CHUNKS)
    assert [g.item for g in routing.groups if g.unit.path == 'jobs'] == list(range(8))
    assert routing.raw['counts'] == {'jobs': 8}


async def test_star_hint_answered_with_a_diff_stars_the_run():
    # the repair round diffs its own previous answer instead of
    # re-emitting the map — the untouched lines are never at risk
    runner = ScriptedRunner(
        agent_result(DENSE),
        agent_result('-1-8 b.0,b.1,b.2,b.3,b.4,b.5,b.6,b.7\n+1-8 b*'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=DENSE_CHUNKS)
    assert [g.item for g in routing.groups if g.unit.path == 'jobs'] == list(range(8))
    assert routing.raw['counts'] == {'jobs': 8}


async def test_empty_diff_reply_declines_the_hint():
    # silence is a declined suggestion: the previous map stands
    runner = ScriptedRunner(agent_result(DENSE), agent_result(''))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=DENSE_CHUNKS)
    assert len(runner.calls) == 2
    shared_jobs(routing)  # the shared form stands


async def test_full_reemission_after_a_hint_still_parses():
    # a draw that ignores the diff instruction degrades to the old path
    runner = ScriptedRunner(agent_result(DENSE), agent_result(DENSE))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=DENSE_CHUNKS)
    assert len(runner.calls) == 2
    shared_jobs(routing)  # the shared form stands


async def test_small_shared_runs_skip_the_star_hint():
    # declared below STAR_HINT_MIN: no confirmation round at all
    chunks = [f'c{i}' for i in range(6)]
    runner = ScriptedRunner(agent_result('0 a\n1-3 b.0,b.1,b.2\n4-5 -\na: 1\nb: 3'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=chunks)
    assert len(runner.calls) == 1
    assert routing.raw['counts']['jobs'] == 3


# --- shared-run hint: instances merged into one long run (the lazy draw) ---

LAZY_CHUNKS = [f'c{i}' for i in range(40)]
LAZY = '0 a\n1-37 b.0\n38-39 -\na: 1\nb: 1'


async def test_lazy_single_instance_gets_recounted_and_resplit():
    # one item spanning 37 chunks for a unit declared once — the mirror
    # of the dense-list case. A fresh conversation recounts (count plus
    # quoted openings) and code re-splits the run at the anchored
    # chunks; the anchored model never sees the map
    runner = ScriptedRunner(agent_result(LAZY), agent_result('b: 2\nc5\nc30'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS)
    recount = runner.calls[1]
    assert 'were left merged as one' in recount['instructions'] \
        and 'b = [jobs | array]' in recount['instructions']
    assert len(runner.calls) == 2
    assert [g.item for g in routing.groups if g.unit.path == 'jobs'] == [0, 1]
    assert routing.raw['counts'] == {'jobs': 2}
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert jobs[0]['chunks'] == list(range(1, 30))
    assert jobs[1]['chunks'] == list(range(30, 38))


async def test_resplit_keeps_co_riding_destinations():
    # a cross-cutting unit riding the shared run's chunks keeps them
    runner = ScriptedRunner(
        agent_result('0 a\n1-37 b.0,c.0\n38-39 -\na: 1\nb: 1'),
        agent_result('b: 2\nc5\nc30'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS)
    misc = [a for a in routing.raw['assignments'] if a['unit'] == MISC]
    assert sorted(c for a in misc for c in a['chunks']) == list(range(1, 38))
    assert [g.item for g in routing.groups if g.unit.path == 'jobs'] == [0, 1]


async def test_unusable_recount_falls_back_to_the_diff_hint():
    # a recount answer without quotes is unusable: the anchored
    # conversation gets the diff round, whose empty reply declines —
    # the shared form stands
    runner = ScriptedRunner(
        agent_result('0 a\n1-37 b.0,b.1,b.2,b.3\n38-39 -\na: 1\nb: 4'),
        agent_result('b: 4'),
        agent_result(''))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS)
    hint = runner.calls[2]['feedback'][0]
    assert hint.code == 'route_hint' and 'one line per instance' in hint.message
    assert len(runner.calls) == 3  # asked once, then declined
    proj = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert len(proj) == 4 and all(len(a['chunks']) == 37 for a in proj)


async def test_per_item_multi_chunk_map_skips_the_shared_hint():
    # each instance already on its own line (entries spanning several
    # chunks — the correct form for career sections): no round at all
    runner = ScriptedRunner(
        agent_result('0 a\n1-20 b.0\n21-37 b.1\n38-39 -\na: 1\nb: 2'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS)
    assert len(runner.calls) == 1
    assert routing.raw['counts'] == {'jobs': 2}


async def test_short_shared_runs_skip_the_shared_hint():
    # the run's excess over the declared count is under STAR_HINT_MIN:
    # a round would cost more than the split could save
    runner = ScriptedRunner(
        agent_result('0 a\n1-6 b.0,b.1\n7-8 -\na: 1\nb: 2'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS[:9])
    assert len(runner.calls) == 1
    assert routing.raw['counts']['jobs'] == 2


# two array units: a dense one for the star hint, a merged one for the
# shared recount — codes a=basic_info, b=jobs, c=projects, d=$misc
TWO_ARRAYS = decompose({
    'type': 'object',
    'properties': {
        'basic_info': SCHEMA['properties']['basic_info'],
        'jobs': SCHEMA['properties']['jobs'],
        'projects': {'type': 'array', 'description': '项目经历',
                     'items': {'type': 'object', 'properties': {
                         'name': {'type': 'string', 'description': '项目名'},
                     }}},
        'created': SCHEMA['properties']['created'],
    },
})


async def test_star_hint_and_shared_recount_both_fire():
    # the conditions are mutually exclusive per unit, so one document
    # can carry both: the star diff round runs first (a resplit done
    # before it would be discarded by the re-parse), the recount
    # re-splits the merged unit after
    runner = ScriptedRunner(
        agent_result('0 a\n1-8 b.0,b.1,b.2,b.3,b.4,b.5,b.6,b.7\n9 -\n'
                     '10-37 c.0\n38-39 -\na: 1\nb: 8\nc: 1'),
        agent_result('-1-8 b.0,b.1,b.2,b.3,b.4,b.5,b.6,b.7\n+1-8 b*'),
        agent_result('c: 2\nc12\nc30'))
    routing = await route(runner, payload=PAYLOAD, units=TWO_ARRAYS,
                          chunks=LAZY_CHUNKS)
    assert 'c = [projects | array] 项目经历 RECOUNT' \
        in runner.calls[2]['instructions']
    assert len(runner.calls) == 3
    assert [g.item for g in routing.groups if g.unit.path == 'jobs'] \
        == list(range(8))
    assert [g.item for g in routing.groups if g.unit.path == 'projects'] \
        == [0, 1]
    assert routing.raw['counts'] == {'jobs': 8, 'projects': 2}


async def test_two_merged_units_recount_in_one_conversation():
    runner = ScriptedRunner(
        agent_result('0 a\n1-18 b.0\n19 -\n20-37 c.0\n38-39 -\n'
                     'a: 1\nb: 1\nc: 1'),
        agent_result('b: 2\nc5\nc15\nc: 2\nc25\nc35'))
    routing = await route(runner, payload=PAYLOAD, units=TWO_ARRAYS,
                          chunks=LAZY_CHUNKS)
    assert 'b = [jobs | array] 工作经历 RECOUNT' \
        in runner.calls[1]['instructions'] \
        and 'c = [projects | array] 项目经历 RECOUNT' \
        in runner.calls[1]['instructions']
    assert len(runner.calls) == 2  # both fixed in the one recount
    assert routing.raw['counts'] == {'jobs': 2, 'projects': 2}
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert jobs[0]['chunks'] == list(range(1, 15))
    assert jobs[1]['chunks'] == list(range(15, 19))
    projs = [a for a in routing.raw['assignments'] if a['unit'] == 'projects']
    assert projs[0]['chunks'] == list(range(20, 35))
    assert projs[1]['chunks'] == list(range(35, 38))


async def test_partial_adoption_skips_the_fallback_round():
    # one unit's recount section usable, the other's not: the adopted
    # split stands and no diff round follows — a round taken now would
    # re-parse the pre-resplit answer text and discard the adoption;
    # the pending unit simply stays shared
    runner = ScriptedRunner(
        agent_result('0 a\n1-18 b.0\n19 -\n20-37 c.0\n38-39 -\n'
                     'a: 1\nb: 1\nc: 1'),
        agent_result('b: 2\nc5\nc15'))
    routing = await route(runner, payload=PAYLOAD, units=TWO_ARRAYS,
                          chunks=LAZY_CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 2, 'projects': 1}
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert jobs[0]['chunks'] == list(range(1, 15))
    assert jobs[1]['chunks'] == list(range(15, 19))
