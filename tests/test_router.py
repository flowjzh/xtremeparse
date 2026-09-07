"""Boundary probes for the router: segment DSL, count declarations, repair."""

import pytest

from xtremeparse.router import (NONE, RECOUNT_PLACEHOLDERS, ROUTE_PLACEHOLDERS,
                                RouterError, Group, _diff_text, _name_list,
                                _overlay_text, _resplit, route)
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
BAD_MAP = agent_result('0 a\n1 b.0\n2 b.1\nb: 2')  # never covers chunk 3


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


@pytest.mark.parametrize('overall, present', [(None, False),
                                              ('Chinese resumes only', True)])
async def test_overall_block_in_the_route_prompt(overall, present):
    runner = runner_ok()
    kwargs = {'overall': overall} if overall is not None else {}
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS, **kwargs)
    rendered = 'Overall Instruction:\n\nChinese resumes only'
    assert (rendered in runner.calls[0]['instructions']) is present


async def test_overall_instruction_renders_into_the_recount_prompt():
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 0'),
        agent_result('b: 0'))  # the recount confirms the zero
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS,
                overall='Chinese resumes only')
    assert 'Overall Instruction:\n\nChinese resumes only' \
        in runner.calls[1]['instructions']


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
    runner = ScriptedRunner(*[BAD_MAP] * 5)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)


async def test_overlap_between_segments_is_repaired():
    runner = ScriptedRunner(
        agent_result('0-1 a\n1 b.0\n2-3 -\nb: 1'),  # chunk 1 in two segments
        agent_result('0 a\n1 b.0\n2-3 -\nb: 1'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    # the conflicting line is named — range and codes — so the repair
    # can quote it instead of hunting for it
    assert 'overlaps line 0-1 (a)' in runner.calls[1]['feedback'][0].message


async def test_out_of_order_map_lines_are_accepted():
    # line order carries no meaning — ranges are explicit; a diff
    # reply's "+" insert lands at its own editing position
    routing = await route_with('0 a\n2 b.1\n1 b.0\n3 -\nb: 2')
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups
            if g.unit.kind == 'array'] == [('jobs', 0, [1]), ('jobs', 1, [2])]


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


async def test_zero_confirm_adopts_silently_on_a_chain_map():
    # the covering chain form (parent line over its chains) must
    # round-trip the adoption tail — see _cover
    schema = {**NESTED_SCHEMA, 'properties': {
        **NESTED_SCHEMA['properties'],
        'certs': {'type': 'array', 'description': '证书', 'items': {
            'type': 'object', 'properties': {
                'name': {'type': 'string', 'description': '名称'}}}}}}
    # a=basic_info, b=jobs, c=jobs.roles, d=certs, e=$misc
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 b.0\n2 b.0.c.0\n3 b.0.c.1\n4 -\n'
                     'b: 1\nb.0.c: 2\nd: 0'),
        agent_result('d: 0'))  # recount confirms the zero
    routing = await route(runner, payload=PAYLOAD,
                          units=decompose(schema), chunks=NESTED_CHUNKS)
    assert len(runner.calls) == 2 and runner.calls[1]['feedback'] is None
    roles = [g for g in routing.groups if g.unit.parent]
    assert [(g.item, g.chunk_ids) for g in roles] == [(0, [2]), (1, [3])]


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


async def test_recount_ranged_claim_adopts_as_the_shared_form():
    # the recount may answer in the ranged spelling, doubled code
    # included — the compact shared form's own syntax, not a crash
    runner = ScriptedRunner(
        agent_result('0-1 a\n2-3 -\nb: 0'),
        agent_result('b: 2\n2-3 b.0-b.1'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 2}
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert [(a['item'], a['chunks']) for a in jobs] == \
        [(0, [2, 3]), (1, [2, 3])]


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


async def test_code_twice_on_a_line_dedupes_instead_of_erroring():
    # a re-claimed destination ('1 b.0,b.0', or a parent repeated after
    # its own chain) claims nothing new — idempotent, never a repair
    # round; only the real inconsistency (c declared, nothing mapped)
    # draws feedback. The declared-undrawn inconsistency takes the
    # fresh-recount round first; this recount is unusable (its quotes
    # anchor nowhere), so the anchored conversation gets the repair —
    # and the feedback still names the count mismatch, never the
    # harmless duplicate
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0,b.0\n2-3 -\nb: 1\nc: 1'),
        agent_result('c: 1\n不存在的行'),  # recount quotes anchor nowhere
        agent_result('0 a\n1 b.0,c.0\n2-3 -\nb: 1\nc: 1'))
    await route(runner, payload=PAYLOAD, units=SHARED_UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 3
    messages = ' '.join(i.message for i in runner.calls[2]['feedback'])
    assert 'appears twice' not in messages
    assert 'declared 1 items but the map uses []' in messages


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


# --- shared-run hint: instances merged into one long run (the lazy draw) ---

LAZY_CHUNKS = [f'c{i}' for i in range(40)]
LAZY = '0 a\n1-37 b.0\n38-39 -\na: 1\nb: 1'


async def test_lazy_single_instance_gets_recounted_and_resplit():
    # one item spanning 37 chunks for a unit declared once — a whole
    # section mapped as one shared item. A fresh conversation recounts
    # (count plus quoted openings) and code re-splits the run at the
    # anchored chunks; the anchored model never sees the map
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
    # the run's excess over the declared count is under HINT_MIN:
    # a round would cost more than the split could save
    runner = ScriptedRunner(
        agent_result('0 a\n1-6 b.0,b.1\n7-8 -\na: 1\nb: 2'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS[:9])
    assert len(runner.calls) == 1
    assert routing.raw['counts']['jobs'] == 2


async def test_decomposed_band_skips_the_recount_below_the_load_bar():
    # one destination per instance already (the co-chunked band's
    # shape): its recount confirmed every time and never adopted —
    # decode-shaving only, not worth a round this thin
    runner = ScriptedRunner(
        agent_result('0 a\n1-8 b.0,b.1\n9-39 -\na: 1\nb: 2'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS)
    assert len(runner.calls) == 1
    assert routing.raw['counts']['jobs'] == 2


async def test_thin_count_one_run_skips_the_shared_recount():
    # a unit declared once over a handful of chunks reads as one long
    # entry as plausibly as a merge — its recount confirmed every time
    # and never adopted (measured), so it obeys the same load bar as
    # every decomposed line instead of recounting on its say-so
    runner = ScriptedRunner(
        agent_result('0 a\n1-6 b.0\n7-8 -\na: 1\nb: 1'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=LAZY_CHUNKS[:9])
    assert len(runner.calls) == 1
    assert routing.raw['counts']['jobs'] == 1


# --- material-overflow split ask: a shared run whose material
# --- overflows one call's capacity

OVER_MAP = '0 a\n1-3 b.0,b.1,b.2\n4-7 b.3,b.4,b.5\n8 -\na: 1\nb: 6'
RIDER_MAP = '0 a\n1-3 b.0,b.1,b.2,c.0\n4-7 b.3,b.4,b.5\n8 -\na: 1\nb: 6'
OVER_CHUNKS = ['姓名张三'] + ['x' * 160] * 7 + ['无关页脚']  # 1120 chars of b material


async def test_overrun_shared_run_adopts_its_blocks_without_a_round():
    # declared 6 over a 7-chunk shared run of fat chunks: beyond one
    # call's capacity, so code computes the even partition and splices
    # it into the map itself — the ask round typed the block lines
    # verbatim every time, a round spent re-taking code's dictation
    runner = ScriptedRunner(agent_result(OVER_MAP))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=OVER_CHUNKS)
    assert len(runner.calls) == 1  # adopted, never asked
    blocks = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [g.items for g in blocks] == [(), (1, 2), (), (4, 5)]
    assert [g.chunk_ids for g in blocks] == [[1], [2, 3], [4, 5], [6, 7]]
    assert routing.raw['counts'] == {'jobs': 6}
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert [a['item'] for a in jobs] == list(range(6))
    assert jobs[0]['chunks'] == [1] and jobs[5]['chunks'] == [6, 7]


async def test_ranged_run_is_the_compact_shared_form_without_overflow():
    # a ranged run parses as one co-owned group; under the material cap
    # nothing asks and the executor-facing shape is one shared slice
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0-1\n3 -\na: 1\nb: 2'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=['姓名张三', '第一段：腾讯', '第二段：阿里', '无关页脚'])
    assert len(runner.calls) == 1
    (group,) = [g for g in routing.groups if g.unit.path == 'jobs']
    assert group.items == (0, 1) and group.chunk_ids == [1, 2]
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert [(a['item'], a['chunks']) for a in jobs] == \
        [(0, [1, 2]), (1, [1, 2])]


async def test_item_range_may_repeat_the_code():
    # the model's natural reading of the house form ("b.0-b.1") —
    # normalized so the ranged draw survives round one instead of
    # cascading into per-instance enumeration
    runner = ScriptedRunner(agent_result('0 a\n1-2 b.0-b.1\n3 -\nb: 2'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 1
    (group,) = [g for g in routing.groups if g.unit.path == 'jobs']
    assert group.items == (0, 1) and group.chunk_ids == [1, 2]


async def test_spaced_item_range_gets_a_named_hint():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0 - b.1\n3 -\nb: 2'),
        agent_result('0 a\n1-2 b.0-1\n3 -\nb: 2'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'never repeats the code' in runner.calls[1]['feedback'][0].message


async def test_sub_numbered_item_folds_to_its_parent():
    # the model sub-numbers entries the document nests under one
    # instance the schema holds flat ("b.0.0" is one employer's first
    # role) — the parent index is the claim; rejecting the spelling
    # sent a canary repair loop circling to exhaustion
    runner = ScriptedRunner(agent_result('0 a\n1-2 b.0.0\n3 b.0.1\nb: 1'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 1
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert [(a['item'], a['chunks']) for a in jobs] == \
        [(0, [1, 2]), (0, [3])]


async def test_shape_failure_names_the_dotted_tail_not_the_drop():
    # the old shape error tacked "a bare '-' maps nothing, drop the
    # line entirely" onto every malformed line — advice that steered a
    # canary repair loop into dropping real lines and cascading into
    # uncovered chunks
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0.0-c\n3 -\nb: 2'),
        agent_result('0 a\n1 b.0\n2 b.1\n3 -\nb: 2'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    fb = runner.calls[1]['feedback'][0].message
    assert 'drop the line' not in fb
    assert 'first index' in fb


async def test_bare_none_line_still_gets_the_drop_hint():
    runner = ScriptedRunner(
        agent_result('0 a\n-\n2 b.1\n3 -\nb: 2'),
        agent_result('0 a\n1 b.0\n2 b.1\n3 -\nb: 2'),
    )
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'drop the line' in runner.calls[1]['feedback'][0].message


async def test_same_range_duplicate_lines_merge_into_the_shared_form():
    # the model answers "two instances share one chunk" with two lines
    # claiming the chunk — same range, one chunk set: the destinations
    # union into the co-chunked shared form instead of a repair round
    # (the repair hint circled to exhaustion on exactly this shape,
    # measured — merging is the answer the hint was asking for)
    routing = await route_with('0 a\n1 b.0\n2 b.1\n2 b.0\n3 -\nb: 2')
    assert [(g.unit.path, g.item, g.chunk_ids) for g in routing.groups
            if g.unit.kind == 'array'] == [
        ('jobs', 0, [1, 2]), ('jobs', 1, [2])]


async def test_diff_rewriting_every_line_keeps_counts_after_the_map():
    # a full "-x/+x" rewrite leaves the count declarations as the only
    # unquoted base lines; the applier used to append the additions
    # behind them, building a counts-first text it then blamed the
    # model for every remaining round (measured: a canary loop
    # exhausted on exactly this flood)
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n1 b.1\n3 -\nb: 2'),
        agent_result('-0 a\n+0 a\n-1 b.0\n+1 b.0\n-1 b.1\n+2 b.1\n-3 -\n+3 -\nb: 2'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert not any(a['unit'] == 'jobs' and a['item'] == 1
                   and 1 in a['chunks'] for a in routing.raw['assignments'])


async def test_single_chunk_range_may_share_its_line():
    # a ranged claim on ONE chunk spells the co-chunked shared form —
    # instances a..b all ride the chunk, exactly what comma-joined
    # indexes mean; the model reaches for it wherever a small unit
    # shares its section heading's chunk, and rejecting it sent a
    # canary repair loop circling to exhaustion
    runner = ScriptedRunner(
        agent_result('0 a\n1 a.0,b.0-b.1\n2-3 -\na: 2\nb: 2'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 1
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert [(a['item'], a['chunks']) for a in jobs] == \
        [(0, [1]), (1, [1])]


async def test_multi_chunk_range_shared_with_other_destinations_expands():
    # the co-chunked shared form spelled compactly on any line length
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 b.0,b.1-2\n4 -\nb: 3'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=CHUNKS + ['无关页脚二'])
    assert len(runner.calls) == 1
    jobs = [g for g in routing.groups if g.unit.kind == 'array']
    assert [(g.item, g.chunk_ids) for g in jobs] == \
        [(0, [1, 2, 3]), (1, [2, 3]), (2, [2, 3])]


async def test_shared_range_expansion_keeps_the_undrawn_gate_open():
    # the expanded map's only error is the count family — the recount
    # fires and adopts in the one round, no diff repairs
    chunks = ['姓名张三', '公司甲·工程师', '公司甲·经理', '无关页脚']
    runner = ScriptedRunner(
        agent_result('0 -\n1-2 a,b.0-1\n3 -\nb: 2\nb.0.c: 2'),
        agent_result('b.0.c: 2\n公司甲·工程师\n公司甲·经理'))
    routing = await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                          chunks=chunks)
    assert len(runner.calls) == 2
    roles = [g for g in routing.groups if g.unit.path == 'jobs.roles']
    assert [(g.item, g.parent, g.chunk_ids) for g in roles] == \
        [(0, 0, [1]), (1, 0, [2])]


async def test_multi_chunk_chain_range_shared_expands_like_the_flat_form():
    # the dotted-chain spelling of a comma-joined range takes the same
    # tolerance — one rule, not two
    runner = ScriptedRunner(
        agent_result('0 -\n1-2 a,b.0.c.0-1\n3 -\nb: 1\nb.0.c: 2'))
    routing = await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                          chunks=['姓名张三', '公司甲·工程师', '公司甲·经理',
                                  '无关页脚'])
    assert len(runner.calls) == 1
    roles = [g for g in routing.groups if g.unit.path == 'jobs.roles']
    assert [(g.item, g.parent, g.chunk_ids) for g in roles] == \
        [(0, 0, [1, 2]), (1, 0, [1, 2])]


async def test_ranged_instances_cannot_repeat_across_lines():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0-1\n3-4 b.1-2\n5 -\na: 1\nb: 3'),
        agent_result('0 a\n1-2 b.0-1\n3-4 b.2\n5 -\na: 1\nb: 3'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=CHUNKS + ['无关页脚', '无关页脚二'])
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'ranged twice' in messages
    blocks = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [(g.items, g.item) for g in blocks] == [((0, 1), None), ((), 2)]


async def test_ranged_initial_draw_adopts_its_blocks_without_a_round():
    # the base prompt's greedy ranged form: the initial draw is ONE
    # short line and the computed blocks splice straight through it —
    # the short ranged line that once made the ask's diff quotable now
    # makes the adoption lossless
    runner = ScriptedRunner(agent_result('0 a\n1-7 b.0-5\n8 -\na: 1\nb: 6'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=OVER_CHUNKS)
    assert 'NEVER enumerate' in runner.calls[0]['instructions']
    assert len(runner.calls) == 1
    blocks = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [g.items for g in blocks] == [(), (1, 2), (), (4, 5)]
    assert [g.chunk_ids for g in blocks] == [[1], [2, 3], [4, 5], [6, 7]]
    # every round's applied map text rides the trace — one round, the
    # initial draw; the adoption rides the groups, not the rounds
    assert len(routing.raw['maps']) == 1


async def test_thin_shared_run_skips_the_split_ask():
    # same shared shape, material under SPLIT_MATERIAL_CAP: one whole
    # call decodes it in seconds — no round
    runner = ScriptedRunner(agent_result(OVER_MAP))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=['姓名张三'] + ['x' * 10] * 7 + ['无关页脚'])
    assert len(runner.calls) == 1
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert len(jobs) == 6  # the shared form stands


async def test_split_ask_on_a_riding_line_falls_back_to_the_round():
    # $misc rides the run's first line: code may not redraw it, so the
    # ask round stays for this run — and silence declines, the shared
    # whole stands
    runner = ScriptedRunner(
        agent_result(RIDER_MAP),
        agent_result(''))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=OVER_CHUNKS)
    ask = runner.calls[1]['feedback'][0]
    assert ask.code == 'route_hint' \
        and 'one ranged line per block' in ask.message \
        and '6-7 b.4-5' in ask.message  # the last computed block line
    assert len(runner.calls) == 2  # asked once, then accepted as declined
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert jobs[0]['chunks'] == [1, 2, 3] and jobs[5]['chunks'] == [4, 5, 6, 7]


async def test_botched_split_reply_declines_to_the_standing_map():
    # a fan-out round that fails validation is worth less than the valid
    # map it answered: decline restores the standing map instead of
    # spending repair rounds on a redraw — silence and wreckage both
    # decline, only a validating answer lands
    runner = ScriptedRunner(
        agent_result(RIDER_MAP),
        agent_result('0 a\n1-2 b.0-2\n3-4 b.1-3\n5-7 b.4-5\n8 -\na: 1\nb: 6'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=OVER_CHUNKS)
    assert len(runner.calls) == 2  # declined — no repair round
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert [a['item'] for a in jobs] == list(range(6))
    assert jobs[0]['chunks'] == [1, 2, 3] and jobs[5]['chunks'] == [4, 5, 6, 7]


async def test_partial_block_adoption_rebases_the_ask_round():
    # two fat units: one's lines code may redraw (adopted in code, no
    # round), one carries a $misc rider so its ask round stays — and
    # that round's diff anchors on the rebased map, the adopted blocks
    # already standing in it
    chunks = (['姓名张三'] + ['x' * 160] * 7 + ['间隔页']
              + ['y' * 160] * 7 + ['无关页脚'])
    runner = ScriptedRunner(
        agent_result('0 a\n1-7 b.0,b.1,b.2,b.3,b.4\n8 -\n'
                     '9-15 c.0,c.1,c.2,c.3,c.4,d.0\n16 -\na: 1\nb: 5\nc: 5'),
        agent_result(''))
    routing = await route(runner, payload=PAYLOAD, units=TWO_ARRAYS,
                          chunks=chunks)
    ask = runner.calls[1]['feedback'][0]
    assert ask.code == 'route_hint' and ask.message.startswith('c:')
    blocks = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [g.items for g in blocks] == [(), (), (), (3, 4)]
    assert [g.chunk_ids for g in blocks] == [[1], [2, 3], [4, 5], [6, 7]]
    assert len(runner.calls) == 2  # b adopted; c asked once, declined
    shared = [a for a in routing.raw['assignments'] if a['unit'] == 'projects']
    assert len(shared) == 5  # the decline keeps the shared whole


async def test_split_ask_precedes_the_lazy_recount():
    # one document carrying both shapes: the dense-fat unit's blocks
    # splice in code first, the lazy merged unit takes the fresh
    # recount after — the recount's re-split adopts in code too, so
    # the two adoptions compose with no ask round between them
    chunks = ['姓名张三'] + ['x' * 350] * 3 + [f'c{i}' for i in range(4, 40)]
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 b.0,b.1,b.2,b.3,b.4\n4 -\n5-37 c.0\n38-39 -\n'
                     'a: 1\nb: 5\nc: 1'),
        agent_result('c: 2\nc12\nc30'))
    routing = await route(runner, payload=PAYLOAD, units=TWO_ARRAYS,
                          chunks=chunks)
    assert 'RECOUNT' in runner.calls[1]['instructions']
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 5, 'projects': 2}


# two array units: one for the split ask, a merged one for the shared
# recount — codes a=basic_info, b=jobs, c=projects, d=$misc
PROJECTS = {'type': 'array', 'description': '项目经历',
            'items': {'type': 'object', 'properties': {
                'name': {'type': 'string', 'description': '项目名'}}}}

TWO_ARRAYS = decompose({
    'type': 'object',
    'properties': {
        'basic_info': SCHEMA['properties']['basic_info'],
        'jobs': SCHEMA['properties']['jobs'],
        'projects': PROJECTS,
        'created': SCHEMA['properties']['created'],
    },
})


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


BOTH_UNITS = decompose({
    'type': 'object',
    'properties': {
        'basic_info': SCHEMA['properties']['basic_info'],
        'jobs': SCHEMA['properties']['jobs'],
        'projects': PROJECTS,
        'certs': {'type': 'array', 'description': '证书',
                  'items': {'type': 'object', 'properties': {
                      'name': {'type': 'string', 'description': '证书名'},
                  }}},
    },
})
# codes: a=basic_info, b=jobs, c=projects, d=certs
BOTH_MAP = ('0 a\n1-18 b.0\n19-20 c.0\n21-22 c.1\n23-38 -\n39 -\n'
            'a: 1\nb: 1\nc: 2\nd: 0')


async def test_merged_and_zero_units_recount_in_one_conversation():
    # both fan-out kinds pend: one fresh conversation recounts them
    # together — the chunks listing is the costly part
    runner = ScriptedRunner(
        agent_result(BOTH_MAP),
        agent_result('b: 2\nc5\nc15\nd: 0'))
    routing = await route(runner, payload=PAYLOAD, units=BOTH_UNITS,
                          chunks=LAZY_CHUNKS)
    recount = runner.calls[1]
    assert len(runner.calls) == 2 and recount['feedback'] is None
    assert 'RECOUNT MERGED' in recount['instructions'] \
        and 'RECOUNT ZERO' in recount['instructions'] \
        and 'b = [jobs | array] 工作经历 RECOUNT MERGED' \
        in recount['instructions'] \
        and 'd = [certs | array] 证书 RECOUNT ZERO' \
        in recount['instructions']
    assert routing.raw['counts'] == {'jobs': 2, 'projects': 2, 'certs': 0}
    jobs = [a for a in routing.raw['assignments'] if a['unit'] == 'jobs']
    assert jobs[0]['chunks'] == list(range(1, 15))
    assert jobs[1]['chunks'] == list(range(15, 19))


async def test_combined_recount_zeros_fallback_leaves_the_map_standing():
    # the zero unit's section claims mapped material and cannot merge:
    # one resolution round follows — a diff re-parses the model's text
    # over any split, so the merged unit's usable sections are not
    # adopted first, and a decline keeps the standing map
    runner = ScriptedRunner(
        agent_result(BOTH_MAP),
        agent_result('b: 2\nc5\nc15\nd: 3\n19 d.0,d.1,d.2'),
        agent_result(''))
    routing = await route(runner, payload=PAYLOAD, units=BOTH_UNITS,
                          chunks=LAZY_CHUNKS)
    assert len(runner.calls) == 3
    assert 'could not be merged' \
        in runner.calls[2]['feedback'][0].message
    assert routing.raw['counts'] == {'jobs': 1, 'projects': 2, 'certs': 0}


async def test_recount_instructions_override_keeps_the_two_asks_apart():
    # a host's override is tuned for the zero case: the combined
    # conversation stays off, the two recounts run serially
    runner = ScriptedRunner(
        agent_result(BOTH_MAP),
        agent_result('b: 2\nc5\nc15'),
        agent_result('d: 0'))
    routing = await route(runner, payload=PAYLOAD, units=BOTH_UNITS,
                          chunks=LAZY_CHUNKS,
                          recount_instructions='Units:\n{legend}\n\n{chunks}')
    assert len(runner.calls) == 3
    assert 'were left merged as one' in runner.calls[1]['instructions']
    assert 'Units:' in runner.calls[2]['instructions'] \
        and 'd = [certs | array] 证书 RECOUNT' \
        in runner.calls[2]['instructions']
    assert routing.raw['counts'] == {'jobs': 2, 'projects': 2, 'certs': 0}


async def test_a_repair_round_diffs_against_the_previous_answer():
    # every repair is a diff against the model's own last map text —
    # hint or invalid alike: the patch applies to that raw text and the
    # reconstruction validates fresh
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nb: 2'),  # b.1 unmapped
        agent_result('-2-3 -\n+2 b.1\n+3 -'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert 'unified diff' in runner.calls[1]['feedback'][0].message
    assert [g.item for g in routing.groups if g.unit.kind == 'array'] == [0, 1]


async def test_an_empty_repair_reply_keeps_the_errors():
    # an empty reply declines the diff: the base — still invalid —
    # stands, the same errors come back, and the loop stays bounded
    runner = ScriptedRunner(BAD_MAP, agent_result(''), BAD_MAP, BAD_MAP, BAD_MAP)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 5


async def test_a_missed_diff_quote_cannot_duplicate_a_line():
    # "-x/+x" rewrites of a line the base no longer holds (a missed
    # quote) must stay no-ops — a duplicated count line is how a whole
    # repair chain used to die ("b declared twice")
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\nb: 1'),  # chunks 2-3 uncovered
        agent_result('-9 z\n+b: 1\n+2-3 -'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2  # no "b declared twice" repair chain
    assert [g.item for g in routing.groups if g.unit.kind == 'array'] == [0]


async def test_a_still_invalid_diff_keeps_its_base_map():
    # a diff reply that does not fix the errors must not become the next
    # round's base — patching a patch loses the map. The next repair
    # diffs against the last map text, where quoting the stray '+4 -'
    # line finds nothing to drop and the error stands
    junk_diff, hopeless = agent_result('+4 -'), agent_result('-4 -')
    runner = ScriptedRunner(BAD_MAP, junk_diff, hopeless, hopeless, hopeless)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 5
    assert 'not covered' in runner.calls[4]['feedback'][0].message


async def test_a_lazy_attach_line_lands_as_an_addition():
    # the contract is never to re-emit an unchanged line, so a bare
    # line beside real markers is a lazy "+": asked to attach its
    # items, the model drew the lines with no prefix — dropped as
    # commentary they silently zeroed the unit while the removal
    # beside them landed, and the phantom burned the repair budget
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 2'),
        agent_result('b: 2\n不存在的内容'),  # the recount anchors nowhere
        agent_result('- 1-3 -\n1 b.0\n2 b.1\n3 -'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 3
    assert [g.chunk_ids for g in routing.groups
            if g.unit.kind == 'array'] == [[1], [2]]


def test_lazy_diff_bare_lines_are_additions_replays_are_no_ops():
    base = '0 a\n1-2 -\nb: 2'
    # the bare attach lands, the count rewrite applies
    assert _diff_text('1 b.0,b.1\n- b: 2\n+ b: 2', base) == \
        '0 a\n1-2 -\n1 b.0,b.1\nb: 2'
    # a bare line the map already holds dedupes — a quoted replay of
    # unchanged lines costs nothing
    assert _diff_text('1-2 -\n+ 0 a', base) == base
    # a marker-less reply is no diff at all: the full re-emission
    # reading stands (a bare fragment must not replace the map)
    assert _diff_text('1 b.0,b.1', base) is None
    # prose beside markers stays commentary
    assert _diff_text('- b: 2\nplease reconsider the summary', base) == \
        '0 a\n1-2 -'
    # bare chain lines compose: the fragment patch reading now holds
    # for marker-mixed replies too
    chained = '0 a\n1-3 b.0\n4 -\nb: 1\nb.0.c: 2'
    patched = _diff_text('2 b.0.c.0\n3 b.0.c.1\n+ b.0.c: 2', chained)
    assert '2 b.0.c.0' in patched and '3 b.0.c.1' in patched


# --- nested arrays: a lifted sub-array addressed through its parent ---

NESTED_SCHEMA = {
    'type': 'object',
    'properties': {
        'basic_info': {'type': 'object', 'description': '基本信息', 'properties': {
            'name': {'type': 'string', 'description': '姓名'},
        }},
        'jobs': {'type': 'array', 'description': '工作经历',
                 'items': {'type': 'object', 'properties': {
                     'company': {'type': 'string', 'description': '公司'},
                     'roles': {'type': 'array', 'description': '任职经历',
                               'items': {'type': 'object', 'properties': {
                                   'title': {'type': 'string', 'description': '职位'},
                               }}},
                 }}},
        'created': {'type': 'string', 'description': '创建时间'},
    },
}
NESTED_UNITS = decompose(NESTED_SCHEMA)  # a=basic_info, b=jobs, c=jobs.roles, d=$misc
NESTED_CHUNKS = ['姓名张三', '公司甲·工程师', '公司甲·经理', '公司乙', '无关页脚']


async def route_nested(text):
    return await route(ScriptedRunner(*[agent_result(text)] * 2),
                       payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)


async def test_chain_line_feeds_parent_and_sub_entry():
    routing = await route_nested('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1')
    assert [(g.unit.path, g.item, g.parent, g.chunk_ids)
            for g in routing.groups] == [
        ('basic_info', None, None, [0]),
        ('jobs', 0, None, [1, 2]),
        ('jobs.roles', 0, 0, [1, 2]),
        ('jobs', 1, None, [3])]
    assert routing.raw['counts'] == {'jobs': 2}
    assert routing.raw['nested_counts'] == {'jobs.roles': {0: 1}}
    assert {'unit': 'jobs.roles', 'item': 0, 'parent': 0, 'chunks': [1, 2]} \
        in routing.raw['assignments']


async def test_field_name_and_bare_tail_spellings_normalize_to_the_chain():
    expected = [('basic_info', None, None), ('jobs', 0, None),
                ('jobs.roles', 0, 0), ('jobs', 1, None)]
    for text in ('0 a\n1-2 b.0.roles.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1',
                 '0 a\n1-2 b.0.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'):
        routing = await route_nested(text)
        assert [(g.unit.path, g.item, g.parent) for g in routing.groups] == expected


async def test_ranged_chain_carries_its_sub_items_inseparably():
    routing = await route_nested('0 a\n1-3 b.0.c.0-1\n4 -\nb: 1\nb.0.c: 2')
    assert [(g.unit.path, g.item, g.parent, g.chunk_ids)
            for g in routing.groups if g.parent is not None] == [
        ('jobs.roles', 0, 0, [1, 2, 3]), ('jobs.roles', 1, 0, [1, 2, 3])]


async def test_doubled_chain_range_normalizes_like_the_flat_form():
    routing = await route_nested('0 a\n1-3 b.0.c.0-c.1\n4 -\nb: 1\nb.0.c: 2')
    assert [g.item for g in routing.groups if g.parent is not None] == [0, 1]


async def test_bare_nested_code_rides_its_parent():
    runner = ScriptedRunner(
        agent_result('0 a\n1 c.0\n2-3 -\nb: 0'),
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'rides its parent' in runner.calls[1]['feedback'][0].message


async def test_top_level_count_on_a_nested_unit_is_named():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nc: 1'),
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'counts per parent' in runner.calls[1]['feedback'][0].message


async def test_nested_units_bare_zero_count_is_silent_noise():
    routing = await route_nested('0 a\n1 b.0\n2-3 b.1\n4 -\nb: 2\nc: 0')
    assert [g.item for g in routing.groups if g.parent is not None] == []


async def test_nested_units_bare_zero_still_hides_no_real_mismatch():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nc: 0'),
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'sub-entries used' in runner.calls[1]['feedback'][0].message


async def test_replayed_count_mismatch_passes_through():
    # a count the model cannot localize comes back as a no-op diff
    # (every line re-quoted as -x/+x): re-asking cannot move it, so the
    # replay passes the mismatch through — the extraction's count
    # arbitration owns the number, the declared count stays in counts
    replay = ('-0 a\n-1 b.0\n-2-3 -\n-\n-b: 2\n'
              '+0 a\n+1 b.0\n+2-3 -\n+\n+b: 2')
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nb: 2'),
        agent_result(replay))
    routing = await route(runner, payload=PAYLOAD, units=UNITS,
                          chunks=CHUNKS)
    assert len(runner.calls) == 2  # replay seen once, pass-through, final
    assert routing.raw['counts'] == {'jobs': 2}
    assert [g.item for g in routing.groups if g.unit.path == 'jobs'] == [0]


async def test_replayed_nested_mismatch_passes_through():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 2'),
        agent_result('-0 a\n-1-2 b.0.c.0\n-3 b.1\n-4 -\n-\n-b: 2\n'
                     '-b.0.c: 2\n+0 a\n+1-2 b.0.c.0\n+3 b.1\n+4 -\n+\n'
                     '+b: 2\n+b.0.c: 2'))
    routing = await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                          chunks=NESTED_CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 2}
    assert routing.raw['nested_counts'] == {'jobs.roles': {0: 2}}
    assert [g.item for g in routing.groups if g.parent is not None] == [0]


async def test_replayed_blocking_errors_still_exhaust():
    # a coverage hole is not a count: a replayed map with one still
    # burns the budget — the pass-through is for mismatches only
    runner = ScriptedRunner(*[agent_result('0 a\n1 b.0\n3 -\nb: 1')] * 5)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)


async def test_chain_without_sub_item_is_named():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0.c\n3 b.1\n4 -\nb: 2\nb.0.c: 1'),
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'needs its sub-item' in runner.calls[1]['feedback'][0].message


async def test_chain_under_a_childless_parent_is_named():
    runner = ScriptedRunner(
        agent_result('0-1 a.0.c.0\n2 b.0\n3-4 -\nb: 1\nb.0.c: 1'),
        agent_result('0 a\n1 b.0.c.0\n2 b.0\n3-4 -\nb: 1\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'does not nest under a' in runner.calls[1]['feedback'][0].message


async def test_nested_chain_count_line_for_a_foreign_unit_is_named():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.a: 1'),
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'not a nested unit chain' in runner.calls[1]['feedback'][0].message


async def test_nested_declared_count_must_match_mapped_sub_items():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 2'),
        agent_result('0 a\n1-2 b.0.c.0\n3 b.0.c.1\n4 -\nb: 1\nb.0.c: 2'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'declared 2 items under b.0' in runner.calls[1]['feedback'][0].message


async def test_sub_item_beyond_the_declaration_names_the_fold():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0.c.0\n2 b.0.c.1\n3 b.0.c.2\n4 -\nb: 1\nb.0.c: 2'),
        agent_result('0 a\n1 b.0.c.0\n2-3 b.0.c.1\n4 -\nb: 1\nb.0.c: 2'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert ("sub-item 2 under b.0 is beyond the declared 2 (0..1) — extend "
            "the previous sub-entry's line over its chunks (\"2-3 b.0.c.1\")"
            in messages)


async def test_coverage_hole_after_a_line_names_the_extension():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0\n4 -\nb: 1'),
        agent_result('0 a\n1-3 b.0\n4 -\nb: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'chunks not covered: [3]' in messages
    assert 'extend the preceding line over them ("1-3 b.0")' in messages


async def test_nested_declaration_without_map_lines_is_named():
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2 b.1\n3-4 -\nb: 2\nb.0.c: 2'),
        agent_result('b.0.c: 2\n无此内容'),  # recount quotes anchor nowhere
        agent_result('0 a\n1 b.0\n2 b.1\n3-4 -\nb: 2'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[2]['feedback'])
    assert 'declared 2 items under b.0 but the map assigns none' in messages
    assert '("b.0.c.0", "b.0.c.1")' in messages


async def test_a_split_of_an_undropped_line_names_the_removal():
    # the prod slip: the model split a wide chain line into two slices
    # without removing the wide line — the remedy quotes the removal
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 b.0.c.1\n1-2 b.0.c.1\n4 -\nb: 1\nb.0.c: 2'),
        agent_result('0 a\n1-2 b.0.c.1\n3 b.0.c.0\n4 -\nb: 1\nb.0.c: 2'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'add "- 1-3 b.0.c.1" and the slices replace it' in messages


async def test_a_removal_uncovering_chunks_names_the_re_add():
    # the collapse's second half: the repair's own removal leaves its
    # chunks bare — the feedback names the line to re-add
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\n4 -\nb: 1\nb.0.c: 1'),
        agent_result('b.0.c: 1\n无此内容'),  # recount quotes anchor nowhere
        agent_result('- 1-3 -\n+ 1-2 b.0.c.0'),
        agent_result('0 a\n1-2 b.0.c.0\n3-4 -\nb: 1\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[3]['feedback'])
    assert 'chunks not covered: [3]' in messages
    assert 'your removal of "1-3 -" uncovered these' in messages


async def test_removal_blame_wins_over_the_neighbour_fold():
    # when the hole is both a removal's doing and a neighbour's
    # continuation, the removal is the cause — one remedy, stated once
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\n4 -\nb: 1'),
        agent_result('b: 1\n无此内容'),  # recount quotes anchor nowhere
        agent_result('- 1-3 -\n+ 1-2 b.0'),
        agent_result('0 a\n1-2 b.0\n3-4 -\nb: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[3]['feedback'])
    assert 'your removal of "1-3 -" uncovered these' in messages
    assert 'extend the preceding line' not in messages


async def test_chain_parent_index_out_of_range_is_named():
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0\n3 b.2.c.0\n4 -\nb: 2\nb.2.c: 1'),
        agent_result('0 a\n1-2 b.0\n3 b.1\n4 -\nb: 2'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    messages = ' '.join(i.message for i in runner.calls[1]['feedback'])
    assert 'chained under b.2 but b declares 2 items' in messages


async def test_nested_declaration_before_the_map_is_rejected():
    runner = ScriptedRunner(
        agent_result('b.0.c: 1\n0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2'),
        agent_result('0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    assert 'follow the map' in runner.calls[1]['feedback'][0].message


def test_overlay_patches_chain_only_replies():
    # the measured marker-less fragment: asked to add the chain lines,
    # the model answers with the chain lines alone — the patch reading
    # keeps the head the replacement reading would have dropped
    base = '0 a\n1-2 b.0\n3 b.1\n4 -\nb: 2\nb.0.c: 2'
    assert _overlay_text('1 b.0.c.0\n2 b.0.c.1', base) == (
        '0 a\n1-2 b.0\n3 b.1\n4 -\n1 b.0.c.0\n2 b.0.c.1\nb: 2\nb.0.c: 2')
    assert _overlay_text('1 b.0.c.0', base) == (
        '0 a\n1-2 b.0\n3 b.1\n4 -\n1 b.0.c.0\nb: 2\nb.0.c: 2')
    # any covering line keeps the replacement reading — the other
    # marker-less intent, omission-to-delete; counts alone patch
    # nothing without chains to ride in on
    assert _overlay_text('1-2 b.0.c.0\n3 b.1', base) is None
    assert _overlay_text('b: 3', base) is None


async def test_chain_fragment_repair_keeps_the_map_head():
    # the chain-lazy repair answered with the chain lines alone: the
    # patch reading lands the chains inside the standing parent run —
    # two rounds, no head re-typing
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0\n3 b.1\n4 -\nb: 2\nb.0.c: 2'),
        agent_result('b.0.c: 2\n公司甲·工程师'),  # count vs quotes mismatch
        agent_result('1 b.0.c.0\n2 b.0.c.1'))
    routing = await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                          chunks=NESTED_CHUNKS)
    assert 'declared 2 items under b.0' in runner.calls[2]['feedback'][0].message
    assert len(runner.calls) == 3
    roles = [g for g in routing.groups if g.unit.path == 'jobs.roles']
    assert [(g.item, g.chunk_ids) for g in roles] == [(0, [1]), (1, [2])]
    assert routing.raw['counts'] == {'jobs': 2}
    assert routing.raw['nested_counts'] == {'jobs.roles': {0: 2}}


async def test_prompt_teaches_the_chain():
    runner = ScriptedRunner(agent_result(
        '0 a\n1-2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS, chunks=NESTED_CHUNKS)
    text = runner.calls[0]['instructions']
    assert 'c = [jobs.roles | array]' in text  # the legend names the parent path
    assert 'nests inside' in text
    assert 'c.0.d: 2' in text and '3-15 c.0.d.0' in text


async def test_chain_lines_coalesce_parent_chunks_but_not_sub_entries():
    # parent item 0 rides both chain lines — its groups coalesce across
    # them; the sub-entries stay apart
    routing = await route_nested('0 a\n1-2 b.0.c.0\n3-4 b.0.c.1\nb: 1\n'
                                 'b.0.c: 2')
    assert [(g.unit.path, g.item, g.parent, g.chunk_ids)
            for g in routing.groups] == [
        ('basic_info', None, None, [0]),
        ('jobs', 0, None, [1, 2, 3, 4]),
        ('jobs.roles', 0, 0, [1, 2]),
        ('jobs.roles', 1, 0, [3, 4])]


async def test_shared_resplit_reads_chain_destinations_through():
    # regression: a shared unit whose owned chunks carry a chain line —
    # the anchor scan must read 3-tuple destinations through, not
    # unpack them flat (a live draw crashed exactly here)
    by_code = {c: u for c, u in zip('abcd', NESTED_UNITS)}
    segments = [(0, 0, (('b', 0, None),)),
                (1, 1, (('b', 0, None), ('c', 0, 0))),
                (2, 2, (('b', 1, None),)),
                (3, 4, ((NONE, None, None),))]
    merged = _resplit(segments, {'b': 1}, {('c', 0): 1}, 'b',
                      (2, ['姓名张三', '公司甲·工程师']),
                      by_code, {}, NESTED_CHUNKS)
    assert merged is not None
    segs, counts = merged
    assert counts == {'b': 2}
    assert segs[1][2] == (('c', 0, 0), ('b', 1, None))


async def test_parent_run_and_chain_on_same_range_lines_merge():
    # the model's natural shape: the parent's run on one line, its
    # sub-entries' chain on a same-range line — one chunk set, one
    # line's semantics (a canary repair loop oscillated between this
    # merged shape and the split shape until merging was allowed)
    routing = await route_nested(
        '0 a\n1-2 b.0\n1-2 b.0.c.0-1\n3 b.1\n4 -\nb: 2\nb.0.c: 2')
    assert [(g.unit.path, g.item, g.parent, g.chunk_ids)
            for g in routing.groups] == [
        ('basic_info', None, None, [0]),
        ('jobs', 0, None, [1, 2]),
        ('jobs.roles', 0, 0, [1, 2]),
        ('jobs.roles', 1, 0, [1, 2]),
        ('jobs', 1, None, [3])]


async def test_comma_joined_chains_share_one_parent_destination():
    routing = await route_nested(
        '0 a\n1-2 b.0.c.0,b.0.c.1\n3 b.1\n4 -\nb: 2\nb.0.c: 2')
    chain = [g for g in routing.groups if g.parent is not None]
    parents = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [(g.item, g.parent) for g in chain] == [(0, 0), (1, 0)]
    assert [(g.item, g.chunk_ids) for g in parents] == [
        (0, [1, 2]), (1, [3])]


async def test_chain_lines_inside_the_parent_run_stay_the_fine_partition():
    # the model splits a parent's sub-entries across the run the
    # parent's own line keeps whole — legal, and the fine partition
    # STANDS: each sub-entry keeps its own slice as its own segment, so
    # the executor fans them out as separate concurrent calls (merging
    # them into the parent's line would collapse the run into one whole
    # call)
    routing = await route_nested(
        '0 a\n1-3 b.0\n2 b.0.c.0\n3 b.0.c.1\n4 -\nb: 1\nb.0.c: 2')
    assert [(g.unit.path, g.item, g.parent, g.chunk_ids)
            for g in routing.groups] == [
        ('basic_info', None, None, [0]),
        ('jobs', 0, None, [1, 2, 3]),
        ('jobs.roles', 0, 0, [2]),
        ('jobs.roles', 1, 0, [3])]
    jobs_roles = [a for a in routing.raw['assignments']
                  if a['unit'] == 'jobs.roles']
    assert [(a['item'], a['parent'], a['chunks']) for a in jobs_roles] == \
        [(0, 0, [2]), (1, 0, [3])]


async def test_standalone_chain_line_is_its_parent_coverage_there():
    # a chain line no parent run contains covers those chunks FOR the
    # parent (the prompt's promise): the implicit parent destination
    # stays, parent and sub-entry each see their own scope, one group
    # apiece — no repair round
    routing = await route_nested(
        '0 a\n1-2 b.0\n3 b.0.c.0\n4 -\nb: 1\nb.0.c: 1')
    assert [(g.unit.path, g.item, g.parent, g.chunk_ids)
            for g in routing.groups] == [
        ('basic_info', None, None, [0]),
        ('jobs', 0, None, [1, 2, 3]),
        ('jobs.roles', 0, 0, [3])]


async def test_chain_line_overlapping_another_units_run_is_named():
    # the sub-entry's slice rides a run that carries a DIFFERENT parent
    # instance — a genuine two-claim conflict, the overlap repair names it
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 b.1\n2 b.0.c.0\n4 -\nb: 2\nb.0.c: 1'),
        agent_result('0 a\n1 b.1\n2-3 b.0.c.0\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                chunks=NESTED_CHUNKS)
    assert 'overlaps line' \
        in runner.calls[1]['feedback'][0].message


async def test_parent_line_overlapping_its_chains_names_the_remedy():
    # the top-down draft — a parent line covering its own chain lines —
    # gets the concrete remedy: the parent keeps only the chunks outside
    # its chains, and the count must match the items drawn
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 b.0-1\n2 b.0.c.0\n4 -\nb: 2\nb.0.c: 1'),
        agent_result('0 a\n1 b.0\n2 b.0.c.0\n3 b.1\n4 -\nb: 2\nb.0.c: 1'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                chunks=NESTED_CHUNKS)
    message = runner.calls[1]['feedback'][0].message
    assert 'outside its chains (1, 3)' in message
    assert 'count line must match' in message


async def test_sub_slice_overlapping_its_sibling_names_the_boundary():
    # two sub-entry slices of the same parent instance overlap — the
    # earlier one ends where the later begins
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0\n1-2 b.0.c.0\n2 b.0.c.1\n3 -\n4 -\nb: 1\nb.0.c: 2'),
        agent_result('0 a\n1 b.0.c.0\n2 b.0.c.1\n3 -\n4 -\nb: 1\nb.0.c: 2'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                chunks=NESTED_CHUNKS)
    assert 'end the earlier one at 1, where b.0.c.1 begins' \
        in runner.calls[1]['feedback'][0].message


async def test_a_no_op_rewrite_is_named_on_the_next_repair():
    # a replayed map ("-" removes a line, "+" adds it right back) is
    # the burn signature: the feedback names the no-op and points at
    # the full-map escape instead of re-firing the bare error
    noop = agent_result('- 1 b.0\n+ 1 b.0')
    runner = ScriptedRunner(BAD_MAP, noop, noop, noop, noop)
    with pytest.raises(RouterError):
        await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert 'changed nothing' in runner.calls[2]['feedback'][0].message
    assert 'changed nothing' not in runner.calls[1]['feedback'][0].message


async def test_exhaustion_with_a_valid_earlier_round_settles_on_it():
    # a resolution round's fix keeps thrashing to the budget's end:
    # the last parse-valid round — the map whose recount disagreed —
    # finalizes instead of raising. A flawed 200 beats a 500
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 0'),
        agent_result('b: 2\n0 b.0,b.1'),  # recount claims chunk 0: unspliceable
        agent_result('x'), agent_result('y'), agent_result('x'), agent_result('y'))
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 6
    assert routing.raw['counts'] == {'jobs': 0}


async def test_exhaustion_passes_count_flaws_to_the_arbitration():
    # never valid, never replayed (every reply a different wrong map):
    # at the budget's end a declared-vs-used mismatch is the
    # arbitration's number, not a reason to raise
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0\n2-3 -\nb: 3'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 4'),
        agent_result('1 b.0\n0 a\n2-3 -\nb: 3'),
        agent_result('0 a\n1 b.0\n2-3 -\nb: 5'),
        agent_result('1 b.0\n2-3 -\n0 a\nb: 3'),
    )
    routing = await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 5
    assert routing.raw['counts'] == {'jobs': 3}


async def test_per_sub_entry_declarations_fold_into_the_parent_count():
    # the model's natural mirror of item counting: 'c.0.d.0: 1' lines —
    # tolerated, and folded: with no explicit parent declaration, the
    # highest numbered sub-entry is the count (map-first)
    routing = await route_nested(
        '0 a\n1 b.0.c.0\n2 b.0.c.1\n3 b.1\n4 -\nb: 2\n'
        'b.0.c.0: 1 @90%\nb.0.c.1: 1 @90%')
    assert routing.raw['nested_counts'] == {'jobs.roles': {0: 2}}


async def test_per_sub_entry_declarations_yield_to_the_explicit_count():
    routing = await route_nested(
        '0 a\n1 b.0.c.0\n2 b.0.c.1\n3 b.1\n4 -\nb: 2\n'
        'b.0.c: 2\nb.0.c.0: 1')
    assert routing.raw['nested_counts'] == {'jobs.roles': {0: 2}}


async def test_comma_joined_parent_and_ranged_sub_entries_folds_to_the_chain():
    # the model comma-joins a parent instance and its ranged sub-entries
    # ('6 c.0,d.0-3') — one more spelling of the chain, measured on a
    # live draw; it folds into the parent destination the line claims
    routing = await route_nested(
        '0 a\n1-4 b.0,c.0-1\nb: 1\nb.0.c: 2')
    assert routing.raw['nested_counts'] == {'jobs.roles': {0: 2}}
    assert [(g.unit.path, g.item, g.parent, g.chunk_ids)
            for g in routing.groups] == [
        ('basic_info', None, None, [0]),
        ('jobs', 0, None, [1, 2, 3, 4]),
        ('jobs.roles', 0, 0, [1, 2, 3, 4]),
        ('jobs.roles', 1, 0, [1, 2, 3, 4])]


async def test_comma_joined_nested_token_with_two_parents_stays_named():
    # the fold needs ONE parent destination to attach to; two on the
    # line is ambiguous and keeps the named error
    runner = ScriptedRunner(
        agent_result('0 a\n1 b.0,b.1,c.0-1\n2-4 -\nb: 2\nb.0.c: 2'),
        agent_result('0 a\n1-2 b.0.c.0,b.0.c.1\n3 b.1\n4 -\n'
                     'b: 2\nb.0.c: 2'))
    await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                chunks=NESTED_CHUNKS)
    assert 'rides its parent' in runner.calls[1]['feedback'][0].message



# --- declared-undrawn units: a count declared, no lines drawn ---
# (the first-pass maps declare the numbers and skip the drawing; a
# fresh recount counts each unit and quotes its instances' openings,
# and code anchors the quotes and lays the lines — _recount_undrawn /
# _fold_undrawn)

async def test_declared_undrawn_unit_is_recounted_and_adopted():
    # b declared, nothing drawn, its material in chunks the map wrote
    # off as irrelevant — the merged round also re-asks the omitted
    # summary (c), all in the one fresh conversation
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 2'),
        agent_result('b: 2\n第一段：腾讯\n第二段：阿里\nc: 0'))
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=CHUNKS)
    assert len(runner.calls) == 2  # the recount adopted — no repair round
    assert runner.calls[1]['feedback'] is None  # a fresh conversation
    assert 'RECOUNT MERGED' in runner.calls[1]['instructions']
    assert 'RECOUNT ZERO' in runner.calls[1]['instructions']
    assert routing.raw['counts'] == {'jobs': 2, 'summary': 0}
    assert [(g.item, g.chunk_ids) for g in routing.groups
            if g.unit.path == 'jobs'] == [(0, [1]), (1, [2])]


async def test_undrawn_instances_sharing_one_chunk_claim_it_together():
    # several instances inside one chunk (a dense list the chunk
    # boundaries cannot separate) — they all claim the same chunk
    chunks = ['姓名张三', '腾讯、阿里', '无关页脚']
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 -\nb: 2'),
        agent_result('b: 2\n腾讯\n阿里\nc: 0'))
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=chunks)
    assert len(runner.calls) == 2
    jobs = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [(g.item, g.chunk_ids) for g in jobs] == [(0, [1]), (1, [1])]


async def test_nested_undrawn_chains_partition_the_parent_run():
    # sub-entries declared under a parent instance but never drawn:
    # the recount's openings anchor inside the parent's run, and each
    # sub-entry claims its own slice
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 b.0\n3 b.1\n4 -\nb: 2\nb.0.c: 2'),
        agent_result('b.0.c: 2\n公司甲·工程师\n公司甲·经理'))
    routing = await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                          chunks=NESTED_CHUNKS)
    assert len(runner.calls) == 2
    roles = [g for g in routing.groups if g.unit.path == 'jobs.roles']
    assert [(g.item, g.parent, g.chunk_ids) for g in roles] == \
        [(0, 0, [1]), (1, 0, [2])]
    assert routing.raw['nested_counts'] == {'jobs.roles': {0: 2}}


async def test_undrawn_chains_leave_the_parent_runs_own_head_and_tail():
    chunks = ['姓名张三', '公司甲', '公司甲·工程师', '公司甲·经理',
              '公司甲·尾注', '无关页脚']
    runner = ScriptedRunner(
        agent_result('0 a\n1-4 b.0\n5 -\nb: 1\nb.0.c: 2'),
        agent_result('b.0.c: 2\n公司甲·工程师\n公司甲·经理'))
    routing = await route(runner, payload=PAYLOAD, units=NESTED_UNITS,
                          chunks=chunks)
    assert len(runner.calls) == 2
    roles = [g for g in routing.groups if g.unit.path == 'jobs.roles']
    assert [(g.item, g.chunk_ids) for g in roles] == [(0, [2]), (1, [3])]
    jobs = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [(g.item, g.chunk_ids) for g in jobs] == [(0, [1, 2, 3, 4])]


async def test_undrawn_recount_number_overrides_the_declaration():
    # the recount counted fewer instances than the map declared — the
    # fresh read is authoritative, the declaration follows
    runner = ScriptedRunner(
        agent_result('0 a\n1-3 -\nb: 2'),
        agent_result('b: 1\n第一段：腾讯\nc: 0'))
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert routing.raw['counts'] == {'jobs': 1, 'summary': 0}
    assert [g.chunk_ids for g in routing.groups
            if g.unit.path == 'jobs'] == [[1]]


async def test_geometry_errors_keep_the_undrawn_family_on_diff_rounds():
    # a map mixing a geometry flaw with the declared-undrawn error
    # stays on the diff path — the recount cannot repair overlap
    runner = ScriptedRunner(
        agent_result('0-1 a\n1-3 -\nb: 2'),  # overlap + declared-undrawn
        agent_result('0 a\n1 b.0\n2 b.1\n3 -\nb: 2'))
    await route(runner, payload=PAYLOAD, units=UNITS, chunks=CHUNKS)
    assert len(runner.calls) == 2
    assert 'overlaps' in runner.calls[1]['feedback'][0].message
    assert runner.calls[1]['instructions'] == ''  # a repair, not a recount


async def test_undrawn_merged_opening_quote_claims_the_chunk_for_all():
    # the recount may answer one quote for several instances whose
    # openings the chunk listing holds on one line (the dense list) —
    # every item claims that chunk, the co-chunked shared form
    chunks = ['姓名张三', '已取得律师执业资格、证券从业资格', '无关页脚']
    runner = ScriptedRunner(
        agent_result('0 a\n1-2 -\nb: 2'),
        agent_result('b: 2\n已取得律师执业资格、证券从业资格'))
    routing = await route(runner, payload=PAYLOAD, units=SHARED_UNITS,
                          chunks=chunks)
    assert len(runner.calls) == 2
    jobs = [g for g in routing.groups if g.unit.path == 'jobs']
    assert [(g.item, g.chunk_ids) for g in jobs] == [(0, [1]), (1, [1])]


def test_map_text_round_trips_through_the_parser():
    # the state-to-text rebuild feeds the next round's diff base — its
    # correctness is no longer re-checked by a hot-path parse, so the
    # round trip is pinned here: parse → render → parse reproduces the
    # state exactly (segments, counts, nested, budgets included)
    from xtremeparse.router import _codes, _map_text, _parse
    text = ('0 a\n1-2 b.0\n2 b.0.c.0\n3 b.1\n4 -\n'
            'b: 2 @90%,90%\nb.0.c: 1 @100%\nd: 1 @15%')
    by_code = _codes(NESTED_UNITS)
    errors, counts, nested, derived, segments, budgets = \
        _parse(text, by_code, len(NESTED_CHUNKS))
    assert not errors
    rendered = _map_text(segments, counts, nested, derived, budgets, by_code)
    errors, counts2, nested2, derived2, segments2, budgets2 = \
        _parse(rendered, by_code, len(NESTED_CHUNKS))
    assert not errors
    assert (segments2, counts2, nested2, derived2, budgets2) == \
        (segments, counts, nested, derived, budgets)


def test_map_text_round_trips_derived_units():
    from xtremeparse.router import _codes, _map_text, _parse
    text = '0 a\n1 b.0\n2 b.1\n3 -\nb: 2\nc = b @30%'
    by_code = _codes(SHARED_UNITS)
    errors, counts, nested, derived, segments, budgets = \
        _parse(text, by_code, len(CHUNKS))
    assert not errors
    rendered = _map_text(segments, counts, nested, derived, budgets, by_code)
    errors, counts2, nested2, derived2, segments2, budgets2 = \
        _parse(rendered, by_code, len(CHUNKS))
    assert not errors
    assert counts2 == counts == {'b': 2}
    assert derived2 == derived == {'c': 'b'}
