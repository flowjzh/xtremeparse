"""Boundary probes for the count arbitration: the fresh-conversation
diff check that settles a declared count against its extraction."""

import pytest

from xtremeparse.arbitration import (_ARBITRATION_CHECK,
                                     _ARBITRATION_SKELETON,
                                     arbitrate_extraction)
from tests.helpers import ScriptedRunner, agent_result

ITEMS_3 = [{'company': '腾讯'}, {'company': '阿里'}, {'company': '华为'}]
DOC_3 = '第一段：腾讯 后端\n第二段：阿里 高级\n第三段：美团 Deliver'
ITEMS_2 = [{'company': '腾讯'}, {'company': '阿里'}]
DOC_2 = '第一段：腾讯 后端\n第二段：阿里 高级'


async def test_arbitration_confirms_a_faithful_list():
    # nothing missing, nothing extra — the extraction stands as extracted
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\nextra:\n')),
        items=ITEMS_2, actual=2, payload=DOC_2)
    assert revised == 2


@pytest.mark.parametrize('extra', ['华为', '- 华为'])
async def test_arbitration_revises_by_missing_and_extra(extra):
    # one document instance the list lacks, one list entry the document
    # does not contain — the count moves by their difference; the "- x"
    # variant echoes the prompt's own bullet marker, which is
    # enumeration, not quoted content
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result(
            f'missing:\n第三段：美团 Deliver\n\nextra:\n{extra}')),
        items=ITEMS_3, actual=3, payload=DOC_3)
    assert revised == 3  # 3 + 1 missing - 1 extra


async def test_arbitration_voids_extras_the_document_contains():
    # a list entry the document itself contains cannot be extra — the
    # verdict is voided, not trimmed (a model correcting a pre-filled
    # form relocates every list entry to extra even though the document
    # holds them)
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result(
            'missing:\n\nextra:\n- 腾讯\n- 阿里')),
        items=ITEMS_2, actual=2, payload=DOC_2)
    assert revised is None


def test_check_template_keeps_the_list_out_of_the_verdict_sections():
    # the answer skeleton carries no holes: a list spliced into a
    # verdict section is unrepresentable — the pre-filled-verdict
    # failure mode (measured on a live trace) is the flip this guards
    assert 'missing' not in _ARBITRATION_CHECK.lower()
    assert _ARBITRATION_CHECK.strip().endswith('{list}')
    sections = _ARBITRATION_SKELETON.split('missing:', 1)[1]
    assert sections.split('extra:', 1)[0].strip() == ''


async def test_arbitration_voids_unanchored_missing_quotes():
    # a missing quote the document does not contain is a hallucination
    # — the whole verdict is voided, not merely trimmed
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\n编造的不存在条目\nextra:\n')),
        items=[{'company': '腾讯'}], actual=1, payload=DOC_2)
    assert revised is None


async def test_arbitration_voids_extra_quotes_outside_the_list():
    # an extra must match a list entry it rejects — otherwise it is
    # indistinguishable from a hallucinated line
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\nextra:\n编造的不存在条目')),
        items=[{'company': '腾讯'}], actual=1, payload=DOC_2)
    assert revised is None


async def test_arbitration_voids_a_both_empty_verdict_on_an_empty_extraction():
    # an empty extraction must not be blessed by a quoteless verdict —
    # the case the both-empty gate (see arbitrate_extraction) exists for
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\n\nextra:\n')),
        items=[], actual=0, payload=DOC_2)
    assert revised is None


async def test_arbitration_still_revises_an_empty_extraction_up():
    # the void must not over-reach: missing quotes anchored to the
    # document are evidence — they revise the collapsed count up so the
    # mend path targets what the document holds
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\n第三段：美团 Deliver\nextra:\n')),
        items=[], actual=0, payload=DOC_3)
    assert revised == 1
