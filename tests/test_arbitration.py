"""Boundary probes for the count arbitration: the fresh-conversation
diff check that settles a declared count against its extraction."""

from xtremeparse.arbitration import arbitrate_extraction
from tests.helpers import ScriptedRunner, agent_result


async def test_arbitration_confirms_a_faithful_list():
    # nothing missing, nothing extra — the extraction stands as extracted
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\nextra:\n')),
        items=[{'company': '腾讯'}, {'company': '阿里'}], actual=2,
        payload='第一段：腾讯 后端\n第二段：阿里 高级')
    assert revised == 2


async def test_arbitration_revises_by_missing_and_extra():
    # one document instance the list lacks, one list entry the document
    # does not contain — the count moves by their difference
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result(
            'missing:\n第三段：美团 Deliver\n\nextra:\n阿里')),
        items=[{'company': '腾讯'}, {'company': '阿里'}], actual=2,
        payload='第一段：腾讯 后端\n第二段：阿里 高级\n第三段：美团 Deliver')
    assert revised == 2


async def test_arbitration_voids_unanchored_missing_quotes():
    # a missing quote the document does not contain is a hallucination
    # — the whole verdict is voided, not merely trimmed
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\n编造的不存在条目\nextra:\n')),
        items=[{'company': '腾讯'}], actual=1, payload='第一段：腾讯 后端')
    assert revised is None


async def test_arbitration_voids_extra_quotes_outside_the_list():
    # an extra must match a list entry it rejects — otherwise it is
    # indistinguishable from a hallucinated line
    revised, _ = await arbitrate_extraction(
        ScriptedRunner(agent_result('missing:\nextra:\n编造的不存在条目')),
        items=[{'company': '腾讯'}], actual=1, payload='第一段：腾讯 后端')
    assert revised is None
