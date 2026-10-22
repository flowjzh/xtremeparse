"""Prompt composition shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
from string import Formatter

from xtremeparse.contracts import JSONSchema
from xtremeparse.verbatim import VERBATIM_CUE, VERBATIM_KEY


SPECIALIST_INSTRUCTIONS = '''You are a structured-extraction specialist. Extract this unit's fields
from the assigned material below. The full document in the shared
context provides background; extract only what the unit card asks
for. For a non-repeating unit, a card field may sit outside the
assigned slice — take the value the shared document shows for that
field. Output only the structured data of the output schema stated at
the end of this message.

Unit:

{card}'''

# appended to array-shaped calls (whole runs and budget batches): left
# alone the model summarizes a multi-instance material into one entry —
# the collapsed-array failure mode every downstream check starts from.
# The card side of the same coin: the mirror failure splits one
# instance's own enumeration into several entries (a promotion route
# drawn as sub-stints), so instanceship stays the card's call
WHOLE_ARRAY_ADDENDUM = '''

The output schema is an array: return every instance the assigned
material holds as its own entry — never merge instances into one
entry, never summarize. An entry the unit card would not fill is not
an instance — never pad the array with one.'''

SPECIALIST_PLACEHOLDERS = frozenset({'card'})


def check_placeholders(template: str, required, name: str) -> None:
    """Fail fast on an override template missing a placeholder the
    pipeline will substitute — str.format would silently render a prompt
    without its chunk listing or legend instead of erroring."""
    present = {field for _, field, _, _ in Formatter().parse(template) if field}
    if missing := sorted(set(required) - present):
        raise ValueError(f'{name} template misses placeholders: {missing}')


def provenance(template) -> str:
    """Trace marker for a prompt slot: 'default' or a short hash of the
    host's override — a regression stays attributable to whose prompt
    produced it."""
    return 'default' if template is None \
        else f'#{hashlib.sha1(template.encode()).hexdigest()[:8]}'


def overall(schema: JSONSchema) -> str | None:
    """The schema's document-level instruction — the root description,
    which the router renders as the "Overall Instruction" block. The
    one derivation shared by the extractor and the eval probes."""
    return schema.get('description') or None


def prompt_schema(node):
    """The model-facing copy of a schema: the ``x-verbatim`` keyword is
    dropped — it is an enforcement key, not model vocabulary — and
    folded into its description as the inline ``@verbatim`` cue. Both
    planes render the cue: the router's budget clause prices a verbatim
    copy, and the specialist's output schema carries the copy rule the
    model reads. The caller's schema is untouched; enforcement still
    reads the keyword there."""
    if isinstance(node, dict):
        out = {k: prompt_schema(v) for k, v in node.items()
               if k != VERBATIM_KEY}
        if VERBATIM_KEY in node:
            desc = out.get('description', '')
            if VERBATIM_CUE not in desc:
                out['description'] = f'{desc} {VERBATIM_CUE}'.strip()
        return out
    if isinstance(node, list):
        return [prompt_schema(v) for v in node]
    return node


def schema_head(schema: JSONSchema) -> str:
    """The schema block the router-plane prompts render as their own
    stable user message — the full schema serialized as JSON, field
    descriptions included (the verbatim contract rides its descriptions
    as ``@verbatim`` cues, never as the enforcement keyword); the
    router-plane calls read it beside the unit-card legend. A host
    override that cannot split messages still gets it inline ahead of
    the chunks listing (see router._chunks_section). Not carried
    in ``content``: content is the document text alone on every call
    (router, recounts and specialists alike), the one byte-identical
    system message the provider's cache serves to the whole fleet —
    dashscope's explicit cache truncates only at message boundaries, so
    a text+schema system message could never serve the specialists'
    text-only one."""
    return ('JSON Schema:\n'
            + json.dumps(prompt_schema(schema), ensure_ascii=False,
                         sort_keys=True))


def router_tokens(text: str, head: str, units=()) -> int:
    """The router call's tps seed, shared by the extractor and the eval
    probes: the text rides content and is re-embodied by the chunks
    part, beside the schema head and the unit cards."""
    return estimate_tokens(text, head, text, *(u.card for u in units))


def value_chars(value) -> int:
    """The characters of a value's leaf content — strings by length,
    containers by the sum of their members' value characters; keys and
    punctuation never count. This is the unit a budget declares and is
    audited in."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(value_chars(v) for v in value.values())
    if isinstance(value, list):
        return sum(value_chars(v) for v in value)
    return len(str(value))


CHARS_PER_TOKEN = 1.6  # zh-heavy calibration, same heuristic as argus


def estimate_tokens(*texts: str) -> int:
    """Seed for a TokenRateScheduler's tps reservation; the runner's
    actual-usage report corrects it afterwards."""
    return round(sum(len(t) for t in texts) / CHARS_PER_TOKEN)
