"""Prompt composition shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
from string import Formatter

from xtremeparse.contracts import JSONSchema


SPECIALIST_INSTRUCTIONS = '''You are a structured-extraction specialist. Extract this unit's fields
from the assigned material below. The full document in the shared
context provides background; extract only what the unit card asks
for, from the assigned material. Output only the structured data of
the output schema stated at the end of this message.

Unit:

{card}'''

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


def shared_payload(text: str, schema: JSONSchema) -> str:
    """The byte-identical content prefix the ROUTER's call carries —
    full text plus the full schema, whose field descriptions feed the
    budget ratios. Specialists carry the text alone (their own partial
    schema rides per call beside the unit card), so the schema's bytes
    are paid once, by the one call that reads them."""
    return f'{text}\n\n---\nJSON Schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}'


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
