"""Prompt composition shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
from string import Formatter

from xtremeparse.contracts import JSONSchema


SPECIALIST_INSTRUCTIONS = '''You are a structured-extraction specialist. Extract this unit's fields
from the assigned material below. The full document and JSON Schema in the
shared context provide background; extract only what the unit card asks
for, from the assigned material. Output only the structured data.

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
    """The byte-identical content prefix every agent call of one
    extraction shares (full text + full schema), so the provider's KV
    cache hits across the router and all specialists behind it."""
    return f'{text}\n\n---\nJSON Schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}'


def json_len(value) -> int:
    """The compact serialized length of a value — keys and punctuation
    included — exactly what a specialist types and pays decode for."""
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


CHARS_PER_TOKEN = 1.6  # zh-heavy calibration, same heuristic as argus


def estimate_tokens(*texts: str) -> int:
    """Seed for a TokenRateScheduler's tps reservation; the runner's
    actual-usage report corrects it afterwards."""
    return round(sum(len(t) for t in texts) / CHARS_PER_TOKEN)
