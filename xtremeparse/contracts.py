"""Neutral contracts between xtremeparse and the host application.

The library never imports a schema vendor, an agent framework, or a
validator. Everything crosses the border as plain data: JSON Schema in,
JSON-shaped dict out, issues and agent calls behind the protocols below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

# A plain JSON Schema mapping (``schema.to_json_schema()`` output, or any
# hand-written schema with the same shape).
JSONSchema = dict[str, Any]


@runtime_checkable
class Issue(Protocol):
    """A validation problem found in extracted data.

    Structural protocol: any object with these attributes satisfies it,
    zero adapter code. ``expected``/``got`` are optional free-form context
    (hosts pass lists for enums, arbitrary values, sentinels for absent).
    """

    path: str
    message: str
    code: str
    expected: object
    got: object


# Validates complete extracted data, returns the issues the host considers
# error-level. Feeding only errors is the host's severity policy: the
# correction loop retries whatever it receives (an issue carrying
# `report_only` is surfaced, never retried).
Validator = Callable[[dict], Sequence[Issue]]


@dataclass
class AgentResult:
    """One agent turn's outcome.

    ``data`` is already coerced to the requested ``result_schema`` by the
    adapter (e.g. PydanticAI structured output). ``history`` is the opaque
    message history for multi-turn continuation; the library passes it back
    verbatim on correction rounds. Usage/cost telemetry is deliberately not
    here — the adapter layer instruments its own agent calls.
    """

    data: Any
    history: list


@runtime_checkable
class AgentRunner(Protocol):
    """The single agent-call abstraction the library owns.

    The host adapts this to its agent framework (PydanticAI today, anything
    else tomorrow). Prompt layout is frozen for cache reasons:
    ``instructions`` carries the unit's semantic card and ``content`` the
    shared, byte-identical payload (full text + full schema) that makes the
    provider's KV cache hit across every call of one extraction;
    ``scope`` carries this call's private slice — the routed chunks of one
    group (None for the router). On ``history`` rounds the transcript
    already carries the card and scope, so the caller passes ``''`` for
    both — re-sending them would re-bill the same text, and only the
    per-round ``feedback`` is fresh. ``result_schema`` is the JSON Schema the
    returned ``data`` must satisfy; ``tools``/``history``/``feedback``
    serve the multi-turn correction loop.
    """

    async def run(
        self,
        *,
        instructions: str,
        result_schema: JSONSchema,
        content: str,
        scope: Optional[str] = None,
        tools: Optional[list] = None,
        history: Optional[list] = None,
        feedback: Optional[Sequence[Issue]] = None,
    ) -> AgentResult:
        ...


@dataclass
class Trace:
    """Orchestration record: what was chunked, routed, run and retried.

    Populated by the pipeline stages; read by eval suites and telemetry.
    Token/latency accounting of agent calls stays with the adapter — this
    trace records orchestration events only. ``prompts`` marks each
    prompt slot 'default' or with a short hash of the host's override
    template (see prompting.provenance) — regressions stay attributable
    to whose prompt produced them.
    """

    chunks: list = field(default_factory=list)
    router: Optional[dict] = None
    groups: list = field(default_factory=list)
    corrections: list = field(default_factory=list)
    recounts: list = field(default_factory=list)
    prompts: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Extraction never raises on bad data — it reports.

    ``data`` is the best-effort merged result (missing pieces stay absent),
    ``issues`` the unresolved error-level issues from the last validation,
    both lenient by design. Strictness is the caller's policy.
    """

    data: dict
    issues: list
    trace: Trace
