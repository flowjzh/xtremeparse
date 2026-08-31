"""Extractor: the long-lived facade composing the whole pipeline.

One instance carries the scheduler across extractions; every extraction
chunks, decomposes, routes, fans out, merges and self-corrects against
the injected validator, then reports leniently.
"""

from __future__ import annotations

from xtremeparse.chunking import MAX_CHARS, chunk_text, normalize_newlines
from xtremeparse.contracts import AgentRunner, ExtractionResult, Trace, Validator
from xtremeparse.corrections import MAX_ROUNDS, correct, count_issues
from xtremeparse.executor import BATCH_BUDGET_CAP, execute
from xtremeparse.prompting import (SPECIALIST_PLACEHOLDERS, check_placeholders,
                                   estimate_tokens, provenance, shared_payload)
from xtremeparse.router import (RECOUNT_PLACEHOLDERS, ROUTE_PLACEHOLDERS, route)
from xtremeparse.scheduling import TaskScheduler
from xtremeparse.units import decompose

DEFAULT_CONCURRENCY = 8


class Extractor:
    """Compose chunk → route → fan out → merge → correct for one schema
    at a time. Construct once, extract many times — within one event
    loop (the scheduler's semaphore binds to it)."""

    def __init__(self, agent_runner: AgentRunner, *, router_runner: AgentRunner = None,
                 scheduler: TaskScheduler = None,
                 router_scheduler: TaskScheduler = None,
                 max_concurrency: int = DEFAULT_CONCURRENCY,
                 unit_strategy: dict = None,
                 max_rounds: int = MAX_ROUNDS, max_chars: int = MAX_CHARS,
                 output_budgets: dict = None,
                 router_instructions: str = None,
                 recount_instructions: str = None,
                 specialist_instructions: str = None,
                 batch_cap: int = BATCH_BUDGET_CAP):
        self.runner = agent_runner
        self.router_runner = router_runner  # serial slice call; may be a
        # different (stronger index-map) model than the concurrent fleet
        self.router_scheduler = router_scheduler  # …and then its own quota
        self.scheduler = scheduler or TaskScheduler(
            max_concurrency)  # injected: one process-global budget for
        # every LLM call (a TokenRateScheduler); the bare
        # default is a plain concurrency cap for tests and small hosts
        self.unit_strategy = unit_strategy or {}
        self.max_rounds = max_rounds
        self.max_chars = max_chars
        self.batch_cap = batch_cap
        self.output_budgets = output_budgets or {}  # per-path override of
        # the router's own @<n> declarations (unit_strategy precedent)
        # prompt template overrides; each must carry its placeholders
        # (docs/prompting.md) — checked here so a broken template fails
        # construction, not its first extraction
        self.router_instructions = router_instructions
        self.recount_instructions = recount_instructions
        self.specialist_instructions = specialist_instructions
        for name, template, required in (
                ('router_instructions', router_instructions, ROUTE_PLACEHOLDERS),
                ('recount_instructions', recount_instructions, RECOUNT_PLACEHOLDERS),
                ('specialist_instructions', specialist_instructions,
                 SPECIALIST_PLACEHOLDERS)):
            if template is not None:
                check_placeholders(template, required, name)
        self.prompts = {'router': provenance(router_instructions),
                        'recount': provenance(recount_instructions),
                        'specialist': provenance(specialist_instructions)}

    async def extract(self, text: str, schema: dict,
                      validator: Validator) -> ExtractionResult:
        """Best-effort structured extraction; never raises on bad data."""
        text = normalize_newlines(text)
        payload = shared_payload(text, schema)
        chunks = chunk_text(text, max_chars=self.max_chars)
        units = decompose(schema)
        routing = execution = None
        if chunks and units:
            route_task = await (self.router_scheduler or self.scheduler).start_task(
                route(self.router_runner or self.runner, payload=payload,
                      units=units, chunks=chunks,
                      instructions=self.router_instructions,
                      recount_instructions=self.recount_instructions),
                # payload + text: router instructions re-embed the chunk listing
                estimated_tokens=estimate_tokens(payload, text))
            routing = await route_task
            budgets = {**(routing.budgets or {}), **self.output_budgets}
            execution = await execute(self.runner, routing, payload=text,
                                      scheduler=self.scheduler,
                                      unit_strategy=self.unit_strategy,
                                      budgets=budgets,
                                      specialist_instructions=self.specialist_instructions,
                                      batch_cap=self.batch_cap)
            counts = routing.raw['counts']  # declared item counts are
            # map-validated ground truth — a short array is collapsed
            # instances, invisible to schema validation; re-run it
            data, issues, rounds = await correct(
                self.runner, execution, payload=text,
                scheduler=self.scheduler, max_rounds=self.max_rounds,
                specialist_instructions=self.specialist_instructions,
                validator=lambda d: list(validator(d) or [])
                + count_issues(counts, d))
        else:  # nothing routable: skip the agent fleet entirely
            data, issues, rounds = {}, list(validator({}) or []), []
        trace = Trace(chunks=chunks,
                      router=routing.raw if routing else None,
                      groups=[{'unit': c.unit.path, 'kind': c.unit.kind, 'item': c.item,
                               'strategy': c.strategy, 'chunk_ids': c.chunk_ids,
                               'budget': c.budget, 'batch': c.batch or None}
                              for c in execution.calls] if execution else [],
                      corrections=[{'unit_path': r.unit_path, 'item': r.item,
                                    'issue_paths': r.issue_paths} for r in rounds],
                      prompts=dict(self.prompts))
        return ExtractionResult(data, issues, trace)
