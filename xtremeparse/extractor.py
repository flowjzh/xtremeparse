"""Extractor: the long-lived facade composing the whole pipeline.

One instance carries the scheduler across extractions; every extraction
chunks, decomposes, routes, fans out, merges and self-corrects against
the injected validator, then reports leniently.
"""

from __future__ import annotations

from xtremeparse.arbitration import arbitrate_extraction
from xtremeparse.chunking import MAX_CHARS, chunk_text, normalize_newlines
from xtremeparse.contracts import AgentRunner, BATCH_BUDGET_CAP, ExtractionResult, Trace, Validator
from xtremeparse.corrections import (MAX_ROUNDS, correct, count_issues,
                                     count_mismatches)
from xtremeparse.executor import execute, values_from_calls
from xtremeparse.merge import merge
from xtremeparse.prompting import (SPECIALIST_PLACEHOLDERS, check_placeholders,
                                   estimate_tokens, provenance, shared_payload)
from xtremeparse.router import (RECOUNT_PLACEHOLDERS, ROUTE_PLACEHOLDERS,
                                route)
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
        recounts, disputed = [], set()
        if chunks and units:
            # the router slice rides its own (stronger-index) model and
            # quota when one is configured — every router-plane call
            # below resolves the same fallback pair
            runner = self.router_runner or self.runner
            scheduler = self.router_scheduler or self.scheduler
            tokens = estimate_tokens(payload, text)  # payload + text:
            # router instructions re-embed the chunk listing
            route_task = await scheduler.start_task(
                route(runner, payload=payload, units=units, chunks=chunks,
                      instructions=self.router_instructions,
                      recount_instructions=self.recount_instructions),
                estimated_tokens=tokens)
            routing = await route_task
            budgets = {**(routing.budgets or {}), **self.output_budgets}
            execution = await execute(self.runner, routing, payload=text,
                                      scheduler=self.scheduler,
                                      unit_strategy=self.unit_strategy,
                                      budgets=budgets,
                                      specialist_instructions=self.specialist_instructions,
                                      batch_cap=self.batch_cap)
            counts = dict(routing.raw['counts'])  # declared item counts are
            # map-validated ground truth — a short array is collapsed
            # instances, invisible to schema validation; re-run it.
            # Working copy: arbitration may revise a disputed count;
            # raw keeps the original declaration for the trace
            values = values_from_calls(execution.calls)
            if mismatches := count_mismatches(counts, merge(values)):
                # declared vs actual disagree before any correction round
                # — arbitrate_extraction owns the why. This seam keeps
                # only the caller-side policy: an unanchored verdict
                # stays out of the repair loop as a report-only issue,
                # but for over-counts alone — a shortfall keeps its
                # declared count, the mend path stands. Every check
                # starts before the first one returns: the calls are
                # independent, the scheduler exists to overlap them
                arb_tokens = estimate_tokens(payload)
                pending = []
                for unit_path, (declared, actual) in sorted(
                        mismatches.items()):
                    task = arbitrate_extraction(
                        runner, items=values.get(unit_path) or [],
                        actual=actual, payload=payload)
                    # the check prompt is payload + bounded openings
                    # (ARBITRATION_LIST_CAP), not the full document again
                    pending.append((unit_path, declared, actual,
                                    await scheduler.start_task(
                                        task,
                                        estimated_tokens=arb_tokens)))
                for unit_path, declared, actual, arb_task in pending:
                    revised, raw = await arb_task
                    if revised is not None:
                        counts[unit_path] = revised
                    elif actual > declared:
                        # an unsettled over-count: the declaration stays
                        # in counts but its issues turn report-only —
                        # visible, never routed into a destructive merge
                        disputed.add(unit_path)
                    recounts.append({'unit': unit_path, 'declared': declared,
                                     'actual': actual, 'revised': revised,
                                     'answer': raw})
            data, issues, rounds = await correct(
                self.runner, execution, payload=text,
                scheduler=self.scheduler, max_rounds=self.max_rounds,
                specialist_instructions=self.specialist_instructions,
                validator=lambda d: list(validator(d) or [])
                + count_issues(counts, d, soft=disputed))
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
                      recounts=recounts,
                      prompts=dict(self.prompts))
        return ExtractionResult(data, issues, trace)
