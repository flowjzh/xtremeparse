"""Scheduling surface — this package's only xtremeflow boundary.

Hosts build the process-global scheduler from a spec string; runner
adapters report actual token usage against the dispatching scheduler.
Both live here so xtremeflow stays a dependency of xtremeparse alone.
Quotas are per model: one scheduler per model in use — a unified
slice/specialist model means one scheduler serves the whole fleet."""

from __future__ import annotations

from xtremeflow.scheduler import TaskScheduler
from xtremeflow.scheduler.rate_limit import get_context
from xtremeflow.scheduler.token import TokenRateScheduler, report_token_usage

VALID_KEYS = frozenset({'rps', 'tps', 'rpm', 'tpm', 'burst'})


def scheduler_from_spec(spec: int | str) -> TaskScheduler:
    """A rate-limited scheduler from a compact spec: a concurrency
    ceiling plus optional rps/tps (or rpm/tpm) budgets and a burst
    allowance, e.g. ``"8|rpm:30000|tpm:5000000|burst:0"``."""
    result: dict = {}
    for part in str(spec).split('|'):
        if ':' in part:
            key, value = part.split(':', 1)
            if key not in VALID_KEYS:
                raise ValueError(
                    f"Invalid key '{key}', must be one of: {', '.join(sorted(VALID_KEYS))}"
                )
            result[key] = float(value) if key == 'burst' else int(value)
        else:
            result['concurrency'] = int(part)
    if 'concurrency' not in result:
        raise ValueError(
            "concurrency string must specify concurrency (e.g., '10|rps:100')"
        )
    if result['concurrency'] <= 0:
        raise ValueError('concurrency must be positive')
    if 'rpm' in result and 'rps' in result:
        raise ValueError("Cannot specify both 'rpm' and 'rps'")
    if 'tpm' in result and 'tps' in result:
        raise ValueError("Cannot specify both 'tpm' and 'tps'")
    if 'rpm' in result:
        result['rps'] = result.pop('rpm') / 60.0
    if 'tpm' in result:
        result['tps'] = result.pop('tpm') / 60.0
    return TokenRateScheduler(max_concurrency=result['concurrency'],
                              max_rps=result.get('rps'), max_tps=result.get('tps'),
                              burst_ratio=result.get('burst', 1.0))


async def report_tokens(actual: int) -> None:
    """Correct the dispatching token-rate scheduler's reservation with
    actual usage; a no-op outside token-rate scheduling (bare
    TaskScheduler or none)."""
    if (ctx := get_context()) and isinstance(ctx.scheduler, TokenRateScheduler):
        await report_token_usage(actual)
