"""Boundary probes for the scheduling surface: spec → scheduler, usage report."""

import pytest
from xtremeflow.scheduler import TaskScheduler
from xtremeflow.scheduler.token import TokenRateScheduler

from xtremeparse.scheduling import report_tokens, scheduler_from_spec


def test_bare_int_gets_a_plain_cap_without_rates():
    scheduler = scheduler_from_spec(8)
    assert isinstance(scheduler, TokenRateScheduler)
    assert scheduler._max_rps is None and scheduler._max_tps is None
    assert scheduler.semaphore._value == 8


def test_full_spec_string_parses():
    scheduler = scheduler_from_spec('8|rps:10|tps:1000|burst:0')
    assert scheduler._max_rps == 10 and scheduler._max_tps == 1000
    assert scheduler._burst_ratio == 0


def test_minute_units_convert_to_per_second():
    scheduler = scheduler_from_spec('2|rpm:600|tpm:6000')
    assert scheduler._max_rps == 10.0 and scheduler._max_tps == 100.0
    assert scheduler._burst_ratio == 1.0


@pytest.mark.parametrize('spec', [
    '8|xps:10', 'rps:10', '0', '8|rpm:600|rps:10', '8|tpm:6|tps:10', 3.5])
def test_invalid_specs_raise(spec):
    with pytest.raises(ValueError):
        scheduler_from_spec(spec)


class SpyScheduler(TokenRateScheduler):
    def __init__(self):
        super().__init__(max_concurrency=2)
        self.corrections = []

    async def _apply_correction(self, actual: int):
        self.corrections.append(actual)


async def test_reported_tokens_correct_the_dispatching_scheduler():
    spy = SpyScheduler()
    task = await spy.start_task(report_tokens(18))
    await task
    assert spy.corrections == [18]


async def test_report_is_inert_outside_token_rate_scheduling():
    await report_tokens(5)  # no scheduler context at all
    task = await TaskScheduler(1).start_task(report_tokens(5))  # plain cap
    await task
