"""XtremeParse: extreme-concurrency structured extraction — chunk, route, fan out, self-correct."""

from importlib.metadata import PackageNotFoundError, version

from xtremeparse.contracts import (
    AgentResult,
    AgentRunner,
    ExtractionResult,
    Issue,
    JSONSchema,
    Trace,
    Validator,
)
from xtremeparse.extractor import Extractor
from xtremeparse.scheduling import TaskScheduler, report_tokens, scheduler_from_spec

try:
    __version__ = version('xtremeparse')
except PackageNotFoundError:
    __version__ = '0.1.0'

__all__ = [
    'Extractor',
    'AgentResult',
    'AgentRunner',
    'ExtractionResult',
    'Issue',
    'JSONSchema',
    'Trace',
    'Validator',
    'TaskScheduler',
    'report_tokens',
    'scheduler_from_spec',
]
