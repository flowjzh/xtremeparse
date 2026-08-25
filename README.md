# XtremeParse

> **"Feed it text and a schema. Get structure back — fast."**

XtremeParse is an extreme-concurrency structured-extraction engine for LLM
pipelines. Given plain text and a JSON Schema, it:

1. **Chunks** the text deterministically (markdown structure → sentence
   punctuation → character windows — single-line inputs included).
2. **Routes** chunks to extraction units via one light agent call that
   outputs only an index map (minimal output tokens, maximal speed) and
   itemizes repeated content for fan-out.
3. **Fans out** one specialist per unit — per item for long repeated
   sections, whole-array for short ones — under xtremeflow's scheduler.
4. **Self-corrects**: validates the merged result with your validator,
   routes issues back to the failing specialists, and re-runs only them
   with the errors appended to their conversation history.

The full-text prefix is shared across all calls to maximize provider-side
KV-cache hits; the router runs first and warms the cache for the fleet
behind it.

## Quick start

```python
from xtremeparse import Extractor

class MyRunner:  # adapt your agent framework (PydanticAI, ...)
    async def run(self, *, instructions, result_schema, content,
                  scope=None, tools=None, history=None, feedback=None): ...

extractor = Extractor(MyRunner())

result = await extractor.extract(text, json_schema, validator)
result.data    # best-effort schema-shaped dict
result.issues  # unresolved error-level issues (lenient, never raises)
result.trace   # chunks, router index map, specialist calls, correction rounds
```

## Neutral by design

The library has zero vendor dependencies beyond `xtremeflow`. It never
imports your schema tool, your agent framework, or your validator:

- **Structure contract** — JSON Schema dict in, schema-conforming dict out.
- **Validation contract** — you inject a `validator` callable returning
  error-level `Issue` objects (anything with `path`/`code`/`message`/
  `expected`/`got` conforms, zero adapter code).
- **Agent contract** — you adapt an `AgentRunner` to your framework
  (PydanticAI, or whatever comes next).

Extraction never raises on imperfect data: `ExtractionResult.data` is the
best effort and `.issues` tells the truth. Strictness is your policy.

Tuning knobs (constructor): `unit_strategy={'path': 'per-item' | 'whole'}`
prior knowledge for array units, `output_budgets={'path': n | [n, ...]}`
to override the router's own `@<n>` arrangements, `max_rounds` for the
correction budget, `max_chars` for chunk size, `max_concurrency` for the
scheduler. The three pipeline prompts are replaceable too —
`router_instructions` / `recount_instructions` / `specialist_instructions`
(see [docs/prompting.md](docs/prompting.md); the routing DSL they elicit
is a public contract, [docs/dsl.md](docs/dsl.md)).

## Trace contract

`result.trace` is eval-facing API — stable keys, populated by the
pipeline stages:

| key | shape |
|---|---|
| `chunks` | the chunk list the router saw |
| `router` | `{'counts': {path: n}, 'assignments': [{'unit', 'item', 'chunks'}], 'budgets': {path: n \| [n, ...]}}` |
| `groups` | one per specialist call: `{'unit', 'kind', 'item', 'strategy', 'chunk_ids', 'budget', 'batch'}` |
| `corrections` | one per re-run: `{'unit_path', 'item', 'issue_paths'}` |
| `prompts` | per prompt slot: `'default'` or `#<sha1-8>` of a host override |

Eval support: `xtremeparse.evalkit` (zero-dep) reads budgets and router
invariants back out of a trace; `xtremeparse.judging` runs one rubric
call through the same `AgentRunner` neutrality (bring your own judge —
a model other than the extraction fleet's and temperature 0 both read
better, measured); `pip install xtremeparse[evals]` adds pydantic-evals
adapters (`RouterOverlap`, `BudgetFit`, `RubricJudge`) over those
mechanisms for hosts building regression suites.

## Status

Pipeline complete (chunk → route → fan out → merge → correct), 160+
deterministic tests, zero LLM required for the suite.
