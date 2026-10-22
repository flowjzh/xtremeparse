# Prompt overrides

xtremeparse ships default prompts. They are replaceable — real
effectiveness ultimately depends on the host's schema and eval — but an
override is an escape hatch for domain/model/language adaptation, not
the primary tuning path.

## Layering order

1. **Schema `description`s first.** Field-level semantics live in the
   schema the host passes in: unit cards and legends are composed from
   them, and field fidelity follows the description's wording. The
   schema's root `description` is the document-level counterpart — the
   router prompt's final Overall Instruction.
2. **Prompt overrides second.** For adapting the generic extraction
   discipline to a new document domain, a weaker model, or another
   language.
3. **Never** the DSL ([dsl.md](dsl.md)) or the validation / recount /
   correction loops — that machinery is the library's core contract.

## The three slots

```python
Extractor(runner,
          router_instructions=...,      # the segment-map prompt
          recount_instructions=...,     # the zero-unit recount prompt
          specialist_instructions=...)  # the extraction prompt
```

| kwarg | placeholders |
|---|---|
| `router_instructions` | `top`, `none`, `legend`, `chunks` |
| `recount_instructions` | `legend`, `chunks` |
| `specialist_instructions` | `card` |

- Templates are `str.format` strings. Required placeholders are
  validated at construction: a template missing one raises `ValueError`
  instead of silently rendering a prompt without its chunk listing.
- `top` is the highest chunk id, `none` the NONE marker, `chunks` the
  inline material an override gets — the schema block, the `Chunks:`
  heading and the numbered listing under it; the default router
  assembles the listing and schema as separate stable messages, and the
  default recounts send the listing alone; `card` the unit's semantic
  card — all built by code. `legend` is one `code = <unit card>` block
  per unit — the card header plus its field descriptions, a second
  carrier of the field context the schema message already renders per
  unit; its format is fixed by code (the card header is the cue the
  default prompts rely on — an override legend interpretation is the
  override's to keep consistent). The default router template therefore
  carries only `top`, `none`, `legend`; an override that wants the
  material inline keeps `{chunks}`.
- The recount templates carry one optional slot, `{overall}`: the
  schema's root `description`, rendered as an "Overall Instruction"
  block at the prompt's very end, so a document-selection constraint
  out-shouts the chunk listing it governs. The default router prompt
  renders the same instruction as its final message (the final,
  uncached message); a router override may carry `{overall}` to get it
  in the template. Wording a root description must satisfy: no template
  self-reference — the sentence is read out of context, as this block
  and inside the router's JSON Schema message — a positional
  discriminator ("the contiguous … resume") that keeps a merged
  bilingual document's parallel half out of the route, and a dismissal
  that names a parallel version (a vague "discontiguous content"
  dismissal also matches foreign fragments embedded in a monolingual
  resume). It is not in the required placeholder sets — an override
  without it never shows the block. The extractor fills it from the
  schema automatically; direct `route()` callers pass `overall=`.
- Correction rounds re-dispatch with the same override automatically —
  a correction round continues the original call's conversation history.
- The judge prompt (`xtremeparse.judging.judge(instructions=...)`) is
  overridable the same way — one placeholder, `rubric`.

## Judging guidance

Judge-side settings live at the adapter, as recommendations:

- A model other than the extraction fleet's reads kinder to
  self-judging bias.
- Temperature 0 steadies verdicts.
- Thinking on for containment judging. Scope it to the containment
  judge — the structured-output rubrics keep the cheaper default
  decode. The judge runs outside the extraction scheduler, so its extra
  latency stays out of the fleet's budget.
- Judge the host's flattened-digest projection and carry the schema's
  field rules in the rubric — each field's description (exclusions
  included) and option set, not just its name; names alone cannot tell
  the judge what a field owns. A containment protocol supplied by the
  host makes the check mechanical: quote each candidate loss from the
  source, normalize whitespace and markdown markers on both sides, and
  cross the candidate out when its normalized text occurs anywhere in
  the output; only surviving quotes may appear in the verdict. Pass it
  through the judge's `instructions=` slot for the digest rubric only —
  the default template stays projection-agnostic. Presence checking
  belongs to deterministic anchors, not to the judge alone.

## Provenance

`result.trace.prompts` marks every slot `'default'` or `#<sha1-8>` of
the host's template, so eval regressions stay attributable to whose
prompt produced them.

## Not overridable

- The budget suffix handling — the router's arrangement forms are
  resolved and consumed by code, not by prompt wording.
- Repair and recount feedback messages — generated from validation
  errors, not prose.
- The diff protocol — a correction round with a previous result asks
  for a JSON Patch (RFC 6902) against it, and every router repair asks
  for a unified diff against the previous answer (see
  [repairs.md](repairs.md)).
  Hosts shape what is retried via the validator's issues, never the
  reply contract.
- The shared payload prefix (`content`) — the document text alone,
  byte-identical across every call it serves (router, recounts and
  specialists alike): the provider's cache serves one common block,
  and dashscope's explicit cache truncates only at message
  boundaries, so a text+schema system message could never serve the
  specialists' text-only one. The router plane reads the schema as its
  own stable message (verbatim markers render as `@verbatim` cues, the
  enforcement keyword is omitted) and the same field descriptions again
  in the legend's unit cards.

## Schema rendering

Every model-facing schema copy drops the `x-verbatim` keyword and folds
it into the field description as an inline `@verbatim` cue. The keyword
is not model-facing vocabulary. The cue is rendered in both planes —
the router's budget clause prices a verbatim copy, and the specialist's
output schema carries the copy rule; the host schema keeps the keyword
for enforcement.

The default recount plane reuses the chunk listing as its own stable
message and never repeats the schema; a host recount override owns its
material and still gets schema+chunks inline. The recount requires the
chunk listing: without its boundaries a zero-declared unit loses its
anchors and a shared/mixed recount drops merged sub-entries.

## Cache layout

The provider's explicit cache is prefix-based: a boundary closes only
at a message edge, the request must carry the identical prefix, and a
shorter request never matches a longer block. DashScope keeps up to
four boundaries per request (5-minute TTL, 1,024-token minimum). The
adapter marks exactly the leading stable messages a call declares —
`1 + len(prompt_parts)` from the runner, capped at four. A fresh
specialist call marks only the system payload; a history round marks
its first user message.

The router plane sends the document (system), the chunk listing, the
schema and the rules+legend as separate stable user messages; the
overall instruction is the final, unmarked message. Recounts reuse the
chunks block (zero and shared/mixed); the schema is not repeated.

## Stability caveat

The defaults encode calibration: map-before-count, card-header legends,
form-matched budget declarations, and the ranged dense-list form. An
override discards it. A/B against the default on a frozen corpus before
shipping one.
