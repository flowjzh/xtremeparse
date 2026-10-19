# Prompt overrides

xtremeparse ships measured default prompts. They are replaceable — real
effectiveness ultimately depends on the host's schema and eval — but an
override is an escape hatch for domain/model/language adaptation, not
the primary tuning path.

## Layering order (measured)

1. **Schema `description`s first.** Field-level semantics live in the
   schema the host passes in: unit cards and legends are composed from
   them, and field fidelity follows the description's wording (a
   fidelity clause in a field description moves output where generic
   prompt rules never did). The schema's root `description` is the
   document-level counterpart — it rides the router prompt as an
   "Overall Instruction" block.
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
  schema block, the `Chunks:` heading, and the numbered listing under
  it (the schema rides ahead of the heading — it cannot be positioned
  or omitted independently), `card` the unit's semantic
  card — all built by
  code. `legend` is one `code = <unit card>` block per unit — the card
  header plus its field descriptions, a second carrier of the field
  context the schema block above already renders per unit; its
  format is fixed by code (the card header is the cue the default
  prompts rely on — an override legend interpretation is the override's
  to keep consistent).
- The default router and recount templates carry one optional slot,
  `{overall}`: the schema's root `description`, rendered as an
  "Overall Instruction" block at the prompt's very end, so a
  document-selection constraint out-shouts the chunk listing it
  governs (above the rules it lost rounds to that listing's pull,
  measured). Wording a root description must satisfy (measured, same
  A/B): no template self-reference — the sentence is read out of
  context, as this block and inside the JSON Schema block the chunks
  section renders — a
  positional discriminator ("the contiguous … resume") that keeps a
  merged bilingual document's parallel half out of the route, and a
  dismissal that names a parallel version — vaguer "discontiguous
  content" also fired on foreign fragments embedded in a mono-lingual
  resume and dropped their facts (measured). It is
  not in the
  required placeholder sets — an override without it never shows the
  block. The extractor fills it from the schema automatically; direct
  `route()` callers pass `overall=`.
- Correction rounds re-dispatch with the same override automatically —
  a correction round continues the original call's conversation history.
- The judge prompt (`xtremeparse.judging.judge(instructions=...)`) is
  overridable the same way — one placeholder, `rubric`.

## Judging (measured guidance)

Judge-side settings live at the adapter, as recommendations:

- A model other than the extraction fleet's tends to read kinder to
  self-judging bias.
- Temperature 0 steadies verdicts (a judge that flips run to run
  measures noise, not quality).
- Thinking on for containment judging. With the verbatim retention
  clause, a non-thinking decode produced false losses on facts the
  digest carried and missed real omissions (measured on the suite);
  thinking on with the evidence protocol below caught the real losses
  and respected schema exclusions. Scope it to the containment judge —
  the structured-output rubrics keep the cheaper default decode. The
  judge runs outside the extraction scheduler, so its extra latency
  stays out of the fleet's budget.

Shape the judged material before trusting exclusion clauses — a judge
shown field structure keeps demanding a field-for-field mirror of the
source even against explicit exclusions, measured across models and
reasoning modes. For containment questions (is every fact anywhere?),
judge the host's flattened-digest projection and carry the schema's
field rules in the rubric — each field's description (exclusions
included) and option set, not just its name; names alone cannot tell
the judge what a field owns. A containment protocol supplied by the
host makes the check mechanical: quote each candidate loss from the
source, normalize whitespace and markdown markers on both sides, and
cross the candidate out when its normalized text occurs anywhere in
the output; only surviving quotes may appear in the verdict. Pass it
through the judge's `instructions=` slot for the digest rubric only —
the default template stays projection-agnostic.
The open-diff judge
measures ~zero sensitivity to seeded deletions — presence checking
belongs to deterministic anchors beside it, not to the judge alone.

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
  specialists' text-only one. The router plane reads the full schema
  in its chunks section (the block ahead of the `Chunks:` heading) and
  the same field descriptions again in the legend's unit cards — both
  measured: dropping the block from the prompts cost ~30% more router
  rounds at unchanged count quality (K=5, full corpus), so it stays.

## Stability caveat

The defaults carry measured calibration: map-before-count (a count-first
declaration made models drop a section's tail entry), card-header
legends (bare paths cost recognition), form-matched budget declarations
(ratio by default, keyword and absolute for their shapes). An override
discards all of it. A/B against the default on a frozen corpus
before shipping one.
