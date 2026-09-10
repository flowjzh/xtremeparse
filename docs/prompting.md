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
  numbered chunk listing, `card` the unit's semantic card — all built by
  code. `legend` is one `code = <unit card header>` line per unit; its
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
  context, as this block and inside the payload's schema JSON — and a
  positional discriminator ("the contiguous … resume") that keeps a
  merged bilingual document's parallel half out of the route. It is
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

Shape the judged material before trusting exclusion clauses — a judge
shown field structure keeps demanding a field-for-field mirror of the
source even against explicit exclusions, measured across models and
reasoning modes. For containment questions (is every fact anywhere?),
judge the host's flattened-digest projection and carry the schema's
field inventory in the rubric. The open-diff judge measures
~zero sensitivity to seeded deletions — presence checking belongs to
deterministic anchors beside it, not to the judge alone.

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
- The shared payload prefix (`content`) — byte-identical across every
  call it serves (router: full text + full schema; specialists: the
  text alone, each call's own partial schema riding beside its unit
  card); the provider KV cache depends on the byte identity.

## Stability caveat

The defaults carry measured calibration: map-before-count (a count-first
declaration made models drop a section's tail entry), card-header
legends (bare paths cost recognition), form-matched budget declarations
(ratio by default, keyword and absolute for their shapes). An override
discards all of it. A/B against the default on a frozen corpus
before shipping one.
