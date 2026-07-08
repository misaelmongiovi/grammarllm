# Pydantic → Strict-JSON Grammar Conversion — Design

**Date:** 2026-07-07
**Status:** Approved (brainstorming session)
**Scope:** Rework of `grammarllm/utils/pydantic_to_grammar.py` so that generated text is always valid JSON, round-trippable into the source Pydantic model.

## Problem

The current converter emits bare key/value terminals. A `Person` model produces output like `namemarioaddresscitymilanmoodhappy` — grammar-valid but not parseable by anything standard. The `_emit_object` docstring promises `:` separators the code never emits. Users who want custom formats can already write productions by hand; the pydantic entry point should therefore be opinionated: **strict JSON**.

## Goals

1. Every constrained generation passes `json.loads()` and `Model.model_validate_json()`.
2. One-call setup: the converter supplies its own regex terminals; no hand-built `regex_dict` footgun.
3. Support recursive models (trees, linked lists) when the cycle is breakable.

## Non-goals

- Configurable output styles (compact/key-value). Manual grammars cover that.
- JSON string escape sequences (`\"`, `\n`, `\uXXXX`) — v1 excludes `"`, `\`, and control chars from string content. Documented limitation.
- Omittable optional keys. Keys are always emitted (see Decisions).
- Integer enums, `minItems`/`maxItems` enforcement, `additionalProperties` schemas.

## Decisions (with rationale)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Single opinionated target: strict JSON | Custom formats go via manual grammars; pydantic users expect structured output à la JSON mode |
| D2 | API returns `(productions, regex_dict)` — breaking change from dict-only return | Terminal names and their regexes must never drift apart; module is new/untracked so the break is free |
| D3 | Optional fields: key always present, value alternates `X \| null` | Omittable keys make commas conditional → comma left-factoring, fragile with consecutive optionals. Fixed skeleton is trivially LL(1); pydantic accepts `null` for `Optional` |
| D4 | Emission approach: skeleton-chunk tags | All punctuation between values is constant (thanks to D3) → one `<<tag>>` per gap. Fewest NTs and masking steps; tokenizer decides subword splits; existing prefix-grouping absorbs multi-token tags |
| D5 | Quotes always belong to the string value NT, never the skeleton | Uniform rule that survives `Optional[str]` alternation (`"..."` vs `null`) without special cases |
| D6 | $ref cycles allowed iff ≥1 edge in the cycle is Optional or array-item | Termination guaranteed by `null`/`]` exits; grammar stays right-recursive (recursive NT always preceded by a constant chunk). Unbreakable cycles still rejected |
| D7 | `int \| float` unions rejected at validation | Both branches start with a digit — guaranteed FIRST conflict; fail fast with an actionable message |

## API

```python
prods, regex_dict = pydantic_to_productions(Model)   # tuple return (D2)
pt, mtt = get_parsing_table_and_map_tt(tokenizer, prods, regex_dict)
```

- `type_terminal_map` override parameter is kept.
- All conversion failures raise `PydanticGrammarError` (subclass of `ValueError`), messages phrased in pydantic vocabulary: which field, what to change.
- The rest of the pipeline (`ProductionRuleProcessor`, `parsing_table`, `generate_token_maps`, PDA, logits processor) is untouched.

## Emission rules

Object skeleton — constant chunks between value NTs, one tag per gap. Canonical spacing: `": "` after keys, `", "` between pairs, no padding inside `{}` / `[]`:

```
S*       → <<{"name": >> NAME_V <<, "mood": >> MOOD_V <<}>>
```

| Construct | Production shape |
|---|---|
| enum (str) | `V → <<"happy">> \| <<"sad">>` (quotes inside tag) |
| Optional[X] | `V → X_V \| <<null>>` |
| str | `V → <<">> CHARS <<">>` ; `CHARS → json_char CHARS \| ε` |
| int | `V → SIGN_OPT digit DIGITS` ; `SIGN_OPT → <<->> \| ε` ; `DIGITS → digit DIGITS \| ε` |
| float | `V → INT_PART FRAC` ; `FRAC → <<.>> digit DIGITS \| ε` |
| bool | `V → <<true>> \| <<false>>` |
| array | `V → <<[>> BODY` ; `BODY → ITEM TAIL \| <<]>>` ; `TAIL → <<, >> ITEM TAIL \| <<]>>` (empty arrays allowed) |
| nested object | value slot references the nested skeleton NT |
| $ref | named shared NT per `$defs` entry (e.g. `NODE`); `_emitted` set prevents duplicate emission |

LL(1) safety notes:
- Array `TAIL`: FIRST = `{',', ']'}` — disjoint by construction.
- Optional value: FIRST(X_V) starts with `"`, digit, `{`, `[`, `true/false` — all disjoint from `null`.
- anyOf across distinct JSON types is FIRST-disjoint via the leading character class; same-type branches remain rejected (existing rule), now including `int | float` (D7).

## Default regex terminals (returned in `regex_dict`)

Token-level regexes — each matches whole vocabulary tokens:

| terminal | regex | note |
|---|---|---|
| `regex_json_char` | `^[^"\\\x00-\x1f]+$` | string content; digit/`}`/`,` tokens are legal inside strings — the closing `"` is the only exit |
| `regex_digit` | `^[0-9]+$` | multi-digit vocab tokens reduce step count |

`-`, `.`, `true`, `false`, `null`, and all skeleton chunks are exact `<<tags>>`.

Overlap safety: `json_char` token set ⊇ `digit` token set, but the two are never simultaneously valid — string context is always entered through `"`, so no reachable PDA state has both as current terminals.

## Recursion (D6 detail)

Validator change: instead of rejecting any `$ref` on the expansion stack, track whether the path from the ref'd definition back to itself crosses an Optional or array-item edge.

- Crossed → allowed. Translator emits the named NT once; recursive value slots reference it.
- Not crossed → `PydanticGrammarError` naming the field to make `Optional[...]`.

```python
class Node(BaseModel):
    value: int
    next: Optional["Node"] = None
# → {"value": 1, "next": {"value": 2, "next": null}}
```

## Error handling

- All Phase-1 rejections unchanged (`if/then/else`, `patternProperties`, open `additionalProperties`, same-type unions, non-string enums, `array` without `items`) plus D6/D7 rules.
- No new runtime failure modes: everything detectable is detected at conversion time.

## Testing

1. **Unit** (no tokenizer): per-construct assertions on productions + regex_dict; reachability check — every emitted NT reachable from `S*`.
2. **Integration** (real tokenizer, `Qwen/Qwen2.5-0.5B-Instruct`): representative models (flat literals, Optional, nested, array, recursive Node, primitives) → `get_parsing_table_and_map_tt` must not raise. This layer is mandatory: the previous double-ε bug passed all unit tests and only crashed in the table builder.
3. **Round-trip** (no model, fast): random-walk the PDA over its valid-token sets to synthesize complete outputs; every walk must pass `json.loads` + `Model.model_validate_json`. Depth-capped walks for the recursive model.
4. **Smoke E2E**: one real `generate_text` run with the small model asserting round-trip on actual generation.

## Affected files

- `grammarllm/utils/pydantic_to_grammar.py` — translator rewrite (validator largely intact, +D6/D7).
- `grammarllm/tests/test_pydantic_to_grammar.py` — update expectations, add layers 2–3.
- No changes to core pipeline modules.
