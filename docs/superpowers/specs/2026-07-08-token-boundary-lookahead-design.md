# Token-Boundary Lookahead (g_t_r) — Design

**Date:** 2026-07-08
**Status:** Approved (brainstorming session) — **superseded in part, see the amendment below**

> ## Amendment: the collision policy was wrong
>
> This spec settled token/path collisions with *"first path found wins (setdefault), scan
> order"*. That decision is the source of a silent correctness bug and has been reverted.
>
> Admitting merged and mid-terminal tokens makes the string→token map one-to-many: a single
> token can be compatible with **several** places in the grammar. Keeping one path and
> discarding the others does not merely pick a ranking — it takes a decision that belongs to
> the model, and then forces every later token to live with it. A model emitting the token
> that canonically opens one terminal could be routed into a *different* terminal purely
> because that alternative appeared earlier in scan order, with no way to recover. The output
> stays grammatical, so nothing looks wrong.
>
> `lookahead_paths()` now returns **every** compatible path, and the processor carries a
> **set** of live PDA states (`PdaSet`): the mask is their union, and the set collapses only
> as the model writes. The grammar never guesses.
>
> The general rule: **when a constrained decoder admits more than one tokenization, ambiguity
> is unavoidable and must be carried, not resolved.** Any policy that collapses it early —
> first match, scan order, canonical-only — silently overrides the model.
>
> Note also that this spec's headline metric (preserved probability mass) is a poor proxy for
> what the feature is worth: it rewards admitting *more* tokens regardless of whether the
> grammar then interprets them correctly, and it stayed healthy while accuracy was being lost.
> Judge the engine on downstream accuracy, not on mask width.
>
> See [`token-boundary-lookahead.md` §6](../../token-boundary-lookahead.md#6-advancing-after-the-model-picks-a-token).
**Scope:** Remove GrammarLLM's token-boundary limitation: let the model emit its natural merged tokens (e.g. `" b"`, `"{ ci"`) even when they span grammar-terminal boundaries or end mid-terminal. Companion doc: [`2026-07-08-regex-lookahead-future-work.md`](2026-07-08-regex-lookahead-future-work.md).

## Problem

Today `map_terminal_tokens` requires every vocabulary token to lie *entirely inside one grammar terminal*. Any merged token spanning a boundary (`": `, `", `, `"{ ci`) can never be generated, so the model is forced into non-canonical tokenizations: output text is correct, but the model generates off its natural distribution — measurably worse quality on long structured outputs, and the root cause of quality gaps versus byte-level engines (Outlines, XGrammar).

Two concrete scenarios this design targets (user's formulation):

1. **Multi-terminal exact spans** — token `"{ "` fully consuming terminals `LPAR` and `WS`.
2. **Mid-terminal cuts** — token `"{ ci"` fully consuming `LPAR`+`WS` and partially consuming `STR="ciao"`, leaving `"ao"` stored as state residue.

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| L1 | Fragment granularity: **residue states** — PDA state becomes `(stack, residue)` | Full granularity hiding (tokens may end anywhere inside a terminal) at moderate complexity; char-level PDA reformulation rejected as a rewrite, terminal-boundary-only rejected as half a fix |
| L2 | Regex terminals: **punt in v1** — they participate only as whole-token matches; merged tokens stop at regex-terminal boundaries | Exact-string terminals (tags, JSON skeleton chunks, labels) are where the boundary pain is; crossing regex classes needs a trie∩DFA product walk — specified in the future-work doc, not built now |
| L3 | Integration: **enabled by default** (`token_lookahead=True`), flag is an **opt-out** | Canonical tokenization and natural probability distribution are the standard out-of-the-box experience; `token_lookahead=False` exists for the A/B baseline and preserved-mass measurements |
| L4 | Tests: **one suite, parametrized over both engines** | `pytest.mark.parametrize` runs the relevant unit tests with `lookahead=True` and `False`; the legacy boundary-strict engine stays fully tested for A/B; no second test folder |
| L5 | Memoization key: **full-state digest** `(tuple(stack), residue)` in v1 | Always correct; the bounded nullable-prefix digest is a future optimization, not assumed for correctness |

## State model

- `PushdownAutomaton` gains `residue: str = ""` — the unconsumed suffix of a partially-covered terminal.
- `eos()` becomes: stack empty **and** `residue == ""`.
- `clone()` copies the residue. `reset()` clears it.
- `StatelessLogitsProcessor` cache keys stay `(prompt_idx, token_history)` — the token history uniquely determines the residue, and cached PDA objects carry it.

**Alphabet note:** everything operates in the tokenizer's *surface* alphabet (byte-level BPE forms — `Ġ` for space, `Ċ` for newline, etc.). Grammar terminals are already subword strings produced by `tokenizer.tokenize()`, and the vocab trie is built over raw vocabulary strings; both sides therefore share one alphabet, no conversion layer.

**Vocab trie:** built once per tokenizer (~150k entries, one-time cost ~1s), cached on the processor / module level. Nodes carry `token_id` when the path spells a complete vocabulary token.

## Mask algorithm — g_t_r with explicit accumulator

The original whiteboard pseudocode lost the accumulated prefix across recursion; the corrected form threads it (as the current trie node) and consumes each fragment before recursing:

```
lookahead_tokens(pda, trie) -> { token_id: path }
    # path = (fragments_fully_consumed: list[terminal], chars_into_last: int)

dfs(pda_state, trie_node, path):
    fragments = [residue] if pda_state.residue else valid_terminals(pda_state)
                                       # valid_terminals = FIRST-of-stack scan
    for frag in fragments:
        if frag is a regex terminal:
            if path is empty:
                yield its whole-token vocab matches      # today's behavior
            continue                   # L2: no crossing regex boundaries in v1
        node = trie_node
        for i, ch in enumerate(frag):
            node = node.child(ch)
            if node is None:
                break                  # vocabulary has no token continuing this way
            if node.token_id is not None:
                yield node.token_id, path + (frag, i + 1)
                                       # mid-fragment cut → residue = frag[i+1:]
        if node is not None:           # fragment fully inside the trie
            dfs(advance(pda_state, frag), node, path + [frag])
```

Properties:

- **Old ⊆ new**: depth-0 exact matches reproduce today's mask exactly; lookahead only widens it.
- **Trie pruning**: every branch the vocabulary cannot realize dies at the first missing child — cost is bounded by trie nodes visited, not by grammar breadth.
- **`advance`** uses `pda.clone()` per DFS node (O(stack)); no undo-log. The existing `next_state_terminal` machinery is reused unchanged.
- A token fully consuming its last fragment gets `chars_into_last == len(frag)` → residue `""`.

**Advancing after generation:** the mask cache stores `path` per yielded token. `next_state(token_id)` looks the path up, consumes the whole fragments via `next_state_terminal`, then sets `residue = last_frag[chars_into_last:]`. Tokens absent from the current mask still raise `ValueError` (no silent bypass — same invariant as today). The forced-EOS replay helper `_advance_token` is unchanged.

**Memoization:** `{ (tuple(stack), residue): (valid_token_ids, paths) }` with LRU bounding, alongside the existing PDA cache. Identical states recur heavily across beams and steps.

## Integration

- New module `grammarllm/modules/lookahead.py`: `VocabTrie`, `lookahead_tokens(pda, trie)`, digest-keyed mask/path cache.
- `generate_grammar_parameters(..., token_lookahead=True)` — **default True** (L3). The flag reaches the PDA (`pda.lookahead` attribute); `StatelessLogitsProcessor` routes mask computation through `lookahead_tokens()` when set, `get_tokens()` otherwise.
- Everything else is untouched: LRU history cache, dead-end fallback, forced-EOS replay, streamer, result format. Only the two primitives — "what is valid here" and "consume this token" — are swapped.
- `token_lookahead=False` is the documented A/B baseline: same grammar/prompt through both engines, compared with `compare_analyses(..., metric="preserved_mass")`.

## Error handling

- No new user-facing failure modes: grammar construction is unchanged; the trie build cannot fail; DFS falls back to today's behavior at regex boundaries.
- The dead-end fallback (empty mask, stack non-empty → force EOS) applies identically; with lookahead the mask is a superset, so dead ends can only become rarer.
- Invalid tokens in re-simulation raise `ValueError` exactly as today.

## Testing (L4 — one parametrized suite in `grammarllm/tests/`)

1. **Parametrized regression**: existing PDA/processor unit tests run under `@pytest.mark.parametrize("lookahead", [False, True])` where the assertion is engine-independent (advance, reset, replay, cache semantics). Legacy engine stays fully covered.
2. **Old ⊆ new**: for every test grammar and state, the legacy mask is a subset of the lookahead mask.
3. **Merged-token unit tests** (synthetic trie + grammar): `"{ "` span across two terminals appears in the mask; `"{ ci"` yields residue `"ao"`; `eos()` false while residue non-empty; advance-by-path lands on the correct `(stack, residue)`.
4. **Canonical-tokenization acceptance (headline)**: for valid sentences of a real grammar, `tokenizer.encode(sentence)` — the model's natural tokenization — must be accepted token-by-token by the masks. Today's engine fails this by construction; the lookahead engine must pass.
5. **Round-trip walks**: random walks over lookahead masks (including merged-token choices) decode to grammar-valid text; for the pydantic JSON grammar, `json.loads` + `model_validate` hold.
6. **A/B preserved-mass**: same grammar/prompt through both engines; assert lookahead preserved-mass ≥ legacy (and report the delta — the paper-worthy measurement).
7. **Perf budget**: memoized mask lookup is O(1) on hit; cold mask computation for a representative JSON grammar state stays under ~5ms on the 150k-token Qwen vocabulary.

## Known limitations (documented, not fixed here)

- **Regex-terminal crossing** — v1 stops merged tokens at regex boundaries; full solution (trie∩DFA product walk) specified in the future-work doc.
- **Probability-mass split across tokenizations**: once multiple token paths spell the same string, beam hypotheses differing only in tokenization compete; standard for every character-level engine. Documented; measurable with the analysis tooling.
- **DFS worst case**: grammars with hundreds of exact-string alternatives at one state; mitigated by trie pruning and memoization, guarded by the perf-budget test.

## Affected files

- `grammarllm/modules/lookahead.py` — new: trie, DFS, mask cache.
- `grammarllm/modules/automaton.py` — `residue` field, `eos()`, `clone()`, `reset()`, path-replay in `next_state`.
- `grammarllm/modules/logits_processor.py` — mask routing by `pda.lookahead`.
- `grammarllm/generate_with_constraints.py` — `token_lookahead=True` parameter.
- `grammarllm/tests/` — parametrization + new lookahead tests (local-only, per repo policy).
- `docs/usage.md`, `README.md` — document the default and the opt-out baseline.
