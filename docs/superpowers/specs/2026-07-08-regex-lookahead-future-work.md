# Regex-Terminal Lookahead Crossing — Future Work

**Date:** 2026-07-08
**Status:** Open problem — deliberately deferred from [`2026-07-08-token-boundary-lookahead-design.md`](2026-07-08-token-boundary-lookahead-design.md) (decision L2).

## The unsolved problem

The v1 lookahead engine builds merged tokens by concatenating **known strings** (exact-string terminals) and walking them through the vocabulary trie. A regex terminal (`json_char`, `digit`, `word`, …) is an *open class*: there is no single string to concatenate, so the DFS cannot cross it. v1 rule: merged tokens stop at regex-terminal boundaries; regex terminals participate only as whole-token matches (the pre-lookahead behavior).

Concrete cost, using the pydantic JSON grammar:

```
S* → <<{"n": >> digit JSON_DIGITS <<}>>
```

The model's natural tokenization of `{"n": 42}` likely contains a token like `": 4` — constant chunk suffix + first digit merged. That token crosses from the chunk into the `digit` class, so v1 cannot offer it: the model is still forced to split at the chunk/number boundary. Every skeleton↔value boundary in JSON output keeps this residual distortion. The same applies on the way *out* of an open class (`2}` — last digit + closing brace).

## Solution logic

Generalize the DFS so that walking *through* a regex terminal walks the vocabulary trie and the regex's **character-level DFA in lockstep** (a product walk):

1. **Compile** each regex terminal to a DFA over the tokenizer's surface alphabet (byte-level BPE forms — `Ġ`, `Ċ`, and the U+0100–U+011F control-byte range must be members or non-members of each class explicitly).
2. **Product walk**: at a regex terminal, iterate the current trie node's children; for each child character also step the DFA. Both structures prune: a character outside the class kills the DFA; a character no vocab token continues kills the trie. Yield `token_id` whenever the trie node completes a token.
3. **Residue generalizes**: ending a token mid-match means the PDA is *inside* the regex terminal. `residue: str` becomes `residue: str | (terminal_name, dfa_state)` — a string suffix for exact terminals (v1 semantics preserved), a DFA state for open classes. `eos()`, `clone()`, cache digests extend accordingly.
4. **Exiting the class — the genuinely hard part.** A regex match has no explicit end marker: while the DFA sits in an accepting state, the *next* character may either extend the match or begin the next grammar symbol. The product walk must branch on both interpretations (fork: stay-in-class vs advance-grammar). This introduces end-of-match ambiguity:
   - If FIRST(next symbol) is disjoint from the class alphabet (e.g. `"` after `json_char`, `}` after `digit`), the fork is deterministic — the common case in this codebase, and why the JSON grammar remains LL(1) in spirit.
   - If they overlap (e.g. `word word` — two adjacent open classes over the same alphabet), both branches stay live and the same token string becomes derivable through multiple split points. Policy choice required: **longest-match** (greedy, standard lexer semantics; recommended) versus **all-splits** (complete but multiplies states and re-introduces token-level ambiguity that `get_tokens()`'s disjointness check would reject). The chosen policy must also be applied consistently in `next_state` path replay.
5. **Memoization key** extends to `(trie_node, dfa_state, stack_digest)` — the product walk revisits identical `(trie, dfa)` pairs constantly across states; this cache is what keeps the walk tractable.

## Technology required

| Need | Options |
|---|---|
| Regex → DFA compiler over the surface alphabet | `interegular` (what Outlines uses for FSM construction — proven, handles intersection), `greenery` (regex algebra, DFA minimization), or hand-rolled Thompson construction + subset construction — viable here because GrammarLLM's regex terminals are simple character classes with `+`/`*`, not full PCRE |
| Alphabet handling | DFAs must be built over BPE surface characters, not logical characters: compile the user's regex, then remap its character classes through the byte-level BPE encoding table (the U+0100–U+011F shift discovered during the strict-JSON work) |
| Product-walk engine | Extension of `lookahead.py`'s DFS: state = `(pda_state, trie_node, dfa_state \| None)`; ~memoized as above |
| Residue variant type | `str \| (terminal, dfa_state)` in `PushdownAutomaton`; serialization in cache digests |

## Precedent

This construction converges to Outlines' FSM index (regex → token-level FSM via trie∩DFA), computed **lazily per PDA state** instead of offline per grammar, and to XGrammar's byte-level pushdown with its adaptive token-mask cache. The lazy formulation fits GrammarLLM because masks here are stack-dependent (CFG, not a single regex), so a full offline index is not directly applicable — the per-state memoized product walk is the natural middle ground.

## Acceptance criteria (when implemented)

1. Canonical-tokenization acceptance extends to grammars with open classes: `tokenizer.encode('{"n": 42}')` replays through the masks of the pydantic JSON grammar without splitting at chunk↔digit boundaries.
2. Longest-match policy documented and enforced identically in mask construction and `next_state` replay.
3. Old ⊆ new mask invariant holds relative to v1 lookahead.
4. Perf budget unchanged (≤ ~5ms cold per state on the 150k vocab) thanks to the `(trie_node, dfa_state)` memo.
