# Token-Boundary Lookahead (g_t_r) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the model emit its natural merged tokens (`"{ "`, `"{ ci"`) across grammar-terminal boundaries and mid-terminal, via a vocab-trie-guided DFS over PDA states with residue tracking.

**Architecture:** PDA state extends to `(stack, residue)`. A new `lookahead.py` module holds a per-tokenizer `VocabTrie` and `lookahead_tokens(pda, trie)` — a DFS that concatenates valid terminal fragments, prunes against the trie, yields `(token_id → path)` including mid-fragment cuts. The processor routes mask/advance through it when `pda.lookahead` is set (default ON); paths replay deterministically in `next_state`-equivalent `apply_lookahead_path`. Regex terminals participate only as whole-token matches at depth 0 (spec L2; crossing is future work).

**Tech Stack:** Python ≥3.10, existing GrammarLLM pipeline, real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer in tests (no mocks — repo test policy), pytest via `uv run --with pytest`.

**Spec:** `docs/superpowers/specs/2026-07-08-token-boundary-lookahead-design.md` (decisions L1–L5) + companion `docs/superpowers/specs/2026-07-08-regex-lookahead-future-work.md`.

## Global Constraints

- `token_lookahead=True` is the **default** in `generate_grammar_parameters` (L3); `False` is the documented A/B baseline.
- Fragment granularity: residue states — `(stack, residue)`; `eos()` requires stack empty AND `residue == ""` (L1).
- Regex terminals: whole-token matches at depth 0 only; DFS never crosses a regex boundary (L2).
- One test suite, parametrized over both engines with `pytest.mark.parametrize`/fixtures (L4); legacy engine stays fully tested.
- Memoization digest: `(tuple(stack), residue)` full-state key (L5).
- Old ⊆ new invariant: the legacy mask is always a subset of the lookahead mask.
- **Tests are NOT committed**: `grammarllm/tests/` is gitignored by user choice; commits carry source/docs only. Run tests with `uv run --with pytest pytest …` from the repo root.
- No mocks (repo test policy): real tokenizer, real torch tensors; synthetic grammar tables are test data.
- All strings live in the tokenizer's surface alphabet (`Ġ`, `Ċ`, U+0100–U+011F byte forms).
- Token-collision policy: when two DFS paths yield the same token id, the first found (scan order, shallowest depth first) wins — `setdefault`, logged at DEBUG.

## File Structure

| File | Responsibility |
|---|---|
| `grammarllm/modules/lookahead.py` (new) | `VocabTrie` (+ per-tokenizer cache), `lookahead_tokens(pda, trie)` DFS, `REGEX_TERMINALS_KEY` constant |
| `grammarllm/modules/automaton.py` (modify) | `residue` field, `lookahead` flag, `regex_terminals` metadata, `eos()`, `clone()`, `reset()`, `apply_lookahead_path()` |
| `grammarllm/scripts/map_terminal_tokens.py` (modify) | writes `REGEX_TERMINALS_KEY` metadata entry into the returned map |
| `grammarllm/modules/logits_processor.py` (modify) | `_valid_ids()` / `_advance()` routing, digest-keyed mask cache, trie ownership |
| `grammarllm/generate_with_constraints.py` (modify) | `token_lookahead=True` parameter on `generate_grammar_parameters` |
| `grammarllm/tests/test_lookahead.py` (new, local-only) | trie, DFS, residue, canonical acceptance, walks, perf |
| `grammarllm/tests/test_logit_processor_beam.py`, `test_dev_review_fixes.py` (modify, local-only) | engine parametrization, vocab-consistent token ids |
| `docs/usage.md`, `README.md` (modify) | document default + opt-out baseline |

---

### Task 1: VocabTrie + per-tokenizer cache

**Files:**
- Create: `grammarllm/modules/lookahead.py`
- Test: `grammarllm/tests/test_lookahead.py` (new)

**Interfaces:**
- Produces: `class VocabTrie` with `children: dict[str, VocabTrie]`, `token_id: int | None`, classmethod `from_tokenizer(tokenizer) -> VocabTrie`, method `child(ch) -> VocabTrie | None`; `get_vocab_trie(tokenizer) -> VocabTrie` (cached per `tokenizer.name_or_path`); `REGEX_TERMINALS_KEY = "__regex_terminal_names__"`.

- [ ] **Step 1: Write the failing tests**

Create `grammarllm/tests/test_lookahead.py`:

```python
"""
Tests for the token-boundary lookahead engine (g_t_r).
Spec: docs/superpowers/specs/2026-07-08-token-boundary-lookahead-design.md
Real tokenizer via the session fixture in conftest.py — no mocks.
"""
import pytest

from lookahead import VocabTrie, get_vocab_trie, REGEX_TERMINALS_KEY


class TestVocabTrie:
    def test_contains_known_tokens(self, tokenizer):
        trie = get_vocab_trie(tokenizer)
        vocab = tokenizer.get_vocab()
        for tok_str in ("a", "Ġhappy", "{\""):
            if tok_str not in vocab:
                continue
            node = trie
            for ch in tok_str:
                node = node.child(ch)
                assert node is not None, f"trie lost {tok_str!r} at {ch!r}"
            assert node.token_id == vocab[tok_str]

    def test_prefix_nodes_have_no_token_id_unless_token(self, tokenizer):
        trie = get_vocab_trie(tokenizer)
        vocab = tokenizer.get_vocab()
        # walk a long token; every strict prefix must carry either None
        # or the id of a shorter REAL token, never a wrong id
        long_tok = max(vocab, key=len)
        node = trie
        for i, ch in enumerate(long_tok):
            node = node.child(ch)
            prefix = long_tok[: i + 1]
            if node.token_id is not None:
                assert vocab.get(prefix) == node.token_id

    def test_cache_returns_same_instance(self, tokenizer):
        assert get_vocab_trie(tokenizer) is get_vocab_trie(tokenizer)

    def test_missing_child_is_none(self, tokenizer):
        trie = get_vocab_trie(tokenizer)
        assert trie.child("￿") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'lookahead'`.

- [ ] **Step 3: Implement**

Create `grammarllm/modules/lookahead.py`:

```python
"""
lookahead.py
============
Token-boundary lookahead (g_t_r): vocab trie + DFS over PDA fragment
transitions, so the model can emit merged tokens that span grammar-terminal
boundaries or end mid-terminal.

Spec: docs/superpowers/specs/2026-07-08-token-boundary-lookahead-design.md
Regex-terminal crossing is out of scope (L2) — see
docs/superpowers/specs/2026-07-08-regex-lookahead-future-work.md
"""

import logging

# Reserved key in map_terminal_tokens carrying the NAMES of regex terminals.
# Written by generate_token_maps, read (and skipped) by PushdownAutomaton.
REGEX_TERMINALS_KEY = "__regex_terminal_names__"


class VocabTrie:
    """Character trie over the tokenizer's raw vocabulary strings
    (surface alphabet: byte-level BPE forms Ġ/Ċ/…)."""

    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children = {}
        self.token_id = None

    def child(self, ch):
        return self.children.get(ch)

    @classmethod
    def from_tokenizer(cls, tokenizer):
        root = cls()
        for tok_str, tok_id in tokenizer.get_vocab().items():
            node = root
            for ch in tok_str:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = cls()
                    node.children[ch] = nxt
                node = nxt
            node.token_id = tok_id
        return root


_TRIE_CACHE: dict = {}


def get_vocab_trie(tokenizer):
    """One trie per tokenizer, built once (~1s for 150k vocab) and cached."""
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    trie = _TRIE_CACHE.get(key)
    if trie is None:
        trie = VocabTrie.from_tokenizer(tokenizer)
        _TRIE_CACHE[key] = trie
        logging.info(f"VocabTrie built for {key}")
    return trie
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit (source only)**

```bash
git add grammarllm/modules/lookahead.py
git commit -m "feat(lookahead): vocabulary trie with per-tokenizer cache"
```

---

### Task 2: PDA residue state + regex-terminal metadata + path replay

**Files:**
- Modify: `grammarllm/modules/automaton.py` (`__init__`, `clone`, `reset`, `eos`, new `apply_lookahead_path`)
- Modify: `grammarllm/scripts/map_terminal_tokens.py` (write `REGEX_TERMINALS_KEY`)
- Test: `grammarllm/tests/test_lookahead.py` (append)

**Interfaces:**
- Consumes: `REGEX_TERMINALS_KEY` from Task 1.
- Produces on `PushdownAutomaton`: `residue: str` (default `""`), `lookahead: bool` (default `False`), `regex_terminals: set[str]`; `apply_lookahead_path(fragments: tuple[str, ...], chars_into_last: int) -> None`; `eos()` now `not self.stack and not self.residue`. `generate_token_maps` adds `map_terminal_tokens[REGEX_TERMINALS_KEY] = [names…]` (list of terminal names WITHOUT the `regex_` prefix; empty list when no regex_dict).

- [ ] **Step 1: Write the failing tests**

Append to `grammarllm/tests/test_lookahead.py`:

```python
from automaton import PushdownAutomaton


def tid(tokenizer, tok_str):
    """Real vocab id of a token string; fails loudly if absent."""
    v = tokenizer.get_vocab()
    assert tok_str in v, f"test needs vocab token {tok_str!r}"
    return v[tok_str]


def brace_grammar(tokenizer):
    """S* → '{' ' ' 'ciao'  as three exact terminals (surface forms)."""
    grammar = {'S*': {'{': ['{', 'Ġ', 'ciao']}}
    map_tt = {
        '{': [tid(tokenizer, '{')],
        'Ġ': [tid(tokenizer, 'Ġ')],
        'ciao': [tid(tokenizer, 'ciao')] if 'ciao' in tokenizer.get_vocab() else [],
        REGEX_TERMINALS_KEY: [],
    }
    return PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)


class TestResidueState:
    def test_defaults(self, tokenizer):
        pda = brace_grammar(tokenizer)
        assert pda.residue == ""
        assert pda.lookahead is False
        assert pda.regex_terminals == set()

    def test_regex_terminal_names_channel(self, tokenizer):
        grammar = {'S*': {'digit': ['digit']}}
        map_tt = {'digit': [tid(tokenizer, '4')], REGEX_TERMINALS_KEY: ['digit']}
        pda = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
        assert pda.regex_terminals == {'digit'}
        # the sentinel key must NOT leak into token→terminal inversion
        assert all(not (isinstance(t, str) and t == REGEX_TERMINALS_KEY)
                   for terms in pda.map_tokens_terminals.values() for t in terms)

    def test_eos_requires_empty_residue(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.stack = []
        pda.residue = "ao"
        assert not pda.eos()
        pda.residue = ""
        assert pda.eos()

    def test_clone_and_reset_carry_residue(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        pda.residue = "ao"
        c = pda.clone()
        assert c.residue == "ao" and c.lookahead is True
        pda.reset()
        assert pda.residue == "" and pda.stack == ['S*']


class TestApplyLookaheadPath:
    def test_full_span_two_terminals(self, tokenizer):
        # token "{ " consumes '{' and 'Ġ' fully → stack ['ciao'], no residue
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ'), 1)
        assert pda.stack == ['ciao']
        assert pda.residue == ""

    def test_mid_terminal_cut_sets_residue(self, tokenizer):
        # token "{ ci": '{' + 'Ġ' full, 'ciao' cut after 2 chars → residue 'ao'
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 2)
        assert pda.stack == []
        assert pda.residue == "ao"
        assert not pda.eos()

    def test_residue_consumed_next_step(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 2)   # residue 'ao'
        pda.apply_lookahead_path(('ao',), 2)              # spell the rest
        assert pda.residue == "" and pda.eos()

    def test_partial_residue_consumption(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 1)   # residue 'iao'
        pda.apply_lookahead_path(('iao',), 1)             # consume 'i' only
        assert pda.residue == "ao"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q -k "Residue or ApplyLookahead"`
Expected: FAIL — `AttributeError` on `residue` / `apply_lookahead_path`, and the sentinel-key test errors inside `__init__` (the inversion loop tries to treat the names list as token ids).

- [ ] **Step 3: Implement**

In `grammarllm/modules/automaton.py`:

(a) top of file, after `import logging`:

```python
try:
    from .lookahead import REGEX_TERMINALS_KEY
except ImportError:                      # bare-import test path
    from lookahead import REGEX_TERMINALS_KEY
```

(b) in `__init__`, replace the beginning of the body with:

```python
        self.stack = [startSymbol]
        self.start_symbol = startSymbol
        self.grammar = grammar
        self.map_terminals_tokens = map
        self.map_tokens_terminals = {}
        # ── lookahead state (spec L1) ─────────────────────────────────
        self.residue = ""            # unconsumed suffix of a partially-covered terminal
        self.lookahead = False       # engine flag, set by generate_grammar_parameters
        self.regex_terminals = set(map.get(REGEX_TERMINALS_KEY, []))

        for non_terminal, value in map.items():
            if non_terminal == REGEX_TERMINALS_KEY:
                continue             # metadata, not a terminal→tokens entry
```

(the rest of the inversion loop is unchanged).

(c) in `clone()`, add alongside the other copied fields:

```python
        new_pda.residue = self.residue
        new_pda.lookahead = self.lookahead
        new_pda.regex_terminals = self.regex_terminals   # read-only, shared
```

(d) in `reset()`, before `self.get_tokens()`:

```python
        self.residue = ""
```

(e) `eos()` body becomes:

```python
        return not self.stack and not self.residue
```

(f) new method after `next_state_terminal`:

```python
    def apply_lookahead_path(self, fragments, chars_into_last):
        """
        Consume a merged token described by its lookahead path.

        fragments : tuple[str, ...]
            Terminal strings the token covers, in order. If self.residue is
            non-empty, fragments[0] IS the residue (grammar already advanced
            for it — only its characters remain to be spelled).
        chars_into_last : int
            How many characters of fragments[-1] the token covers.
            == len(fragments[-1]) → fully consumed, residue becomes "".

        Grammar-advance strategy: a terminal is consumed from the stack the
        moment the token ENTERS it (next_state_terminal); the unspelled
        suffix lives in self.residue. eos() stays False until the residue
        is spelled out.
        """
        for i, frag in enumerate(fragments):
            last = (i == len(fragments) - 1)
            if i == 0 and self.residue:
                if frag != self.residue:
                    raise ValueError(
                        f"Lookahead path expected residue {self.residue!r}, got {frag!r}"
                    )
                self.residue = frag[chars_into_last:] if last else ""
                continue
            self.next_state_terminal(frag)
            self.residue = frag[chars_into_last:] if last else ""
        self.get_tokens()
```

In `grammarllm/scripts/map_terminal_tokens.py`, at the end of `generate_token_maps` just before `return map_terminal_tokens`:

```python
    # Metadata channel for the lookahead engine: which terminal names are
    # open regex classes (DFS must not spell them character-by-character).
    map_terminal_tokens[REGEX_TERMINALS_KEY] = (
        sorted(name[len("regex_"):] for name in regex_dict) if regex_dict else []
    )
```

with the import at the top of that file:

```python
from ..modules.lookahead import REGEX_TERMINALS_KEY
```

(Use a try/except relative-then-bare import mirroring (a) if the module is ever imported bare.)

Also in `check_tokens_conflicts` nothing changes (it iterates table rows, never the map keys). But `PushdownAutomaton.get_tokens()` iterates terminals from the stack scan — the sentinel never appears there, no change needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q`
Expected: all pass (4 from Task 1 + 8 new). Then run the whole suite to prove no regression:
`uv run --with pytest pytest grammarllm/tests/ -q -m "not e2e"` → everything green (the metadata key is invisible to the legacy path).

- [ ] **Step 5: Commit**

```bash
git add grammarllm/modules/automaton.py grammarllm/scripts/map_terminal_tokens.py
git commit -m "feat(lookahead): PDA residue state, path replay, regex-terminal metadata"
```

---

### Task 3: lookahead_tokens DFS

**Files:**
- Modify: `grammarllm/modules/lookahead.py` (add `lookahead_tokens`)
- Test: `grammarllm/tests/test_lookahead.py` (append)

**Interfaces:**
- Consumes: `PushdownAutomaton.recursive_get_tokens(stack)` (FIRST-of-stack scan returning terminal strings), `clone()`, `next_state_terminal(frag)`, `residue`, `regex_terminals`, `map_terminals_tokens`; `VocabTrie`.
- Produces: `lookahead_tokens(pda, trie) -> dict[int, tuple[tuple[str, ...], int]]` — token id → `(fragments, chars_into_last)`. Depth-0 regex terminals contribute `(name,), len(name)` paths (replayed as a plain `next_state_terminal`). First-found path wins on collision (`setdefault`).

- [ ] **Step 1: Write the failing tests**

Append to `grammarllm/tests/test_lookahead.py`:

```python
from lookahead import lookahead_tokens


class TestLookaheadTokens:
    def test_old_subset_of_new(self, tokenizer):
        pda = brace_grammar(tokenizer)
        legacy = set(pda.get_tokens())
        mask = lookahead_tokens(pda, get_vocab_trie(tokenizer))
        assert legacy <= set(mask), "legacy mask must be a subset of lookahead mask"

    def test_merged_span_token_present(self, tokenizer):
        # "{Ġ" is the surface form of "{ " — present in Qwen's vocab
        merged = tokenizer.get_vocab().get("{Ġ")
        if merged is None:
            pytest.skip("tokenizer has no '{ ' merged token")
        pda = brace_grammar(tokenizer)
        mask = lookahead_tokens(pda, get_vocab_trie(tokenizer))
        assert merged in mask
        frags, chars = mask[merged]
        assert frags == ('{', 'Ġ') and chars == 1

    def test_mid_terminal_cut_path(self, tokenizer):
        # any vocab token that spells '{' + 'Ġ' + a strict prefix of 'ciao'
        vocab = tokenizer.get_vocab()
        candidates = {s: i for s, i in vocab.items()
                      if s.startswith("{Ġci") and len(s) < len("{Ġciao")}
        if not candidates:
            pytest.skip("no mid-cut token in this vocab")
        s, i = next(iter(candidates.items()))
        pda = brace_grammar(tokenizer)
        mask = lookahead_tokens(pda, get_vocab_trie(tokenizer))
        assert i in mask
        frags, chars = mask[i]
        assert frags == ('{', 'Ġ', 'ciao')
        assert chars == len(s) - 2          # chars consumed inside 'ciao'

    def test_path_replay_reaches_consistent_state(self, tokenizer):
        pda = brace_grammar(tokenizer)
        mask = lookahead_tokens(pda, get_vocab_trie(tokenizer))
        for token_id, (frags, chars) in list(mask.items())[:20]:
            clone = pda.clone()
            clone.apply_lookahead_path(frags, chars)
            spelled = "".join(frags[:-1]) + frags[-1][:chars]
            # spelled chars must equal the vocab string of the token
            inv = {i: s for s, i in tokenizer.get_vocab().items()}
            assert inv[token_id] == spelled

    def test_regex_terminal_depth0_only(self, tokenizer):
        # S* → '"' digit : digit tokens appear whole; nothing crosses '"'→digit
        grammar = {'S*': {'"': ['"', 'digit']}, }
        four = tid(tokenizer, '4')
        map_tt = {'"': [tid(tokenizer, '"')], 'digit': [four],
                  REGEX_TERMINALS_KEY: ['digit']}
        pda = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
        mask = lookahead_tokens(pda, get_vocab_trie(tokenizer))
        # the quote itself is offered
        assert tid(tokenizer, '"') in mask
        # merged '"4' must NOT be offered (would cross into the regex class)
        merged = tokenizer.get_vocab().get('"4')
        if merged is not None:
            assert merged not in mask
        # and at the NEXT state, the regex token is offered whole
        pda.next_state_terminal('"'); pda.get_tokens()
        mask2 = lookahead_tokens(pda, get_vocab_trie(tokenizer))
        assert four in mask2
        assert mask2[four] == (('digit',), len('digit'))

    def test_dfs_from_residue(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 2)   # residue 'ao'
        mask = lookahead_tokens(pda, get_vocab_trie(tokenizer))
        ao = tokenizer.get_vocab().get("ao")
        if ao is None:
            pytest.skip("no 'ao' token")
        assert ao in mask and mask[ao] == (('ao',), 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q -k LookaheadTokens`
Expected: FAIL at import — `cannot import name 'lookahead_tokens'`.

- [ ] **Step 3: Implement**

Append to `grammarllm/modules/lookahead.py`:

```python
def lookahead_tokens(pda, trie):
    """
    g_t_r: DFS over PDA fragment transitions pruned by the vocab trie.

    Returns { token_id: (fragments, chars_into_last) } — every vocabulary
    token realizable from the current (stack, residue) state, including
    merged tokens spanning terminal boundaries and tokens ending
    mid-terminal. Depth-0 exact matches reproduce the legacy mask, so the
    legacy valid set is always a subset of this one.

    Collision policy (spec): first path found wins (setdefault), scan order.
    Regex terminals (pda.regex_terminals) are yielded whole at depth 0 and
    never crossed (spec L2 — see the regex-lookahead future-work doc).
    """
    results = {}

    def dfs(state, node, consumed):
        if state.residue:
            fragments = [state.residue]
            from_residue = True
        else:
            fragments = state.recursive_get_tokens(list(state.stack))
            from_residue = False

        for frag in fragments:
            if not from_residue and frag in state.regex_terminals:
                if not consumed:
                    # depth 0: regex class participates as whole tokens,
                    # replayed as a plain terminal consumption
                    path = ((frag,), len(frag))
                    for token_id in state.map_terminals_tokens.get(frag, []):
                        results.setdefault(token_id, path)
                continue

            n = node
            alive = True
            for i, ch in enumerate(frag):
                n = n.child(ch)
                if n is None:
                    alive = False
                    break
                if n.token_id is not None:
                    path = (tuple(consumed) + (frag,), i + 1)
                    if results.setdefault(n.token_id, path) != path:
                        logging.debug(
                            f"lookahead collision on token {n.token_id}: kept first path"
                        )
            if alive:
                child = state.clone()
                if from_residue:
                    child.residue = ""
                else:
                    child.next_state_terminal(frag)
                dfs(child, n, consumed + [frag])

    dfs(pda, trie, [])
    return results
```

Note: `recursive_get_tokens` returns a deduplicated terminal list (FIRST-of-stack scan) and never contains the metadata sentinel. The clone-per-node is O(stack); no undo log.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q`
Expected: all pass (some may skip on vocab-dependent tokens; at least `old_subset_of_new`, `regex_terminal_depth0_only`, `path_replay` must run).

- [ ] **Step 5: Commit**

```bash
git add grammarllm/modules/lookahead.py
git commit -m "feat(lookahead): g_t_r DFS with trie pruning and path yields"
```

---

### Task 4: Processor routing + digest mask cache + default-on flag

**Files:**
- Modify: `grammarllm/modules/logits_processor.py` (`__init__`, `_valid_ids`, `_advance`, `_advance_token`, `__call__` step 4, `reset`)
- Modify: `grammarllm/generate_with_constraints.py` (`generate_grammar_parameters` signature)
- Test: `grammarllm/tests/test_lookahead.py` (append)

**Interfaces:**
- Consumes: `lookahead_tokens`, `get_vocab_trie` (Tasks 1/3); `pda.lookahead`, `apply_lookahead_path` (Task 2).
- Produces: `StatelessLogitsProcessor._valid_ids(pda) -> tuple[list[int], dict | None]` (ids + paths, paths `None` on the legacy path); `StatelessLogitsProcessor._advance(pda, token_id, paths=None) -> None`; `self.mask_cache: dict[(tuple, str), dict]` cleared by `reset()`; `generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens, num_return_sequences=1, token_lookahead=True)`.

- [ ] **Step 1: Write the failing tests**

Append to `grammarllm/tests/test_lookahead.py`:

```python
import torch
from logits_processor import StatelessLogitsProcessor


def make_processor(tokenizer, pda, prompt_len=1):
    return StatelessLogitsProcessor(
        tokenizer=tokenizer, base_pdas=[pda],
        sequences_per_prompt=1, prompt_len=prompt_len,
    )


def scores_for(tokenizer):
    return torch.zeros((1, max(tokenizer.get_vocab().values()) + 1))


class TestProcessorRouting:
    def test_legacy_flag_off_matches_get_tokens(self, tokenizer):
        pda = brace_grammar(tokenizer)          # lookahead defaults False
        proc = make_processor(tokenizer, pda)
        ids, paths = proc._valid_ids(pda.clone())
        assert set(ids) == set(pda.get_tokens())
        assert paths is None

    def test_lookahead_flag_widens_mask(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        ids, paths = proc._valid_ids(pda.clone())
        assert set(pda.get_tokens()) <= set(ids)
        assert paths is not None

    def test_mask_cache_hit(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        proc._valid_ids(pda.clone())
        digest = (tuple(pda.stack), pda.residue)
        assert digest in proc.mask_cache
        proc.reset()
        assert proc.mask_cache == {}

    def test_advance_merged_token_via_call(self, tokenizer):
        merged = tokenizer.get_vocab().get("{Ġ")
        if merged is None:
            pytest.skip("no '{ ' merged token")
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        out = proc(torch.tensor([[5]]), scores_for(tokenizer))
        assert torch.isfinite(out[0, merged]).item()
        # feed the merged token back: re-simulation must accept it
        out2 = proc(torch.tensor([[5, merged]]), scores_for(tokenizer))
        cached = proc.pda_cache[(0, (merged,))]
        assert cached.stack == ['ciao'] and cached.residue == ""

    def test_invalid_token_still_raises_in_lookahead(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        with pytest.raises(ValueError):
            proc.get_pda_for_sequence([999999999 % len(tokenizer.get_vocab())])


class TestDefaultOn:
    def test_generate_grammar_parameters_default_lookahead(self, tokenizer):
        from generate_with_constraints import generate_grammar_parameters
        grammar = {'S*': {'a': ['a']}}
        map_tt = {'a': [tid(tokenizer, 'a')], REGEX_TERMINALS_KEY: []}
        pdas, _ = generate_grammar_parameters(tokenizer, grammar, map_tt)
        assert all(p.lookahead for p in pdas)
        pdas_off, _ = generate_grammar_parameters(
            tokenizer, grammar, map_tt, token_lookahead=False)
        assert all(not p.lookahead for p in pdas_off)
```

Add to `grammarllm/tests/pytest.ini` pythonpath so `generate_with_constraints` imports bare: the file lives in `grammarllm/`, so:

```ini
pythonpath = ../modules ../scripts ..
```

(NOTE: `generate_with_constraints.py` uses package-relative imports; if the bare import fails, import it as `from grammarllm.generate_with_constraints import generate_grammar_parameters` instead — the project root is importable via `uv run`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q -k "ProcessorRouting or DefaultOn"`
Expected: FAIL — `AttributeError: ... has no attribute '_valid_ids'` and `TypeError: generate_grammar_parameters() got an unexpected keyword argument 'token_lookahead'`.

- [ ] **Step 3: Implement**

In `grammarllm/modules/logits_processor.py`:

(a) imports, after the existing ones:

```python
try:
    from .lookahead import lookahead_tokens, get_vocab_trie
except ImportError:
    from lookahead import lookahead_tokens, get_vocab_trie
```

(b) in `__init__`, after `self.pda_cache = {}`:

```python
        # Digest-keyed lookahead mask cache: {(tuple(stack), residue): {tid: path}}
        self.mask_cache = {}
        self.vocab_trie = None
        if any(getattr(p, "lookahead", False) for p in base_pdas):
            self.vocab_trie = get_vocab_trie(tokenizer)
```

(c) in `reset()`, add:

```python
        self.mask_cache = {}
```

(d) new methods (place next to `_advance_token`):

```python
    def _valid_ids(self, pda):
        """
        Valid next token ids for *pda*, routed by engine.

        Legacy: (pda.get_tokens(), None).
        Lookahead: (ids, paths) from the digest-memoized g_t_r mask.
        """
        if not getattr(pda, "lookahead", False):
            return pda.get_tokens(), None
        digest = (tuple(pda.stack), pda.residue)
        entry = self.mask_cache.get(digest)
        if entry is None:
            entry = lookahead_tokens(pda, self.vocab_trie)
            if len(self.mask_cache) >= _MAX_CACHE_SIZE:
                for old_key in list(self.mask_cache.keys())[:_MAX_CACHE_SIZE // 4]:
                    del self.mask_cache[old_key]
            self.mask_cache[digest] = entry
        return list(entry.keys()), entry

    def _advance(self, pda, token, paths=None):
        """Consume *token* on *pda*, routed by engine. ValueError on invalid."""
        if not getattr(pda, "lookahead", False):
            pda.next_state(token)
            return
        if paths is None:
            _, paths = self._valid_ids(pda)
        path = paths.get(token)
        if path is None:
            raise ValueError(
                f"Token {token} is not derivable from state "
                f"(stack={pda.stack}, residue={pda.residue!r}) under lookahead"
            )
        pda.apply_lookahead_path(*path)
```

(e) rewrite `_advance_token` to route through them (same semantics as today):

```python
    def _advance_token(self, pda, token):
        if pda.eos():
            return False
        if token in (self.tokenizer.bos_token_id, self.tokenizer.pad_token_id,
                     getattr(self.tokenizer, 'unk_token_id', None)):
            return True
        if token == self.tokenizer.eos_token_id:
            valid, paths = self._valid_ids(pda)
            if token not in valid:
                return False        # EOS forced by the dead-end fallback
            self._advance(pda, token, paths)
            return True
        self._advance(pda, token)
        return True
```

(the docstring from the current `_advance_token` is kept verbatim above the body).

(f) in `__call__` step 4, replace `valid_tokens = pda.get_tokens()` with:

```python
                valid_tokens, _paths = self._valid_ids(pda)
```

(the eos/dead-end branches and the masking code below are unchanged — `valid_tokens` keeps the same type).

In `grammarllm/generate_with_constraints.py`, change the signature and body of `generate_grammar_parameters`:

```python
def generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens,
                                num_return_sequences=1, token_lookahead=True):
```

and after `base_pda = PushdownAutomaton(...)`:

```python
    # Token-boundary lookahead (spec L3): ON by default. Pass
    # token_lookahead=False for the boundary-strict A/B baseline.
    base_pda.lookahead = token_lookahead
```

(the deepcopy loop below copies the flag automatically). Update the function's docstring Parameters section with:

```
    token_lookahead : bool
        Default True — masks are computed with the g_t_r lookahead engine,
        allowing merged tokens across terminal boundaries (canonical
        tokenization). Set False for the legacy boundary-strict engine,
        used as the A/B baseline in preserved-mass measurements.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q` → all pass.
Then the whole suite: `uv run --with pytest pytest grammarllm/tests/ -q -m "not e2e"`.
Expected: pydantic integration/walk tests now run under lookahead-by-default and must still pass (walks pick from wider masks; round-trip assertions unchanged). If a walk fails, debug the DFS — do not pin the tests to `token_lookahead=False`.

- [ ] **Step 5: Commit**

```bash
git add grammarllm/modules/logits_processor.py grammarllm/generate_with_constraints.py
git commit -m "feat(lookahead): processor routing, digest mask cache, default-on flag"
```

---

### Task 5: Parametrize the existing suite over both engines

**Files:**
- Modify: `grammarllm/tests/test_logit_processor_beam.py`, `grammarllm/tests/test_dev_review_fixes.py`, `grammarllm/tests/conftest.py` (all local-only)

**Interfaces:**
- Consumes: everything above.
- Produces: fixture `engine` in `conftest.py`, `params=[False, True]`, `ids=["legacy", "lookahead"]`; engine-independent tests run under both.

- [ ] **Step 1: Add the engine fixture**

Append to `grammarllm/tests/conftest.py`:

```python
@pytest.fixture(params=[False, True], ids=["legacy", "lookahead"])
def engine(request):
    """Parametrizes engine-independent tests over both masking engines."""
    return request.param
```

- [ ] **Step 2: Make the synthetic grammars vocab-consistent and wire the fixture**

The lookahead engine derives token ids by SPELLING terminals through the real
vocab trie, so synthetic grammars must map each terminal to its real vocab id
(a terminal `'a'` mapped to a fake id 10 would diverge between engines).

In `grammarllm/tests/test_logit_processor_beam.py`:

(a) replace the id constants and `make_pda`/`make_proc` helpers:

```python
from lookahead import REGEX_TERMINALS_KEY


def vocab_id(tokenizer, s):
    v = tokenizer.get_vocab()
    assert s in v, f"test requires vocab token {s!r}"
    return v[s]


def make_pda(tokenizer, engine=False, grammar=None, terminals=("a", "b")):
    if grammar is None:
        grammar = {'S*': {t: [t] for t in terminals}}
    map_tt = {t: [vocab_id(tokenizer, t)] for t in terminals}
    map_tt[REGEX_TERMINALS_KEY] = []
    pda = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
    pda.lookahead = engine
    return pda


def make_proc(tokenizer, engine=False, **kw):
    return StatelessLogitsProcessor(
        tokenizer=tokenizer, base_pdas=[make_pda(tokenizer, engine)],
        sequences_per_prompt=1, prompt_len=kw.get("prompt_len", 1),
    )
```

(b) update each test to take `(self, tokenizer, engine)` and pass `engine` into the helpers, replacing the old module-level `A_ID`/`B_ID` with `vocab_id(tokenizer, 'a')` / `vocab_id(tokenizer, 'b')` where asserted. Cache-mechanics tests (`TestCacheHitClone`, `TestLRUEviction`, `TestScoreHistoryReset`) are engine-independent and take the fixture; `TestCallRealTensors.test_call_masks_all_but_grammar_tokens` asserts under legacy the exact set `{a, b}` and under lookahead a **superset** of it:

```python
    def test_call_masks_all_but_grammar_tokens(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        out = proc(torch.tensor([[5]]), scores_for(tokenizer))
        finite = set(torch.isfinite(out[0]).nonzero().flatten().tolist())
        base = {vocab_id(tokenizer, 'a'), vocab_id(tokenizer, 'b')}
        if engine:
            assert base <= finite
        else:
            assert finite == base
```

(`scores_for` as defined in `test_lookahead.py` — move it into `conftest.py` and import from there to avoid duplication.)

In `grammarllm/tests/test_dev_review_fixes.py`: the PDA-semantics tests (`TestFirstOfStackScan`, `TestResetRecomputesTerminals`) test the legacy scan directly — leave them unparametrized. Parametrize `TestAdvanceToken` and `TestScoreHistoryGating` with `engine`, setting `base.lookahead = engine` inside `make_dead_end_processor(tokenizer, engine)`; the forced-EOS semantics must hold under BOTH engines (`_advance_token` routes through `_valid_ids`, so under lookahead the dead-end mask is also empty — 'x' maps to no tokens and nothing is spellable). Update the map to include `REGEX_TERMINALS_KEY: []`.

- [ ] **Step 3: Run the suite**

Run: `uv run --with pytest pytest grammarllm/tests/ -q -m "not e2e"`
Expected: all pass, with the parametrized tests showing `[legacy]`/`[lookahead]` variants. If a lookahead variant fails, it found a real routing bug — fix source, not the test.

- [ ] **Step 4: Commit (only if source changed while fixing)**

```bash
git diff --quiet || { git add -u ':!grammarllm/tests' && git commit -m "fix(lookahead): issues surfaced by engine-parametrized suite"; }
```

---

### Task 6: Canonical-tokenization acceptance + walks + perf budget

**Files:**
- Test: `grammarllm/tests/test_lookahead.py` (append)

**Interfaces:**
- Consumes: pydantic converter (`pydantic_to_productions`), pipeline, `_valid_ids`/`_advance`.
- Produces: the headline acceptance evidence; no source changes expected unless a bug surfaces.

- [ ] **Step 1: Write the tests**

Append to `grammarllm/tests/test_lookahead.py`:

```python
import json
import time
from typing import Literal, Optional
from pydantic import BaseModel
from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions
from grammarllm.generate_with_constraints import (
    get_parsing_table_and_map_tt, generate_grammar_parameters,
)


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]


def build(tokenizer, model, token_lookahead=True):
    prods, rgx = pydantic_to_productions(model)
    pt, mtt = get_parsing_table_and_map_tt(tokenizer, prods, rgx)
    pdas, _ = generate_grammar_parameters(tokenizer, pt, mtt,
                                          token_lookahead=token_lookahead)
    proc = StatelessLogitsProcessor(tokenizer=tokenizer, base_pdas=pdas)
    return proc, pdas[0]


CANONICAL = '{"label": "positive"}'


@pytest.mark.integration
class TestCanonicalTokenization:
    def test_lookahead_accepts_canonical_encoding(self, tokenizer):
        """HEADLINE: the model's natural tokenization must replay through
        the masks token-by-token. The legacy engine fails this by design."""
        proc, base = build(tokenizer, Sentiment, token_lookahead=True)
        pda = base.clone()
        for tid_ in tokenizer.encode(CANONICAL, add_special_tokens=False):
            valid, paths = proc._valid_ids(pda)
            assert tid_ in valid, (
                f"canonical token {tid_} ({tokenizer.decode([tid_])!r}) rejected; "
                f"stack={pda.stack} residue={pda.residue!r}"
            )
            proc._advance(pda, tid_, paths)
        assert pda.eos()

    def test_legacy_rejects_canonical_encoding(self, tokenizer):
        """A/B counterpart: proves the boundary limitation exists in the
        legacy engine (if this ever passes, the A/B story changes)."""
        proc, base = build(tokenizer, Sentiment, token_lookahead=False)
        pda = base.clone()
        rejected = False
        for tid_ in tokenizer.encode(CANONICAL, add_special_tokens=False):
            valid, paths = proc._valid_ids(pda)
            if tid_ not in valid:
                rejected = True
                break
            proc._advance(pda, tid_, paths)
        assert rejected, "legacy engine unexpectedly accepted the canonical encoding"


@pytest.mark.integration
class TestLookaheadWalks:
    @pytest.mark.parametrize("seed", range(5))
    def test_walk_round_trip_under_lookahead(self, tokenizer, seed):
        import random
        proc, base = build(tokenizer, Sentiment, token_lookahead=True)
        rng = random.Random(seed)
        pda = base.clone()
        ids = []
        for _ in range(200):
            if pda.eos():
                break
            valid, paths = proc._valid_ids(pda)
            choices = [t for t in valid if t != tokenizer.eos_token_id] or valid
            t = rng.choice(choices)
            proc._advance(pda, t, paths)
            ids.append(t)
        else:
            raise AssertionError("walk did not terminate")
        data = json.loads(tokenizer.decode(ids))
        Sentiment.model_validate(data)


@pytest.mark.integration
class TestPerfBudget:
    def test_cold_and_hot_mask_cost(self, tokenizer):
        proc, base = build(tokenizer, Sentiment, token_lookahead=True)
        pda = base.clone()
        t0 = time.perf_counter()
        proc._valid_ids(pda)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        proc._valid_ids(pda)
        hot = time.perf_counter() - t0
        # spec target ~5ms typical; 20ms ceiling absorbs CI variance
        assert cold < 0.020, f"cold mask {cold*1e3:.1f}ms over budget"
        assert hot < 0.001, f"cache hit {hot*1e3:.2f}ms not O(1)"
```

- [ ] **Step 2: Run them**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q -m integration`
Expected: all pass. Failure triage: canonical rejection → inspect which boundary the DFS missed (print stack/residue from the assertion message); walk `json.loads` failure → a path replay set the wrong residue; perf failure → check the mask cache digest is actually hit.

- [ ] **Step 3: Full suite + commit (only if source changed)**

Run: `uv run --with pytest pytest grammarllm/tests/ -q -m "not e2e"` → green.

```bash
git diff --quiet || { git add -u ':!grammarllm/tests' && git commit -m "fix(lookahead): issues surfaced by canonical-tokenization acceptance"; }
```

---

### Task 7: A/B preserved-mass E2E + docs

**Files:**
- Test: `grammarllm/tests/test_lookahead.py` (append)
- Modify: `docs/usage.md`, `README.md`

**Interfaces:**
- Consumes: `generate_text`, `compute_generation_analysis` (`grammarllm/utils/generation_analysis.py`).
- Produces: the preserved-mass measurement; user-facing docs for the default and the opt-out.

- [ ] **Step 1: Write the E2E test**

Append to `grammarllm/tests/test_lookahead.py`:

```python
@pytest.mark.e2e
def test_preserved_mass_improves_with_lookahead(tokenizer):
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct")
    from grammarllm.generate_with_constraints import generate_text
    from grammarllm.utils.generation_analysis import compute_generation_analysis

    prods, rgx = pydantic_to_productions(Sentiment)
    pt, mtt = get_parsing_table_and_map_tt(tokenizer, prods, rgx)
    prompt = 'Classify: "I love sunny days." Reply as JSON:'

    results = {}
    for flag in (False, True):
        pdas, streamer = generate_grammar_parameters(
            tokenizer, pt, mtt, token_lookahead=flag)
        r = generate_text(model, tokenizer, prompt, pdas, streamer,
                          max_new_tokens=20, output_scores=True)
        results[flag] = compute_generation_analysis(
            r, tokenizer, label="lookahead" if flag else "legacy")

    legacy, lookahead = results[False], results[True]
    print(f"\npreserved mass: legacy={legacy.mean_preserved_mass:.4f} "
          f"lookahead={lookahead.mean_preserved_mass:.4f} "
          f"delta={lookahead.mean_preserved_mass - legacy.mean_preserved_mass:+.4f}")
    # generated paths differ, so per-step dominance is not guaranteed;
    # the aggregate must not regress (small tolerance for path effects)
    assert lookahead.mean_preserved_mass >= legacy.mean_preserved_mass - 0.02
```

- [ ] **Step 2: Run it**

Run: `uv run --with pytest pytest grammarllm/tests/test_lookahead.py -q -m e2e -s`
Expected: `1 passed` with the printed delta (the headline number for the paper/README).

- [ ] **Step 3: Update docs**

In `docs/usage.md`, add to the "Generation modes" notes list:

```markdown
- Token-boundary lookahead is ON by default: masks include the model's
  natural merged tokens (e.g. `": ` or `{ ci`) even when they span grammar
  terminals, so generation follows the canonical tokenization. Pass
  `generate_grammar_parameters(..., token_lookahead=False)` for the
  boundary-strict legacy engine — useful as the A/B baseline when measuring
  constraint impact with `compare_analyses(..., metric="preserved_mass")`.
  Merged tokens stop at regex-terminal boundaries (see
  `docs/superpowers/specs/2026-07-08-regex-lookahead-future-work.md`).
```

In `README.md`, add a features bullet after the beam-search one:

```markdown
* 🔤 **Canonical tokenization by default** — trie-guided lookahead lets the model emit its natural merged tokens across grammar boundaries (opt-out flag for A/B baselines)
```

- [ ] **Step 4: Final verification + commit**

Run: `uv run --with pytest pytest grammarllm/tests/ -q -m "not e2e"` → green.

```bash
git add docs/usage.md README.md
git commit -m "feat(lookahead): ship token-boundary lookahead by default — docs

Implements docs/superpowers/specs/2026-07-08-token-boundary-lookahead-design.md"
```

---

## Self-Review Notes

- **Spec coverage:** L1 residue states (Task 2), g_t_r DFS + trie (Tasks 1/3), L2 regex depth-0 rule (Task 3, `test_regex_terminal_depth0_only`), L3 default-on + opt-out (Task 4), L4 parametrized single suite (Task 5), L5 full-state digest cache (Task 4), old⊆new (Tasks 3/4/5), canonical acceptance headline + legacy counterpart (Task 6), walks (Task 6), perf budget (Task 6, 20ms ceiling vs 5ms typical target — deviation documented inline), A/B preserved-mass (Task 7), docs (Task 7), collision policy (global constraints + `setdefault` in Task 3), forced-EOS/dead-end invariants preserved (Task 4 `_advance_token` routes both engines).
- **Type consistency:** `_valid_ids(pda) -> (list[int], dict | None)` and `_advance(pda, token, paths=None)` used identically in Tasks 4–7; path type `(tuple[str, ...], int)` matches between `lookahead_tokens` (Task 3) and `apply_lookahead_path` (Task 2); `REGEX_TERMINALS_KEY` defined once (Task 1), consumed in Tasks 2/3/5.
- **Judgment calls:** vocab-dependent tests skip when the Qwen vocab lacks a specific merged token (each class keeps at least one non-skippable assertion); Task 4 note about `generate_with_constraints` bare-vs-package import gives both fallbacks explicitly.
