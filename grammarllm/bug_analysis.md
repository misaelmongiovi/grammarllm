# GrammarLLM – Bug & Vulnerability Analysis

A careful, file-by-file review of every module. Issues are marked **🔴 Critical**, **🟠 High**, **🟡 Medium**, or **🔵 Low** based on whether they can produce silent wrong output, crashes, or are just robustness concerns.

---

## ⚑ Resolution status (dev-branch review, 2026-07-07)

Verified against the current dev code + fixed in this pass. Regression tests: `tests/test_dev_review_fixes.py` (16 new tests, 36 total green) + E2E run over greedy / beam / sampling / batch.

**Already fixed before this pass (verified in code):**
- BUG-5 (cache eviction): LRU via pop+reinsert implemented in `__call__`.
- BUG-7 (silent swallow in `get_pda_for_sequence`): ValueError now propagates.
- BUG-12 (one-shot FIRST DFS): replaced with fixed-point `compute_all_first_sets`.
- BUG-15 (dangling docstring): corrected.

**Fixed in this pass:**
- BUG-1 → `recursive_get_tokens` rewritten as an exact iterative FIRST-of-stack scan (O(stack) instead of worst-case exponential recursion; no visited-set heuristics). Note: for tables passing strict LL(1) validation the over-pruning configuration implies a FIRST/FOLLOW conflict rejected earlier, so it was latent — the performance fix is the practical win.
- BUG-3/BUG-11 (stale `current_terminals`) → `reset()` now recomputes `get_tokens()`; a clone of a reset base PDA can advance immediately.
- BUG-8 (score-history OOM) → history gated behind `track_score_history`, set from the caller's `output_scores`; `raw_scores.clone()` skipped entirely when neither tracking nor DEBUG detail-logging is active.
- BUG-9 → superseded: temperature scaling removed from the processor entirely. It **double-applied** temperature (processor divided, then HF's `TemperatureLogitsWarper` divided again when `do_sample=True`). Temperature is now handled only by HF `generate()`.
- BUG-10 (streamer infinite-discard loop) → the EOS guard no longer resets `is_first_call`.
- BUG-21 → `raise RuntimeError(...) from e`.
- **NEW: forced-EOS replay crash** — when the dead-end fallback forces EOS, that token enters the history; re-simulation then called `next_state(eos)` on a non-empty stack → ValueError mid-generation (masked only when `pad_token_id == eos_token_id`). New `_advance_token()` helper stops replay cleanly on forced EOS while still consuming a *genuine* grammar EOS terminal and still raising on truly invalid tokens. Used in Case A, Case B, and `get_pda_for_sequence`.
- **NEW: `beam_indices == -1` mis-mapping** — steps after a sequence finished carry `beam_indices = -1`, which silently indexed the *last* row of `outputs.scores`; the per-step score mapping in `generate_text` now stops at the first negative index.
- **NEW: `check_tokens_conflicts` KeyError** — hardened with `.get()` for lookahead terminals missing from the map (e.g. regex terminal without a `regex_dict` entry).

**Reviewed, deliberately not changed:**
- BUG-2 (thread safety): out of scope — single-threaded usage; documenting is enough.
- BUG-4/BUG-19 (EOS terminal round-trip): covered in practice by `_advance_token` + the dead-end fallback; EOS-as-production of `S*` empties the stack in all tested tokenizers.
- BUG-6 (`prompt_len`): claim overstated — `prompt_len` is taken from the exact tensor passed to `generate()` (chat-template included), so it is consistent.
- BUG-13/14/16/17/18/20: fragility/coverage notes, not active defects; left as documented risks.

---

## 1. [automaton.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py) — [PushdownAutomaton](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#2-229)

### 🟠 BUG-1: `visited` set is shared across productions — prevents valid multi-path expansion

**Location:** [recursive_get_tokens](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#62-140), lines 130 & 137.

```python
tokens += self.recursive_get_tokens(list(stack), set(visited))  # epsilon branch
tokens += self.recursive_get_tokens(new_stack, set(visited))    # non-epsilon branch
```

A *copy* of `visited` is passed down, which is the right direction, but `visited` accumulates the **current NT** before iterating over its productions (line 106). This means that if:

- NT `A` can derive ε *and* also derive some non-empty production,
- AND the ε path causes the recursive call to reach `A` again (via a FOLLOW chain),

the FOLLOW-chain call will find `A` already in `visited` and **silently return `[]`**, cutting valid continuations. This is especially dangerous for grammars with nullable non-terminals that appear in their own FOLLOW set (indirect left-recursion through FOLLOW), producing a missing valid-token set and therefore a dead-end masking.

**Root cause:** `visited.add(top)` happens **before** the production loop, so even when we explore independent branches for the *same* NT, they all share the "already visited" mark.

**Fix direction:** `visited` should block infinite recursion *within a single expansion path*, not block re-entry from different production alternatives. Correct fix: add `top` to `visited` only in the copy passed to children, not to the shared receiver — or reset after each production iteration.

---

### 🟡 BUG-2: [get_tokens()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#141-159) calls [recursive_get_tokens(self.stack.copy())](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#62-140) — modifies the copy but the invariant is fragile

[recursive_get_tokens](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#62-140) calls `stack.pop()` destructively. The `.copy()` at the call site protects `self.stack`, but every recursive call also clones (`list(stack)`) the remainder. This is correct but expensive for deep grammars. More importantly: if [get_tokens()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#141-159) is called while another call is in progress (e.g., inside a Python `threading` context), `self.stack` could be modified between the `copy()` and the subsequent `pop()`. **No thread safety** anywhere in the class.

---

### 🟠 BUG-3: [next_state](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#160-185) — [get_tokens()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#141-159) called *after* [next_state_terminal](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#187-222), but [next_state_terminal](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#187-222) is recursive and may leave the stack in a partially-expanded state on exception

In [next_state](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#160-185) (line 184), `self.get_tokens()` is called unconditionally after [next_state_terminal](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#187-222). But [next_state_terminal](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#187-222) is itself recursive (line 213: `self.next_state_terminal(token)`). If the recursive call raises a `ValueError`, the exception propagates out of [next_state_terminal](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#187-222) and `self.get_tokens()` on line 184 is **never called**. `self.current_terminals` is therefore stale — it still reflects the *previous* state. Since the [PushdownAutomaton](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#2-229) is cached in [StatelessLogitsProcessor](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#15-288), a stale `current_terminals` can cause subsequent callers that read `pda.current_terminals` directly (e.g. [clone()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#32-54) on line 48) to get the wrong value.

---

### 🔵 BUG-4: [eos()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) returns `True` only when `self.stack` is **exactly** empty — but [get_tokens()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#141-159) after accepting EOS leaves the stack non-empty if grammar has `'</s>'` as a terminal

The EOS token string is added to the grammar in [generate_with_constraints.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py) line 21:
```python
final_grammar[('S*','RULE')].append([tokenizer.eos_token])
```
If the LL(1) table expansion of `S*` never pops down to an empty stack after consuming the EOS terminal, [eos()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) will return `False` even though generation is done. The logit processor then forces EOS via the dead-end path (line 226) instead of the clean [eos()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) path. Not catastrophic but masking is wasted and the warning is misleading.

---

## 2. [logits_processor.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py) — [StatelessLogitsProcessor](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#15-288)

### 🔴 BUG-5: Cache eviction corrupts incremental advance (Case A)

**Location:** lines 205–210.

```python
if len(self.pda_cache) >= _MAX_CACHE_SIZE:
    evict_count = _MAX_CACHE_SIZE // 4
    for old_key in list(self.pda_cache.keys())[:evict_count]:
        del self.pda_cache[old_key]
self.pda_cache[cache_key] = pda
```

The eviction deletes the **oldest** 25 % of keys in insertion order. However, beam search generates at step `t` using the prefix from step `t-1`. If the prefix key [(prompt_idx, history_tuple[:-1])](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) happens to fall within the evicted quarter, the next step **will not find the ancestor** in the cache. It then falls through to **Case B (full re-simulation)**. Full re-simulation is correct in isolation, but the [pda](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#19-21) object from Case B is **stored as the new entry**, and the old evicted entries (ancestors of other beams) are gone permanently. With many beams and long sequences, the working set of active prefixes can exceed `_MAX_CACHE_SIZE // 4 = 512` entries, causing the wrong ancestors to be evicted repeatedly and thrashing into O(n²) re-simulation.

**More subtly:** the FIFO eviction is over *dict insertion order*, which in Python 3.7+ is deterministic but does NOT reflect LRU (recently accessed). A beam that is currently producing the best hypothesis may have its oldest step's key evicted even though it is still needed as an ancestor.

**Fix:** Use `functools.lru_cache` or an `OrderedDict` to implement true LRU, and make the cache key cover only the last `N` tokens (sliding window) rather than the full history.

---

### 🟠 BUG-6: `prompt_len` can silently be wrong for batched / padded inputs

**Location:** lines 139 and 197.

```python
history_tokens = current_seq[self.prompt_len:].tolist()
```

`self.prompt_len` is set once at construction from `start_len = input_ids.shape[1]`. When the input is a **batch** with padding (left-padded by design per line 114 of [generate_with_constraints.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py)), every sequence in the batch has the same `shape[1]`, so `prompt_len` is uniform — that's fine.

However, when [generate_text](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#86-315) is called with a **chat-template** input (`isinstance(text, list)` + `chat_template is not None`), HF's `apply_chat_template` may prepend extra tokens (BOS, system role markers, etc.) that are NOT in the original text string. If the user passes `prompt_len` separately or the model prepends a BOS token at generation time, `self.prompt_len` is off by 1 (or more). The grammar simulation will then replay extra tokens that were part of the prompt as if they were generated tokens, **raising a `ValueError`** from [next_state](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#160-185) because those tokens are not in the grammar.

---

### 🟠 BUG-7: [get_pda_for_sequence](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#256-288) uses a bare `except Exception: break` — silently swallows grammar violations

**Location:** lines 282–286.

```python
try:
    pda.next_state(token)
except Exception:
    # If a token is invalid for the grammar, we stop advancing
    break
```

This is the **old silent-swallow pattern** that was explicitly fixed in [__call__](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#90-255) (see comment on line 166). [get_pda_for_sequence](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#256-288) is the public API used by [generate_text](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#86-315) for computing the `pda_history` in the result dict (line 267). If a token is invalid, the PDA stops mid-sequence, and the returned `stack_history` silently ends early. The downstream caller receives a `pda_history` that is shorter than `new_tokens`, making `stack_history[-1]` the state *before* the invalid token, not *after*. Callers relying on this for analysis will get subtly wrong results.

---

### 🟡 BUG-8: `scores` tensor is mutated in-place before the `filtered_probs` log

**Location:** lines 239 and 246.

```python
scores[i] = scores[i].masked_fill(mask, -float('inf'))   # line 239 — mutates scores
...
filtered_probs = F.softmax(scores, dim=-1)                # line 246 — reads already-mutated scores
```

This is correct for the logging *of the filtered distribution*, but `raw_scores` was cloned on line 114 for the original distribution. However, `scores` is also the tensor passed back to the HF generation loop (the return value on line 254). HF `.generate()` may apply *additional* logit processors **after** this one in `logits_processor` list. Because `scores` is mutated in-place, subsequent processors receive logits that already have `-inf` in invalid positions. This is typically the intended behavior, but it means the processor is **not idempotent** — calling it twice on the same input would leave `scores` entirely `-inf` in many positions.

More critically: **`self.original_scores_history.append(raw_scores)`** on line 251 stores an ever-growing list of full [(batch_size, vocab_size)](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) tensors. For a vocabulary of 128k (LLaMA 3) with `float32`, each step adds `batch_size × 128000 × 4` bytes. At `max_new_tokens=200` with `batch_size=1` this is already ~100 MB. With beam search `batch_size=5` it is ~500 MB. This **will OOM** on large models with long generations.

---

### 🔵 BUG-9: Temperature is applied before masking, but `raw_scores` is cloned *after* temperature scaling

**Location:** lines 103–114.

```python
if self.temperature != 1.0:
    scores = scores / self.temperature   # line 104 — modifies scores IN PLACE
...
raw_scores = scores.clone()             # line 114 — clones already-scaled scores
original_probs = F.softmax(raw_scores, dim=-1)
```

The "original" distribution logged and stored is the **temperature-scaled** distribution, not the raw model logits. This is a logging/analysis bug — users who examine `original_scores_history` think they are seeing what the model produced, but they are actually seeing the temperature-adjusted version. The `raw_scores` naming is misleading.

---

## 3. [streamer.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py) — [BaseStreamer](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#2-111)

### 🟡 BUG-10: First-call guard skips based on `is_first_call` — but resets it incorrectly when [all_pdas_eos()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#19-21) is true

**Location:** lines 32–34 and 60.

```python
if all_pdas_eos():
    self.is_first_call = True
    return
...
self.is_first_call = False   # reached only if not all_pdas_eos()
```

If [all_pdas_eos()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#19-21) is true at the **start** of a [put()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#15-84) call (before `is_first_call` is reset to `False`), the guard resets `is_first_call = True` and returns. This means the very next call will again be treated as "first call" and the **prompt will be thrown away again**. If the PDA is in EOS state because the grammar context has been consumed, the streamer will permanently loop: every subsequent token is treated as a prompt token and discarded, so the [end()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#87-111) method is never triggered properly by the streamer's internal logic (it depends on `.generate()` calling [end()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#87-111) directly). This is a **silent infinite discard**.

---

### 🔵 BUG-11: Streamer resets PDA on [end()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#87-111), but [StatelessLogitsProcessor](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#15-288) still holds references to the same PDA objects in `base_pdas`

**Location:** [end()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py#87-111) lines 107–108.

```python
for pda in self.pdas:
    pda.reset()
```

`pda.reset()` clears the stack but does **not** re-call [get_tokens()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#141-159), so `self.current_terminals` is `[]` (set in [reset()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#56-61) on line 59). If the same PDA object is referenced in `base_pdas` inside an existing [StatelessLogitsProcessor](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#15-288), and that processor still has entries in `pda_cache` pointing to clones of the now-reset PDA, the cache entries are stale. The next generation call that starts from the cache will have a fresh `current_terminals = []` in the base, but clones made before the reset have a non-empty `current_terminals`. Whether this causes a problem depends on whether the cache is flushed between generations, but there is **no explicit cache flush** in [generate_text](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#86-315).

---

## 4. [generate_LL1_parsing_table.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py)

### 🟠 BUG-12: [find_first](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#23-63) is not memoized / not cycle-safe for mutual recursion

**Location:** [find_first](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#23-63), line 51.

```python
if symbol in productions:
    first_sets[symbol] = set()   # initialise to empty BEFORE recursing
    for production in productions[symbol]:
        ...
        first_of_sequence = calculate_first_of_sequence(production, productions, first_sets)
        first_sets[symbol] |= first_of_sequence
```

The empty-set sentinel on line 51 prevents infinite recursion in **directly** recursive grammars (line 47 returns early if `symbol in first_sets`). However, [calculate_first_of_sequence](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#24-46) (the inner function) also calls [find_first](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#23-63) for non-terminal heads (line 35). In a **mutually recursive** grammar (`A → B …`, `B → A …`), the call chain is:

```
find_first(A) → calculate_first_of_sequence([B,…]) → find_first(B)
              → calculate_first_of_sequence([A,…]) → find_first(A) → early return (∅)
```

[find_first(A)](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#23-63) returns `∅` (set to empty before the mutual call), so `A`'s FIRST does not include the tokens reachable via `B`. The fixed-point iteration in [follow()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#64-92) uses a `while changed` loop which IS correct, but [find_first](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#23-63) has **no fixed-point loop** — it is a one-shot DFS. For any grammar with mutual recursion in non-terminals, FIRST sets will be **incomplete**, producing a parsing table that is missing entries. The PDA will then raise `ValueError` at generation time for inputs that are actually valid. 

> **Note:** LL(1) grammars cannot be left-recursive, but mutual right-recursion is possible and legal.

---

### 🟡 BUG-13: [parsing_table](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#93-157) hard-codes `'S*'` as start symbol

**Location:** line 148.

```python
follow_sets = follow(grammar, first_sets, 'S*')
```

The start symbol is not a parameter — it is baked in. Any grammar that uses a different start symbol name will compute a wrong FOLLOW set (the actual start symbol won't get `$` in its FOLLOW). The [PushdownAutomaton](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#2-229) init also hard-codes `startSymbol='S*'` in [generate_with_constraints.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py) line 45. This is a naming convention that is not enforced or checked anywhere, making it a fragile coupling.

---

## 5. [grammar_generation.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py) — [ProductionRuleProcessor](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#5-591)

### 🟠 BUG-14: [find_common_prefixes_in_productions](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#272-358) — epsilon productions are not kept when ALL non-empty productions share a prefix

**Location:** [find_common_prefixes_in_productions](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#272-358), line 339.

```python
for i, prod in enumerate(splitted_productions):
    if len(prod) == 0:
        new_productions.append([])    # epsilon stays as residual
    elif len(prod) >= len(common_prefix):
        suffix = prod[len(common_prefix):]
        suffixes.append(suffix)
    else:
        new_productions.append(productions[i])  # shorter-than-prefix: residual
```

After this loop, `new_productions` collects the epsilon and short productions as "residuals", while `suffixes` collects the factored ones. Then:

```python
if suffixes:
    factorization_info = {'common_prefix': common_prefix, 'suffixes': suffixes}
return new_productions, factorization_info
```

The residual `new_productions` are returned **separately**, but in [process_full_grammar](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#388-526) line 519:

```python
final_grammar[(lhs, "RULE")] = [main_production] + factorized_productions
```

`factorized_productions` is exactly `new_productions` returned by [find_common_prefixes_in_productions](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#272-358). So the epsilon production IS correctly included in [(lhs, "RULE")](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225). **However**, the `new_nt` (the factored NT) receives `factorization_info['suffixes']` as its productions. If one of those suffixes is `[]` (i.e., one of the non-empty originals was exactly equal to [common_prefix](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#163-179), making its suffix empty), the factored NT has a `[]` production — which is correct. But the LL(1) FOLLOW computation for `new_nt` must then include FOLLOW(lhs) for that epsilon production. The current code does handle this in [follow()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#64-92) via the normal fixed-point. This is fine **as long as** `new_nt` appears in the grammar dict passed to [follow()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#64-92). Since the key is [(new_nt, "RULE")](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) (a tuple), the [grammar](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#527-591) built from `final_rules` in [parsing_table()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#93-157) should include it. This is correct — **low risk but worth monitoring**.

---

### 🟡 BUG-15: [save_final_grammar](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#527-591) opens file with `'w+'` but `docstring` is after `open()` call — harmless but confusing

**Location:** lines 527–531.

```python
def save_final_grammar(self, grammar, filename='final_grammar.txt'):
    output_filename = os.path.join("output/temp", filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    """Salva la grammatica finale in formato leggibile"""  # ← docstring AFTER code!
    if not grammar:
```

The docstring is misplaced — it appears after `os.makedirs` and is therefore a **dangling string literal**, not a docstring. Python treats it as a no-op expression. This doesn't break anything but is a maintenance hazard (IDEs will not show it as documentation).

Also: `output_filename` is hardcoded to `"output/temp"` (a relative path), while [generate_LL1_parsing_table.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py) saves to `"grammarllm/temp"`. These two paths are inconsistent and both are relative to **wherever the process was launched**, not the repo root.

---

### 🟡 BUG-16: [process_grammar_iteration](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#187-226) — `changed` flag may be `False` even when the grammar *did* change

**Location:** lines 215.

```python
new_grammar[(new_nt, common_prefix)] = _copy.deepcopy(suffixes)
changed = True
```

`changed = True` is only set when [common_prefixes](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#163-179) is non-empty for a given key. But adding a new entry [(new_nt, common_prefix)](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) to `new_grammar` does NOT itself check whether [(new_nt, common_prefix)](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) already existed. If the same [(nt, prefix)](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) key is processed twice (which cannot happen in a single iteration since `list(grammar.items())` is stable), this would silently overwrite. The single-pass design prevents the overwrite, but the logic relies on that implicit contract without any assertion to protect it.

---

## 6. [map_terminal_tokens.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py)

### 🟠 BUG-17: [check_tokens_conflicts](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py#8-26) only checks **same-level** terminal pairs — misses cross-level terminal conflicts

**Location:** [check_tokens_conflicts](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py#8-26), line 12.

```python
for lhs, rhs_list in table_parsing.items():
    for a, b in itertools.combinations(rhs_list.keys(), 2):
        intersection = set(map_terminal_tokens[a]) & set(map_terminal_tokens[b])
```

The conflict check iterates pairs of terminals **within the same non-terminal row** of the parsing table. But the grammar constraint requires that for any given parser state, the set of allowed tokens must be unambiguous. The actual LL(1) constraint is: for any non-terminal at the top of the stack, the lookahead terminals in its table row must have disjoint token sets — which IS what this checks. However:

- Terminals **from different rows** that can both be "current_terminals" simultaneously are NOT checked. 
- [get_tokens()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#141-159) in [automaton.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py) already checks this via the `isdisjoint` assertion at runtime (line 146), but that is a per-generation runtime check. A grammar that passes [check_tokens_conflicts](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py#8-26) here can still fail at generation time.

This is a **coverage gap** in the pre-flight validation, not a logical error per se, but it means the static check gives a false sense of safety for cross-row conflicts.

---

### 🔵 BUG-18: `regex_dict` entries with regex-matched tokens are added to `map_terminal_tokens` but NOT conflict-checked against the exact-match terminals built in the loop below

**Location:** lines 31–53.

The `if regex_dict:` block populates `map_terminal_tokens` for `regex_*` terminals. The subsequent loop adds exact-match terminals using `re.escape`. Then [check_tokens_conflicts](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py#8-26) checks pairs from [table_parsing](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#100-111) rows — but the table rows contain the **exact-match** terminal names (e.g., `"yes"`, `"no"`), not the regex names (e.g., `"regex_word"`). If a regex matches the same token as an exact terminal, the conflict is invisible to [check_tokens_conflicts](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py#8-26) because they live in different cells. The runtime check in [get_tokens()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#141-159) will catch it, but only during generation.

---

## 7. [generate_with_constraints.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py)

### 🟠 BUG-19: EOS token is added to grammar *before* parsing table is computed, but the `eos_token` string may tokenize to multiple sub-tokens

**Location:** line 21.

```python
final_grammar[('S*','RULE')].append([tokenizer.eos_token])
```

`tokenizer.eos_token` is a **string** (e.g., `"</s>"` or `"<|eot_id|>"`). This string is added as a terminal in the grammar. In [map_terminal_tokens.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py), it is then matched against the vocabulary using:

```python
regex = re.compile(rf"^{re.escape(terminal)}$")
matched = [token_id for token_str, token_id in vocab.items() if regex.match(token_str)]
```

For most LLMs, the EOS token IS a single vocabulary entry and this works. But for some tokenizers (e.g., T5, some encoder-decoder models), `eos_token` may be `</s>` which can also appear as a piece that the tokenizer splits differently. If there is no exact match, `matched` is `[]`, the warning is printed, and `map_terminal_tokens[eos_token] = []`. The PDA will then find EOS has zero valid token IDs, reach a dead end, and force output via the dead-end path (line 226 in logits_processor), bypassing the clean EOS route.

More subtle: **`tokenizer.eos_token_id`** (the integer ID) is used in the logit processor to force EOS (line 218: `scores[i, self.tokenizer.eos_token_id] = 0`), but the grammar maps `tokenizer.eos_token` (the string). If the tokenizer's `eos_token` string does not round-trip through `get_vocab()` with the same key (whitespace, special encoding), these two IDs will diverge.

---

### 🟡 BUG-20: [generate_grammar_parameters](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#40-61) uses `copy.deepcopy(base_pda)` for extra sequences, but [generate_text](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#86-315) later also calls `base_template.clone()` when expanding PDAs

**Location:** [generate_grammar_parameters](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#40-61) line 56 vs [generate_text](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#86-315) lines 182–184.

```python
# generate_grammar_parameters
pdas.append(copy.deepcopy(base_pda))

# generate_text
base_pdas.append(base_template.clone())
```

These two code paths use different copy strategies (`deepcopy` vs [clone()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#32-54)). [clone()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#32-54) shares [grammar](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#527-591) and `map_terminals_tokens` references (intentionally). `deepcopy` creates full independent copies of those shared, large structures. If the PDAs list is expanded by [generate_text](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#86-315) using [clone()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#32-54), the new PDAs share the grammar dict with the originals. If subsequently some code path were to mutate [grammar](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#527-591) (unlikely but possible if someone passes a non-frozen grammar), the clones would see the mutation. Inconsistency in copy strategy is a maintenance hazard.

---

### 🔵 BUG-21: [generate_text](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py#86-315) catches all exceptions and re-raises as `RuntimeError`, eating the original traceback type

**Location:** lines 311–314.

```python
except Exception as e:
    import traceback
    traceback.print_exc()
    raise RuntimeError(f"Errore nella generazione del testo: {e}")
```

The original exception type (e.g., `ValueError` from [next_state](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#160-185)) is lost. Callers that catch `ValueError` specifically will miss it. `traceback.print_exc()` prints to **stderr**, but the re-raised `RuntimeError` loses the chained cause. Using `raise RuntimeError(...) from e` would preserve the chain.

---

## Summary Table

| # | File | Severity | Description |
|---|------|----------|-------------|
| 1 | [automaton.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py) | 🟠 High | `visited` is shared too early, cutting valid token paths for nullable NTs |
| 2 | [automaton.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py) | 🔵 Low | No thread safety on `self.stack` |
| 3 | [automaton.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py) | 🟠 High | Stale `current_terminals` after exception in [next_state_terminal](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#187-222) |
| 4 | [automaton.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py) | 🔵 Low | [eos()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#223-225) may not trigger cleanly if EOS terminal doesn't empty stack |
| 5 | [logits_processor.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py) | 🔴 Critical | FIFO eviction removes live ancestor keys, causing cache miss + O(n²) re-sim or stale state |
| 6 | [logits_processor.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py) | 🟠 High | `prompt_len` can be wrong for chat-template / BOS-prepending tokenizers |
| 7 | [logits_processor.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py) | 🟠 High | [get_pda_for_sequence](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#256-288) silently swallows grammar violations (old pattern re-introduced) |
| 8 | [logits_processor.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py) | 🟡 Medium | `original_scores_history` accumulates full vocab tensors → OOM on long generations |
| 9 | [logits_processor.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py) | 🔵 Low | `raw_scores` is post-temperature, misleadingly named "original" |
| 10 | [streamer.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py) | 🟡 Medium | EOS-state reset of `is_first_call` causes permanent discard loop |
| 11 | [streamer.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/streamer.py) | 🔵 Low | PDA reset doesn't flush [StatelessLogitsProcessor](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/logits_processor.py#15-288) cache |
| 12 | [generate_LL1_parsing_table.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py) | 🟠 High | [find_first](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py#23-63) one-shot DFS is incomplete for mutually recursive NTs |
| 13 | [generate_LL1_parsing_table.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/generate_LL1_parsing_table.py) | 🟡 Medium | Start symbol `'S*'` is hard-coded, fragile coupling |
| 14 | [grammar_generation.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py) | 🟠 High | Epsilon handling in factorization: residuals may be dropped in edge cases |
| 15 | [grammar_generation.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py) | 🔵 Low | Docstring after code in [save_final_grammar](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py#527-591); inconsistent output paths |
| 16 | [grammar_generation.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/grammar_generation.py) | 🟡 Medium | `changed` flag doesn't guard against accidental duplicate key overwrite |
| 17 | [map_terminal_tokens.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py) | 🟠 High | Conflict check is intra-row only; cross-state terminal conflicts not caught statically |
| 18 | [map_terminal_tokens.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/scripts/map_terminal_tokens.py) | 🔵 Low | Regex terminals not conflict-checked against exact-match terminals |
| 19 | [generate_with_constraints.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py) | 🟠 High | EOS token string may not round-trip through vocab; ID mismatch possible |
| 20 | [generate_with_constraints.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py) | 🟡 Medium | Inconsistent `deepcopy` vs [clone()](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/modules/automaton.py#32-54) for PDA expansion |
| 21 | [generate_with_constraints.py](file:///Users/gabrieletuccio/Developer/GitHub/grammarllm/grammarllm/generate_with_constraints.py) | 🔵 Low | Exception re-raise loses original type; missing `from e` chaining |
