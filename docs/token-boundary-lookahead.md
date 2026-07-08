# Token-Boundary Lookahead (g_t_r) — How It Works

GrammarLLM's masks used to require every vocabulary token to fit *entirely inside one grammar terminal*. This page shows, visually, how the lookahead engine removes that limit so the model can emit its **natural merged tokens** — spanning several terminals, or stopping mid-terminal.

Measured effect (Qwen2.5-0.5B, same grammar and prompt): mean preserved probability mass **0.065 → 0.160 (+0.095, ~2.5×)**.

- Spec: [`superpowers/specs/2026-07-08-token-boundary-lookahead-design.md`](superpowers/specs/2026-07-08-token-boundary-lookahead-design.md)
- Deferred regex crossing: [`superpowers/specs/2026-07-08-regex-lookahead-future-work.md`](superpowers/specs/2026-07-08-regex-lookahead-future-work.md)

---

## 1. The problem

Grammar terminals: `{` · `␣` (space) · `ciao`. The model's tokenizer, however, likes to merge characters across those boundaries:

```mermaid
flowchart TB
    subgraph WANT["What the model naturally wants to emit"]
        direction LR
        T1["token '{ ci'"] --> T2["token 'ao'"]
    end
    subgraph OLD["What the legacy engine forced"]
        direction LR
        L1["token '{'"] --> L2["token '␣'"] --> L3["token 'ciao'"]
    end
    WANT -.->|"same text, different tokens"| OLD
```

Both spell `{ ciao` — but the bottom sequence is **not** how the model was trained to write it. Forcing it pushes generation off the model's natural distribution: correct text, distorted probabilities.

```mermaid
flowchart LR
    subgraph terminals["grammar terminals"]
        direction LR
        A["'{'"] ---- B["'␣'"] ---- C["'c'  'i'  'a'  'o'"]
    end
    TOK["merged token '{ ci'"]
    TOK ==>|"covers"| A
    TOK ==>|"covers"| B
    TOK ==>|"covers PART of"| C
    style TOK fill:#d33,color:#fff
```

The merged token crosses two boundaries **and** ends in the middle of `ciao`. The legacy engine could never offer it. The lookahead engine can — and remembers the leftover `ao`.

---

## 2. The components

```mermaid
flowchart TB
    TKZ["tokenizer vocabulary<br/>(~150k strings)"] -->|"built once, cached"| TRIE["VocabTrie<br/>modules/lookahead.py"]
    GRAM["LL(1) parsing table"] --> PDA["PushdownAutomaton<br/>state = (stack, residue)"]
    PDA -->|"valid terminal fragments<br/>(FIRST-of-stack scan)"| DFS["g_t_r DFS<br/>lookahead_tokens()"]
    TRIE -->|"prunes impossible branches"| DFS
    DFS -->|"{token_id: path}"| CACHE["mask cache<br/>key = (stack, residue)"]
    CACHE --> PROC["StatelessLogitsProcessor<br/>_valid_ids() / _advance()"]
    PROC -->|"logit mask"| GEN["model.generate()"]
    GEN -->|"chosen token"| PROC
    PROC -->|"apply_lookahead_path()"| PDA
```

Two swappable primitives, everything else untouched: *"what is valid here"* (`_valid_ids`) and *"consume this token"* (`_advance`). `token_lookahead=False` routes both back to the legacy engine (the A/B baseline).

---

## 3. The vocabulary trie

One character-trie over all raw vocab strings (surface alphabet: `Ġ` = space, `Ċ` = newline). Double-circled nodes mark a complete vocabulary token.

```mermaid
graph TD
    root(( )) --> b1["{"]
    b1 --> g1["Ġ"]
    g1 --> c1["c"]
    c1 --> i1["i"]
    b1t((("'{'  id=90")))
    g1t((("'{Ġ'  id=314")))
    i1t((("'{Ġci'  id=…")))
    b1 -.-> b1t
    g1 -.-> g1t
    i1 -.-> i1t
    root --> a1["a"] --> o1["o"]
    o1t((("'ao'  id=3524")))
    o1 -.-> o1t
```

Purpose: while the DFS concatenates grammar fragments character by character, the trie answers two questions in O(1) per char — *"can any vocab token continue this way?"* (child exists) and *"is this prefix itself a complete token?"* (node has `token_id`). Every branch the vocabulary cannot realize dies at the first missing child.

---

## 4. PDA state: `(stack, residue)`

The residue is the unconsumed suffix of a partially-covered terminal. A terminal is popped from the stack the moment a token *enters* it; the unspelled characters wait in `residue`.

```mermaid
stateDiagram-v2
    direction LR
    s0: stack=['ciao','␣','{']<br/>residue=''
    s1: stack=[]<br/>residue='ao'
    s2: stack=[]<br/>residue=''
    [*] --> s0
    s0 --> s1: token '{ ci'<br/>path=(('{','␣','ciao'), 2)
    s1 --> s2: token 'ao'<br/>path=(('ao',), 2)
    s2 --> [*]: eos() = true
    note right of s1
        eos() is FALSE here -
        stack empty but residue
        still to be spelled
    end note
```

`eos()` = stack empty **and** residue empty. Clones and the processor's history cache carry the residue automatically.

---

## 5. The g_t_r DFS

At each PDA state, valid fragments come from the FIRST-of-stack scan (or the residue, if one is pending). The DFS walks each fragment's characters through the trie, **yields** whenever it lands on a complete token, and **recurses** into the next PDA state when a fragment survives whole:

```mermaid
flowchart TB
    START(["dfs(state, trie_node, consumed)"]) --> FRAG{"residue pending?"}
    FRAG -->|yes| RES["fragments = [residue]"]
    FRAG -->|no| SCAN["fragments = FIRST-of-stack scan"]
    RES --> LOOP
    SCAN --> LOOP["for each fragment"]
    LOOP --> RX{"regex terminal?"}
    RX -->|"yes, depth 0"| WHOLE["yield its whole-token matches<br/>(legacy behavior)"]
    RX -->|"yes, deeper"| SKIP["skip — never cross<br/>a regex boundary (v1)"]
    RX -->|no| WALK["walk fragment chars<br/>through the trie"]
    WALK --> DEAD{"child missing?"}
    DEAD -->|yes| LOOP
    DEAD -->|no| TOKEN{"node is a<br/>complete token?"}
    TOKEN -->|yes| YIELD["yield (token_id, path)<br/>path = fragments + cut position"]
    TOKEN --> MORE{"fragment fully<br/>walked?"}
    YIELD --> MORE
    MORE -->|yes| RECURSE["clone PDA, consume fragment,<br/>dfs(child_state, node, consumed+frag)"]
    MORE -->|no| WALK
    RECURSE --> LOOP
```

Worked example — state `stack=['ciao','␣','{']`, walking toward `'{ ci'`:

```mermaid
sequenceDiagram
    participant D as DFS
    participant T as VocabTrie
    participant P as PDA (clones)
    D->>T: walk '{' → node alive, token '{' → yield (('{',),1)
    D->>P: clone, consume '{' → stack ['ciao','␣']
    D->>T: walk '␣' from node '{' → token '{Ġ' → yield (('{','␣'),1)
    D->>P: clone, consume '␣' → stack ['ciao']
    D->>T: walk 'c' → alive · walk 'i' → token '{Ġci' → yield (('{','␣','ciao'),2)
    D->>T: walk 'a' → no child → branch dies
    Note over D: mask now contains '{', '{ ', '{ ci' — all three tokenizations stay legal
```

Key invariant: **old ⊆ new** — depth-0 exact matches reproduce the legacy mask exactly; lookahead only widens it.

---

## 6. Advancing after the model picks a token

The mask stores a replay `path` per token, so consuming a merged token is deterministic — no re-search:

```mermaid
sequenceDiagram
    participant HF as model.generate()
    participant PR as Processor
    participant MC as mask cache
    participant PD as PDA
    HF->>PR: token history [..., '{ ci']
    PR->>MC: digest = (stack, residue)
    MC-->>PR: paths['{ ci'] = (('{','␣','ciao'), 2)
    PR->>PD: apply_lookahead_path
    Note over PD: next_state_terminal('{')<br/>next_state_terminal('␣')<br/>next_state_terminal('ciao')<br/>residue = 'ao'
    PD-->>PR: state (stack=[], residue='ao')
    PR-->>HF: next mask = tokens completing 'ao…'
```

Tokens absent from the mask still raise `ValueError` — the no-silent-bypass invariant is unchanged, as are the dead-end fallback and forced-EOS replay.

---

## 7. What v1 does NOT cross: regex terminals

```mermaid
flowchart LR
    CHUNK["exact chunk<br/>'&quot;n&quot;: '"] --> DIGIT["regex class<br/>digit"]
    TOK2["merged token '&quot;: 4'"]
    TOK2 -.->|"BLOCKED in v1"| CHUNK
    TOK2 -.->|"needs trie ∩ DFA<br/>product walk"| DIGIT
    style TOK2 fill:#999,color:#fff
```

Open classes (`digit`, `json_char`) participate as whole tokens at depth 0 — exactly the pre-lookahead behavior. Crossing them requires walking the trie against the regex's character DFA in lockstep; the full design (residue as a DFA state, longest-match policy, `interegular`/hand-rolled Thompson options) lives in the [future-work doc](superpowers/specs/2026-07-08-regex-lookahead-future-work.md).

---

## 8. Using and measuring it

```python
# ON by default:
pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)

# A/B baseline (legacy boundary-strict engine):
pdas_old, streamer_old = generate_grammar_parameters(
    tokenizer, pars_table, map_tt, token_lookahead=False)

# Quantify the difference on your own grammar/prompt:
a_new = compute_generation_analysis(result_new, tokenizer, label="lookahead")
a_old = compute_generation_analysis(result_old, tokenizer, label="legacy")
compare_analyses([a_old, a_new], metric="preserved_mass")
```

The headline acceptance criterion, verified in the test suite: the tokenizer's **canonical encoding** of any valid sentence replays through the lookahead masks token-by-token — the legacy engine provably rejects it.
