# GrammarLLM — Usage Guide

GrammarLLM constrains the output of any Hugging Face causal LM to a formal grammar. At every generation step, a deterministic pushdown automaton (PDA) derived from your grammar masks the model's logits so that only grammar-valid tokens can be sampled.

- [Pipeline overview](#pipeline-overview)
- [Quick start](#quick-start)
- [Writing grammars](#writing-grammars)
- [Regex terminals](#regex-terminals)
- [Generation modes](#generation-modes)
- [Result format](#result-format)
- [Analyzing constraint impact](#analyzing-constraint-impact)
- [Pydantic models (experimental)](#pydantic-models-experimental)
- [Troubleshooting](#troubleshooting)

## Pipeline overview

```
productions dict (your grammar, <<tag>> notation)
        │
        ▼
get_parsing_table_and_map_tt(tokenizer, productions [, regex_dict])
        │   1. ProductionRuleProcessor — tokenizes <<tags>>, prefix-groups
        │      subword splits, left-factorizes → formal LL(1) grammar
        │   2. parsing_table — FIRST/FOLLOW (fixed-point) → LL(1) table
        │   3. generate_token_maps — terminal → vocabulary token IDs
        ▼
(pars_table, map_terminal_tokens)          # stable per (grammar, tokenizer):
        │                                  # compute once, reuse forever
        ▼
generate_grammar_parameters(tokenizer, pars_table, map_terminal_tokens)
        │   builds the base PushdownAutomaton templates + Streamer
        ▼
(pdas, streamer)
        │
        ▼
generate_text(model, tokenizer, prompt, pdas, streamer, ...)
        │   StatelessLogitsProcessor masks logits each step by
        │   re-simulating the PDA from the token history (LRU-cached),
        │   which makes beam search safe: beams can be reordered or
        │   discarded by HF at any step without corrupting parser state
        ▼
result dict / list  (text, probability, pda_history, scores, ...)
```

Phase 1 is the expensive part and depends only on the grammar and the tokenizer — build it once and reuse it across all generations.

## Quick start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
)

setup_logging()

productions = {
    'S*': ["<<positive>> A", "<<negative>> B"],
    'A':  ["<< happy>>", "<< peaceful>>"],
    'B':  ["<< gloomy>>", "<< angry>>"],
}

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)

result = generate_text(
    model, tokenizer,
    "Classify the sentiment: 'I love sunny days.' Answer:",
    pdas, streamer,
    max_new_tokens=8,
)
print(result["text"])          # e.g. "positive happy"
print(result["probability"])   # joint probability of the sequence
```

For chat models, build the prompt with `create_prompt` and pass `chat_template`:

```python
from grammarllm import create_prompt, chat_template

prompt = create_prompt(
    prompt_input="It's raining and I feel a bit down.",
    system_prompt="You are a sentiment classifier...",
    examples=[
        {"role": "user", "content": "I just got a promotion!"},
        {"role": "assistant", "content": "positive happy"},
    ],
)
result = generate_text(model, tokenizer, prompt, pdas, streamer, chat_template)
```

## Writing grammars

A grammar is a dict `{non_terminal: [production, ...]}`.

| Element | Meaning |
|---|---|
| `S*` | start symbol (required, reserved) |
| `UPPERCASE` symbols | non-terminals |
| `<<some string>>` | exact string terminal — GrammarLLM tokenizes it for you and handles subword splits |
| bare lowercase symbol | open terminal, mapped to vocabulary tokens by a regex you supply in `regex_dict` |
| `ε` | epsilon (empty production) |
| list entries | alternatives (`\|` in formal notation) |

```python
productions = {
    'S*':   ["<<yes>> WHY", "<<no>> WHY"],
    'WHY':  ["<< because>> WORDS", "ε"],
    'WORDS': ["word WORDS", "ε"],          # 'word' comes from regex_dict
}
```

Rules and constraints:

- The grammar must be **LL(1) after tag expansion**. Non-LL(1) grammars are rejected at table-construction time with a `Conflict:` error naming the non-terminal and lookahead.
- Left recursion is not allowed (`'A': ["A x"]`); use right recursion (`'A': ["x A", "ε"]`).
- Alternatives whose exact strings share a leading subword are fine — prefix grouping resolves them (`<<positive>>` vs `<<possible>>` both starting with `pos`).
- Whitespace matters inside `<<tags>>`: `<< happy>>` (leading space) tokenizes differently from `<<happy>>`. Use the leading-space form for words that continue a sentence.
- Two different tags that tokenize to the same single token are reported with a warning — only one of them can ever be generated.

## Regex terminals

Open token classes (numbers, identifiers, arbitrary words) are declared with regexes over **vocabulary token strings** — each regex matches whole tokenizer vocab entries, not characters:

```python
import re

regex_dict = {
    'regex_word':   re.compile(r"[a-zA-Z]+"),   # terminal symbol: word
    'regex_number': re.compile(r"\d+"),         # terminal symbol: number
}

pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions, regex_dict)
```

Key naming contract: `regex_<symbol>` where `<symbol>` is exactly the terminal used in the productions. A ready-made collection is available as `from grammarllm import regex_dict`.

If a terminal matches zero vocabulary tokens you get a warning at setup time and a dead-end (forced EOS) if the parser ever requires it — check warnings in `grammarllm/temp/GRAM-GEN.log`.

## Generation modes

`generate_text(model, tokenizer, text, pdas, streamer, chat_template=None, **options)`

| Mode | Call |
|---|---|
| Greedy (default) | `generate_text(...)` |
| Sampling | `generate_text(..., do_sample=True, top_p=0.9, temperature=0.7)` |
| Beam search | `generate_text(..., num_beams=4)` |
| Multiple outputs | `generate_text(..., num_beams=4, num_return_sequences=3)` |
| Batch of prompts | `generate_text(model, tokenizer, [prompt1, prompt2], ...)` |

Notes:

- `temperature` is applied by Hugging Face's sampling warper only (the grammar processor does not rescale logits).
- With `num_return_sequences > 1` and `num_beams == 1`, sampling is enabled automatically.
- The streamer (live token logging) is active only for `num_beams == 1`; HF does not support streamers with beam search.
- Beam search is safe by construction: parser state is re-derived from each beam's token history at every step, so HF beam reordering cannot corrupt it.
- Any extra kwarg is forwarded to `model.generate()`.

## Result format

Single prompt, `num_return_sequences=1`:

```python
{
    "text": "positive happy",
    "token_ids": [30487, 6247, 151645],
    "probability": 0.947,          # exp(sum of transition log-probs)
    "log_prob": -0.054,
    "pda_history": [[...], ...],   # parser stack after each generated token
    "pda_stack": [],               # final stack; [] = grammar fully satisfied
}
```

- `num_return_sequences=N` → list of N dicts, sorted by probability (descending).
- Batch of prompts → list (one entry per prompt) of the above.
- `return_pda_stack=False` and no `output_scores` → plain string(s) instead of dicts.
- `output_scores=True` additionally returns `transition_scores`, per-step `scores` (post-masking logits) and `original_scores` (pre-masking logits), correctly re-indexed across beam reordering.

A final `pda_stack == []` means the generation is a complete sentence of the grammar. A non-empty stack means generation was cut by `max_new_tokens` or hit a dead end (see the log).

## Analyzing constraint impact

With `output_scores=True` you can quantify how much the grammar forced the model at each step:

```python
from grammarllm.utils.generation_analysis import (
    compute_generation_analysis, print_analysis_summary, plot_generation_analysis,
)

result = generate_text(..., output_scores=True)
analysis = compute_generation_analysis(result, tokenizer, label="my prompt")
print_analysis_summary(analysis)
fig = plot_generation_analysis(analysis)
fig.savefig("analysis.png")
```

Key metric: `preserved_mass` per step — the fraction of the model's free probability mass that fell on grammar-valid tokens. Near 1.0 the model already agreed with the grammar; near 0.0 the constraint forced the output. `compare_analyses([a1, a2], metric="preserved_mass")` overlays multiple runs.

## Pydantic models (experimental)

`grammarllm.utils.pydantic_to_grammar.pydantic_to_productions(Model)` converts a Pydantic v2 model into a productions dict:

```python
from typing import Literal, Optional
from pydantic import BaseModel
from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions

class Person(BaseModel):
    name: Literal["mario", "luisa"]
    mood: Optional[Literal["happy", "sad"]] = None

productions = pydantic_to_productions(Person)
pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
```

**Current output is a compact key/value stream, not JSON** (e.g. `namemariomoodhappy`). A redesign targeting strict JSON output (round-trippable via `Model.model_validate_json`) is specified in [`docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md`](superpowers/specs/2026-07-07-pydantic-json-grammar-design.md) — expect the API to change to `(productions, regex_dict)`.

Unsupported constructs raise `PydanticGrammarError` at conversion time (cyclic `$ref`s, `if/then/else`, open `additionalProperties`, same-type unions, non-string enums, untyped lists).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ValueError: Conflict: NT → terminal ...` at setup | Grammar is not LL(1): two alternatives of the same NT share a lookahead. Left-factor them or make the leading terminals distinct. |
| `Terminal 'x' has no matching token IDs` warning | The exact string / regex matches nothing in the vocabulary. Check leading spaces in `<<tags>>` and regex anchoring. |
| Output stops early, `pda_stack` non-empty, "Dead End" in log | Parser reached a state whose terminals map to no tokens (see warning above) — EOS was forced. |
| `Token conflict: terminal ... overlap` at generation | Two terminals valid in the same parser state map to the same token ID — the grammar is ambiguous at the token level. Rename/re-space the strings or tighten regexes. |
| Slow generation with logging | `setup_logging()` at DEBUG level renders per-step comparison tables. Keep the detail logger at INFO in production. |

Logs are written to `grammarllm/temp/GRAM-GEN.log` (pipeline flow) and `grammarllm/temp/GRAM-DETAIL.log` (per-step distributions, DEBUG only).
