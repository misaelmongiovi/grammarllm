---
icon: lucide/play
---

# Getting Started

## Install

```bash
git clone https://github.com/misaelmongiovi/grammarllm.git
cd grammarllm
uv sync
```

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

# Phase 1 — once per (grammar, tokenizer)
pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
# Phase 2 — once per session
pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)
# Phase 3 — every call
result = generate_text(
    model, tokenizer,
    "Classify the sentiment: 'I love sunny days.' Answer:",
    pdas, streamer,
    max_new_tokens=8,
)
print(result["text"])         # "positive happy"
print(result["probability"])  # 0.947
print(result["pda_stack"])    # [] → grammar fully satisfied
```

## Generation modes

Any extra keyword argument is forwarded to `model.generate()`:

```python
generate_text(..., num_beams=4)                            # beam search
generate_text(..., num_beams=4, num_return_sequences=3)    # top-3 beams, sorted by probability
generate_text(..., do_sample=True, top_p=0.9, temperature=0.7)
generate_text(model, tokenizer, [prompt1, prompt2], ...)   # batch of prompts
generate_text(..., output_scores=True)                     # + per-step pre/post-masking logits
```

Next: the [usage guide](usage.md) covers grammar syntax, regex terminals, result formats, and troubleshooting in full.
