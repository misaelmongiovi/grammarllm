---
icon: lucide/rocket
---

# GrammarLLM

**GrammarLLM** is a Python library for **grammar-constrained text generation**, built on top of Hugging Face Transformer models. Define a formal grammar, and at every decoding step a deterministic pushdown automaton (PDA) masks the model's logits so only grammar-valid tokens can be sampled — ideal for classification, strict JSON, and other structured generation tasks.

## Features

- **Grammar-constrained generation** — define your own production rules
- **Compatible with Hugging Face Transformers**
- **Linear-time decoding via deterministic PDA** — efficient grammar-constrained generation
- **Beam search support** — stateless PDA re-simulation makes beam reordering safe, and beams that HuggingFace pads with grammar-masked tokens are retired instead of derailing the run
- **Canonical tokenization by default** — trie-guided lookahead lets the model emit its natural merged tokens across grammar boundaries ([how it works](token-boundary-lookahead.md))
- **Sampling, batching, multiple return sequences** — all `model.generate()` modes
- **Constraint-impact analysis** — per-step preserved probability mass, entropy, plots
- **Pydantic → grammar conversion** — derive a strict-JSON grammar from a `BaseModel`

## Requirements

- Python ≥ 3.10
- Transformers ≥ 4.30.0
- PyTorch
- A pre-trained causal language model (e.g. GPT-2, LLaMA, Qwen)

## Installation

```bash
git clone https://github.com/misaelmongiovi/grammarllm.git
cd grammarllm
uv sync
uv run python examples/classification.py
```

## Where to go next

- [Getting started](getting-started.md) — quick start and minimal example
- [Usage guide](usage.md) — full pipeline, grammar syntax, generation modes, troubleshooting
- [Architecture](architecture.md) — modules and how the pipeline fits together
- [Examples](examples.md) — runnable scripts in [`examples/`](https://github.com/misaelmongiovi/grammarllm/tree/main/examples)
- [Token-boundary lookahead](token-boundary-lookahead.md) — visual deep dive into the default masking engine
