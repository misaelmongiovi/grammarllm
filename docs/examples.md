---
icon: lucide/code
---

# Examples

Runnable scripts live in [`examples/`](https://github.com/misaelmongiovi/grammarllm/tree/main/examples). Run any of them with `uv run python examples/<script>.py`.

## `classification.py`

Hierarchical sentiment classification: the grammar forces exactly one level-1 label followed by one matching level-2 label (e.g. `"positive joyful"`). Shows chat-formatted prompts via `create_prompt` and `chat_template`.

```bash
uv run python examples/classification.py
```

## `regex_terminals.py`

Open token classes with regex terminals: constrains output to the pattern `"<word> is <number>"`. Bare lowercase symbols in productions (`word`, `number`) are mapped to vocabulary tokens via `regex_dict`, keyed as `regex_<symbol>`.

```bash
uv run python examples/regex_terminals.py
```

## `beam_search_analysis.py`

Beam search with `num_beams` / `num_return_sequences`, the full result-dict format (probabilities, per-step PDA stack history), and constraint-impact analysis: computes `preserved_mass` per step and saves a plot.

```bash
uv run python examples/beam_search_analysis.py
```

## Pydantic → grammar

Not a standalone script, but a one-liner shown in the [usage guide](usage.md#pydantic-models): `pydantic_to_productions(Model)` turns any Pydantic v2 `BaseModel` into a grammar that only ever emits strict, round-trippable JSON.
