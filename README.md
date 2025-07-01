# ⭐️ GrammarLLM — Grammar-Constrained Natural Language Generation

**GrammarLLM** is a powerful Python library for **grammar-constrained text generation**, built on top of pre-trained Transformer models.
It allows you to define and apply constraints using grammars and regex, ideal for classification, vocabulary restriction, and structured generation.

---

## 🚀 Features

* ✅ **Grammar-constrained generation** — define your own production rules
* 🤗 **Compatible with Hugging Face Transformers**
* 🧩 **Supports pattern matching via Regex and formal grammars**
* ⚡️ **Linear-time decoding via deterministic PDA — efficient grammar-constrained generation**

---

## 📦 Requirements

* Python ≥ 3.10
* 🤗 Transformers ≥ 4.30.0
* PyTorch **or** TensorFlow
* A pre-trained causal language model (e.g., GPT-2, LLaMA)

---

## 🔍 Use Cases

### 1. 🔮 Classification

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from grammarllm.main import generate_grammar_parameters, generate_text
from grammarllm.utils.common_regex import regex_dict
from grammarllm.utils.grammar_utils import get_parsing_table_and_map_tt
from grammarllm.utils.logger import setup_logging

def main():
    setup_logging()

    productions = {'S*': ["positive", "negative", "neutral"]}
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
        tokenizer, productions=productions, regex_dict=regex_dict
    )

    prompt = "Are you happy?"

    LogitProcessor, Streamer = generate_grammar_parameters(
        tokenizer, pars_table, map_terminal_tokens
    )

    output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer)
    print(output)  # → "positive"
```

---

### 2. 🧩 Vocabulary Restriction

```python
productions = {'S*': ["Yes", "I'm", "very", "happy", "!"]}

pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
    tokenizer, productions=productions, regex_dict=regex_dict
)

LogitProcessor, Streamer = generate_grammar_parameters(
    tokenizer, pars_table, map_terminal_tokens
)

text = "Are you happy?"
output = generate_text(model, tokenizer, text, LogitProcessor, Streamer)
print(output)  # → "Yes I'm very happy !"
```

---

### 3. 📐 Structured Generation

```python
productions = {
    'S*': ["<<(>> var <<)>>"]
}

pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
    tokenizer, productions=productions, regex_dict=regex_dict
)

LogitProcessor, Streamer = generate_grammar_parameters(
    tokenizer, pars_table, map_terminal_tokens
)

text = "Insert your prompt here"
output = generate_text(model, tokenizer, text, LogitProcessor, Streamer)
print(output)  # → "(https)"
```

---

## ⚙️ API Overview

### `generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens)`

Creates grammar constraints from the parsing table and terminal mappings.

**Arguments:**

* `tokenizer`: a Hugging Face tokenizer
* `pars_tab`: parsing table returned by `get_parsing_table_and_map_tt()`
* `map_terminal_tokens`: terminal → token/regex mapping returned by `get_parsing_table_and_map_tt()`

**Returns:**

* `LogitProcessor`, `Streamer` — to be passed to `generate_text()`

---

### `generate_text(...)`

Generates grammar-constrained text.

**Main arguments:**

* `model`, `tokenizer`, `text`
* `logit_processor`, `streamer`
* Options to control length, sampling, and generation behavior

---

## 🎛 Customization Guide

### 🧩 Production Rules

* Use `<<some string>>` to **generate exact strings**
* Symbols **without** `<<>>` are **terminals**, matched via regex
* **Uppercase** symbols (e.g., `S*`) are **non-terminals**
* Use `'ε'` for epsilon (empty) transitions


---

### 🔍 Regex Dictionary

You can use or extend the built-in `regex_dict` to define terminal patterns:

```python
import re

regex_alfanum = re.compile(r"[a-zA-Z0-9]+")        # e.g., "abc123"
regex_letters = re.compile(r"[a-zA-Z]+")           # e.g., "Hello"
regex_number = re.compile(r"\d+")                  # e.g., "12345"
regex_decimal = re.compile(r"\d+([.,]\d+)?")       # e.g., "3.14"
regex_var = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")  # e.g., "_varName"
regex_left_round_bracket = re.compile(r"\(")       # e.g., "("
regex_right_round_bracket = re.compile(r"\)")      # e.g., ")"

regex_dict = {
    'regex_alfanum': regex_alfanum,
    'regex_letters': regex_letters,
    'regex_number': regex_number,
    'regex_decimal': regex_decimal,
    'regex_var': regex_var,
    'regex_(': regex_left_round_bracket,
    'regex_)': regex_right_round_bracket,
}
```

To define additional terminal patterns, simply extend `regex_dict`:

```python
new_regex = re.compile(r"your-pattern-here")
regex_dict['regex_custom'] = new_regex
```

Use custom regex keys as terminal symbols in your grammar productions.

---

## ⚠️ Limitations

* ❌ Beam search is **not supported**
* Only works with causal models (GPT, LLaMA, etc.)

---

## 🤝 Contributing

All contributions are welcome: bug reports, feature requests, improvements, or documentation updates.

---

## 📄 License

\[Insert license type here, e.g., MIT, Apache-2.0]

---

## 📫 Contact

📧 Email: [gabriele.tuccio@phd.unict.it](mailto:gabriele.tuccio@phd.unict.it)
