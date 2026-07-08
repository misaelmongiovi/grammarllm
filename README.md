# ⭐️ GrammarLLM — Grammar Constrained Natural Language Generation

**GrammarLLM** is a powerful Python library for **grammar-constrained text generation**, built on top of pre-trained Transformer models.
It allows you to define and apply constraints via formal grammars, ideal for classification, vocabulary restriction, and structured generation.

---

## 🚀 Features

* ✅ **Grammar-constrained generation** — define your own production rules
* 🤗 **Compatible with Hugging Face Transformers**
* ⚡️ **Linear-time decoding via deterministic PDA** — efficient grammar-constrained generation
* 🔦 **Beam Search Support** — stateless PDA re-simulation makes beam reordering safe
* 🎲 **Sampling, batching, multiple return sequences** — all `model.generate()` modes
* 📊 **Constraint-impact analysis** — per-step preserved probability mass, entropy, plots
* 🧬 **Pydantic → grammar conversion** — derive a strict-JSON grammar from a `BaseModel`

📚 **Full documentation: [docs/usage.md](docs/usage.md)** — runnable scripts in [examples/](examples/)

---

## 📦 Requirements

* Python ≥ 3.10
* 🤗 Transformers ≥ 4.30.0
* PyTorch **or** TensorFlow
* A pre-trained causal language model (e.g., GPT-2, LLaMA)

---

## ⚙️ Installation

Run the following commands to clone the repository and install dependencies using **uv** (recommended) or pip:

```bash
git clone https://github.com/misaelmongiovi/grammarllm.git
cd grammarllm
```

Using uv:

```bash
uv sync
uv run python examples/classification.py
```

---

## ⚡️ Quick Start

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

Generation modes (any extra kwarg is forwarded to `model.generate()`):

```python
generate_text(..., num_beams=4)                            # beam search
generate_text(..., num_beams=4, num_return_sequences=3)    # top-3 beams, sorted by probability
generate_text(..., do_sample=True, top_p=0.9, temperature=0.7)
generate_text(model, tokenizer, [prompt1, prompt2], ...)   # batch of prompts
generate_text(..., output_scores=True)                     # + per-step pre/post-masking logits
```

See **[docs/usage.md](docs/usage.md)** for the full API, result formats, and troubleshooting.

---

## 🔍 Examples

Runnable scripts (small model, CPU-friendly) in [examples/](examples/):

| Script | Shows |
|---|---|
| [`examples/classification.py`](examples/classification.py) | hierarchical classification with chat template + few-shot prompt |
| [`examples/beam_search_analysis.py`](examples/beam_search_analysis.py) | beam search, multiple sequences, PDA stack history, preserved-mass analysis + plot |
| [`examples/regex_terminals.py`](examples/regex_terminals.py) | open token classes (words, numbers) via `regex_dict` |

### 📐 Structured Generation (RDF triples)

A larger grammar mixing exact strings, non-terminals, and regex terminals:

```python
    productions = {
        'S*': ["SUBJ PRED OBJ . S*"],
        'SUBJ': ["IRI", "BLANKNODE"],
        'PRED': ["IRI"],
        'OBJ': ["IRI", "BLANKNODE", "LITERAL"],
        'IRI': ["< URI >"],
        'BLANKNODE': ["<<_:>> NAME"],
        'LITERAL': ["\" STRING \" DESCRIPTION_LANG"],
        'DESCRIPTION_LANG': ["^^ IRI", "@ LANGTAG", "ε"],

        'URI': [
            # People
            "<<http://example.org/people/MarioRossi>>",
            "<<http://example.org/people/LuisaVerdi>>",
            "<<http://example.org/people/GiovanniBianchi>>",

            # Properties
            "<<http://example.org/properties/hasAge>>",
            "<<http://example.org/properties/hasProfession>>",
            "<<http://example.org/properties/hasSalary>>",

            # Other types or datatypes
            "<<http://www.w3.org/2001/XMLSchema#decimal>>",
            "<<http://www.w3.org/2001/XMLSchema#integer>>",
            "<<http://www.w3.org/2001/XMLSchema#string>>"
        ],

        'STRING':["alfanum STRING", "ε"],
        'NAME': ["ids NAME_C"],
        'NAME_C': ["idc NAME_C", "ε"],

        'LANGTAG': ['<<it >>', '<<en >>', '<<fr >>', '<<sp >>']
    }

    regex_alfanum = re.compile(r"[a-zA-Z0-9]+")  # es. "abc123"
    regex_right_round_bracket = re.compile(r"\)$")  # match only ')'
    regex_left_round_bracket = re.compile(r"\($")  # match only '('
    regex_less_than = re.compile(r"^<$") # match only '<'
    regex_greater_than = re.compile(r"^>$") # match only '>'
    regex_double_quote = re.compile(r'^\"$') # match only '"'
    regex_datatype = re.compile(r"^\^\^$")   # match only '^^'
    regex_langtag = re.compile(r"^@$")       # match only '@'
    regex_dot = re.compile(r"^\.$")  # match only '.'

    # Starting identifier: must start with a letter or an underscore
    regex_ids = re.compile(r'[A-Za-z_][A-Za-z0-9_-]*')
    # Continuation identifier: cannot start with a letter or an underscore
    regex_idc = re.compile(r'(?![A-Za-z_])[0-9_-][A-Za-z0-9_-]*')


    regex_dict = {
        'regex_alfanum': regex_alfanum,
        'regex_)': regex_right_round_bracket,
        'regex_(': regex_left_round_bracket,
        'regex_<': regex_less_than,
        'regex_>': regex_greater_than,
        'regex_"': regex_double_quote,
        'regex_^^': regex_datatype,
        'regex_@': regex_langtag,
        'regex_.': regex_dot,

        'regex_ids':regex_ids,
        'regex_idc':regex_idc

        }


    # prompt = create_prompt(prompt_input=..., system_prompt=..., examples=...)
    # (few-shot RDF prompt omitted for brevity — full version in main.py)

    pars_table, map_tt = get_parsing_table_and_map_tt(
        tokenizer, productions=productions, regex_dict=regex_dict,
    )
    pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)
    result = generate_text(model, tokenizer, prompt, pdas, streamer, chat_template)
    print(result["text"])
    # <http://example.org/people/GiovanniBianchi><http://example.org/properties/hasAge>"30"^^<http://www.w3.org/2001/XMLSchema#integer>.
```

### 🧬 Pydantic → Grammar (experimental)

```python
from typing import Literal, Optional
from pydantic import BaseModel
from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions

class Person(BaseModel):
    name: Literal["mario", "luisa"]
    mood: Optional[Literal["happy", "sad"]] = None

productions, regex_dict = pydantic_to_productions(Person)
pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions, regex_dict)
# generated text is always valid JSON: {"name": "mario", "mood": null}
```

Output is strict JSON — `json.loads` and `Person.model_validate_json` always
succeed. Design: [docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md](docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md).

---

## 🛠 LL(prefix) Grammar Setup

GrammarLLm enforces syntactic correctness in linear time while maintaining expressiveness in grammar rule
definition. To achieve this, we propose LL(prefix) a novel formalization that generalizes the LL(1) class of [CFG](https://en.wikipedia.org/wiki/Context-free_grammar) enabling the user to define grammars without delving into the details of LLM subword tokenization.

### ✍️ Notation

* Use `<<some string>>` to **generate exact strings** the system handles tokenization
* Symbols **without** `<<>>` are **terminals**, which will be mapped into a set of tokens via regex
* **Uppercase** symbols (e.g., `A`,`B`,`C`) are **non-terminals**
* **Uppercase** `S*` symbol is the start symbol.
* Use `'ε'` for epsilon (empty) transitions
* In JSON format, square brackets `[]` represent a production rule, and commas inside the brackets (e.g., `[A, B, C]`) serve the same purpose as the `|` (OR) operator in formal language notation.

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

Important:
Each key in regex_dict must follow the format 'regex_' + symbol_name, where symbol_name exactly matches the corresponding terminal symbol used in your grammar production rules.

---

## ⚠️ Limitations

* Grammars must be **LL(1) after tag expansion** — non-LL(1) grammars are rejected at setup time with a diagnostic `Conflict:` error
* No left recursion (`'A': ["A x"]`) — use right recursion (`'A': ["x A", "ε"]`)
* The streamer (live token logging) is disabled with beam search (HF limitation); generation itself fully supports beams
* Pydantic conversion emits strict JSON; no escape sequences in string content (no ", \ or control chars)

---

## 🤝 Contributing

To contribute, please follow these guidelines:

**Bug Reports**: If you find a bug, please open an issue and provide detailed steps to reproduce it.

**Feature Requests**: For new features or enhancements, please open an issue first to discuss the idea before starting development.

- Code Contributions:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Write your code and ensure it's well-documented and tested.
- Commit your changes using descriptive commit messages.
- Submit a pull request from your branch to the main branch, clearly explaining the purpose of your changes.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🖌 Citation

[![ACL 2025 Paper](https://img.shields.io/badge/ACL%202025-Paper-blue)](https://aclanthology.org/2025.findings-acl.177/)

If you use this work, please cite:

```bibtex
@inproceedings{tuccio2025grammarllm,
  title     = {GRAMMAR-LLM: Grammar-Constrained Natural Language Generation},
  author    = {Tuccio, Gabriele and Bulla, Luana and Madonia, Maria and Gangemi, Aldo and Mongiovì, Misael},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
  pages     = {3412--3422},
  address   = {Vienna, Austria},
  url       = {https://aclanthology.org/2025.findings-acl.177/}
}
```

---

## 📫 Contact

📧 Email:
[gabriele.tuccio@phd.unict.it](mailto:gabriele.tuccio@phd.unict.it)
[luana.bulla@phd.unict.it](mailto:luana.bulla@phd.unict.it)
[misael.mongiovi@unict.it](mailto:misael.mongiovi@unict.it)
