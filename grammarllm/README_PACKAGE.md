# ⭐️ GrammarLLM — Grammar Constrained Natural Language Generation

**GrammarLLM** is a powerful Python library for **grammar-constrained text generation**, built on top of pre-trained Transformer models.
It allows you to define and apply constraints via formal grammars, ideal for classification, vocabulary restriction, and structured generation.

---

## 🚀 Features

* ✅ **Grammar-constrained generation** — define your own production rules
* 🤗 **Compatible with Hugging Face Transformers**
* ⚡️ **Linear-time decoding via deterministic PDA** — efficient grammar-constrained generation

---

## 📦 Requirements

* Python ≥ 3.10
* 🤗 Transformers ≥ 4.30.0
* PyTorch **or** TensorFlow
* A pre-trained causal language model (e.g., GPT-2, LLaMA)

---

## ⚙️ Installation
The package is currently available on Test PyPI. To install the library, use the following command:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple grammarllm
```


---

## 📄 Core Components Documentation

### `get_parsing_table_and_map_tt()`

This function generates the essential components needed by `generate_grammar_parameters()`. It takes your grammar definition and optional regular expressions to prepare for constrained text generation.

**Arguments:**

* `tokenizer`: A Hugging Face tokenizer instance.
* `productions`: Your grammar defined in JSON format. This is a **required** argument; if not provided, the function will raise an exception.
* `regex_dict=None`: An optional dictionary containing regular expressions. Use this if your grammar's terminals need to be mapped to a set of tokens via regex.

**Returns:**

* `pars_tab`: The generated parsing table.
* `map_terminal_tokens`: A mapping between terminals or tokens and their corresponding `token_id` sets. Terminals can map to multiple tokens using regular expressions, while individual tokens map to a single ID.

Both `pars_tab` and `map_terminal_tokens` should be passed to `generate_grammar_parameters()`.

---

### `generate_grammar_parameters()`

This function creates the two critical parameters, a `LogitProcessor` and a `Streamer`, that enable grammar-constrained generation. The `LogitProcessor` dynamically masks tokens that don't adhere to your grammar, while the `Streamer` updates the internal pushdown automaton as new tokens are generated.

**Arguments:**

* `tokenizer`: A Hugging Face tokenizer instance.
* `pars_tab`: The parsing table returned by `get_parsing_table_and_map_tt()`.
* `map_terminal_tokens`: The terminal-to-token mapping returned by `get_parsing_table_and_map_tt()`.

**Returns:**

* `LogitProcessor`: An object responsible for dynamic token masking based on the grammar.
* `Streamer`: An object used to update the pushdown automaton during token generation.

Both the `LogitProcessor` and `Streamer` should be passed to `generate_text()`.

---

### `generate_text()`

This function generates text that adheres to your specified grammar constraints.

**Main Arguments:**

* `model`: The language model to use for generation.
* `tokenizer`: The tokenizer associated with the model.
* `text`: The initial prompt or input text.
* `logit_processor`: The `LogitProcessor` object obtained from `generate_grammar_parameters()`.
* `streamer`: The `Streamer` object obtained from `generate_grammar_parameters()`.

**Additional Options:**

* `chat_template`: Pass this argument if you need to provide examples and want to format the full prompt appropriately for a chat model.
* Other options to control generation length, sampling strategies, and overall behavior.

----
## 🔍 Use Cases

### 1. 🔮 Classification

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from grammarllm import (
    generate_grammar_parameters,
    generate_text,
    get_parsing_table_and_map_tt,
    setup_logging,
    create_prompt,
    chat_template,
)

setup_logging()

productions = { 'S*': ["<<positive >> A", "<<negative >> B", "<<neutral >> C"],
                'A': ["<<happy>>", "<<peaceful>>", "<<joyful>>"],
                'B': ['<<sad>>', '<<angry>>', '<<frustrated>>'],
                'C': ['<<calm>>', '<<indifferent>>', '<<unemotional>>']
                }

system_prompt = """You are a hierarchical classification assistant. Your task is to classify the user input 
                    into one of the following hierarchical categories as shown in the followig examples\n\n"""

examples = [
    {"role": "user", "content": "I just got a promotion!"},
    {"role": "assistant", "content": "positive joyful"},

    {"role": "user", "content": "Nothing ever goes my way."},
    {"role": "assistant", "content": "negative frustrated"},

    {"role": "user", "content": "The lake was still and quiet."},
    {"role": "assistant", "content": "neutral calm"},

    {"role": "user", "content": "I miss my family so much."},
    {"role": "assistant", "content": "negative sad"}
]

prompt=create_prompt(
    prompt_input="It's raining and I feel a bit down.",
    system_prompt=system_prompt,
    examples=examples
)


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(tokenizer, productions)

LogitProcessor, Streamer = generate_grammar_parameters(
    tokenizer, pars_table, map_terminal_tokens
)

output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template)
print(output)  # → "negative sad"
```

---

### 2. 🧩 Vocabulary Restriction

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from grammarllm import (
    generate_grammar_parameters,
    generate_text,
    get_parsing_table_and_map_tt,
    setup_logging,
    create_prompt,
    chat_template,
)

setup_logging()

productions = {
'S*': [
    "<< Yes>> S*",
    "<< I'm>> S*",
    "<< very>> S*",
    "<< happy>> S*",
    "<< !>> S*",
    "<< so>> S*",
    "<< really>> S*",
    "<< excited>> S*",
    "<< today>> S*",
    "<< thanks>> S*",
    "<< you>> S*",
    "<< much>> S*",
    "<< great>> S*",
    "<< good>> S*",
    "<< fine>> S*",
    "<< amazing>> S*",
    ]
}

system_prompt = """You are a text generation assistant. When generating responses, you must use only the words that
                    appear in the provided examples below. You should not introduce any new words outside of those examples."""


examples = [
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm very happy"},

    {"role": "user", "content": "Is everything okay?"},
    {"role": "assistant", "content": "Yes, I'm so excited!"},

    {"role": "user", "content": "How was your day?"},
    {"role": "assistant", "content": "I'm really happy today!"},

    {"role": "user", "content": "Do you feel good?"},
    {"role": "assistant", "content": "Yes, I feel great, thanks!"},

    {"role": "user", "content": "What's up?"},
    {"role": "assistant", "content": "I'm fine, thank you so much!"},

    {"role": "user", "content": "Anything special today?"},
    {"role": "assistant", "content": "I'm very excited and happy today!"}
    ]

prompt=create_prompt(
    prompt_input="Say something of positive:",
    system_prompt=system_prompt,
    examples=examples
    )

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(tokenizer, productions)

LogitProcessor, Streamer = generate_grammar_parameters(
    tokenizer, pars_table, map_terminal_tokens
)

output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template)
print(output)  # → "I'm happy"
```

---

### 3. 📐 Structured Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from grammarllm import (
    generate_grammar_parameters,
    generate_text,
    get_parsing_table_and_map_tt,
    setup_logging,
    create_prompt,
    chat_template,
)


setup_logging()
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


system_prompt = """You are an assistant that converts natural language sentences into RDF triples syntax.

Follow these rules:

1. Use URIs (`<...>`) for:
- Identifiable entities such as people, properties, or concepts.
- Example:
    <http://example.org/people/MarioRossi> <http://example.org/properties/hasFriend> <http://example.org/people/LuisaVerdi> .

2. Use literals (`"..."`) for:
- Plain values such as professions, cities, names, numbers, dates, or booleans.
- Add datatypes (`^^<...>`) or language tags (`@lang`) if needed.
- Examples:
    "engineer"@en  
    "40"^^<http://www.w3.org/2001/XMLSchema#integer>

3. Use blank nodes (`_:`) only if:
- The object is anonymous and has internal structure (i.e., it has its own properties).
- Example:
    <http://example.org/people/MarioRossi> <http://example.org/properties/hasAddress> _:b1 .
    _:b1 <http://example.org/properties/street> "Via Roma" .
    _:b1 <http://example.org/properties/city> "Milano" .

Never use a blank node (`_:`) for simple values like "engineer" or "teacher". Use a literal (`"..."`) instead.

Now use the following examples to generate clean and correct RDF triples from user input."""

examples = [
    {"role": "user", "content": "Mario Rossi is 40 years old."},
    {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasAge> \"40\" ^^<http://www.w3.org/2001/XMLSchema#integer> ."},

    {"role": "user", "content": "Luisa Verdi is an engineer."},
    {"role": "assistant", "content": "<http://example.org/people/LuisaVerdi> <http://example.org/properties/hasProfession> \"engineer\" @en ."},

    {"role": "user", "content": "Giovanni Bianchi earns 55000."},
    {"role": "assistant", "content": "<http://example.org/people/GiovanniBianchi> <http://example.org/properties/hasSalary> \"55000\" ^^<http://www.w3.org/2001/XMLSchema#decimal> ."},

    {"role": "user", "content": "Mario Rossi has an anonymous node as a contact."},
    {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasContact> _:ids ."},

    {"role": "user", "content": "Mario Rossi has the profession of teacher."},
    {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasProfession> \"teacher\" @en ."}
]

prompt=create_prompt(
    prompt_input="Giovanni Bianchi was born 30 years ago.",
    system_prompt=system_prompt,
    examples=examples
)

# Initialize tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Generate grammar parameters
pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
    tokenizer, 
    productions=productions, 
    regex_dict=regex_dict,
)

LogitProcessor, Streamer = generate_grammar_parameters(tokenizer, pars_table, map_terminal_tokens)
output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template)
print(output) # Example output: "<http://example.org/people/GiovanniBianchi><http://example.org/properties/hasAge>"30"^^<http://www.w3.org/2001/XMLSchema#integer>."
  
```

---

## 🛠 LL(prefix) Grammar Setup

GrammarLLm enforces syntactic correctness in linear time while maintaining expressiveness in grammar rule
definition. To achieve this, we propose LL(prefix) a novel formalization that generalizes the LL(1) class of [CFG](https://en.wikipedia.org/wiki/Context-free_grammar) enabling the user to define grammars without delving into the details of LLM subword tokenization.

### ✍️ Notation

* Use `<<some string>>` to **generate exact strings** the system handles tokenization
* Symbols **without** `<<>>` are **terminals**, which will be mapped into a set of tokens via regex
* **Uppercase** symbols (e.g., `A`,`B`,`C`) are **non-terminals**
* **Uppercase** `S*` symbol is the start symbol.
* Use `ε` for epsilon (empty) transitions
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

* ❌ Beam search is **not supported**
* You cannot define multiple <<exact_string>> in the same rule

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📫 Contact

📧 Email:  
[gabriele.tuccio@phd.unict.it](mailto:gabriele.tuccio@phd.unict.it)
[luana.bulla@phd.unict.it](mailto:luana.bulla@phd.unict.it)
[misael.mongiovi@unict.it](mailto:misael.mongiovi@unict.it)



