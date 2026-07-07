"""
Open token classes via regex terminals: constrain output to
"<word> is <number>" repeated one or more times.

Bare lowercase symbols in productions (word, number) are mapped to
vocabulary tokens by the regexes in regex_dict; the key must be
'regex_' + symbol name.

Run:  uv run python examples/regex_terminals.py
"""
import re

from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    setup_logging()

    productions = {
        'S*':   ["word REST"],
        'REST': ["<< is>> NUM"],
        'NUM':  ["number"],
    }
    regex_dict = {
        'regex_word':   re.compile(r"^[A-Z][a-z]+$"),   # capitalized word tokens
        'regex_number': re.compile(r"^\d+$"),            # pure digit tokens
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    pars_table, map_tt = get_parsing_table_and_map_tt(
        tokenizer, productions, regex_dict=regex_dict
    )
    pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)

    result = generate_text(
        model, tokenizer,
        "State a person's age. Example: Marco is 30. Answer:",
        pdas, streamer,
        max_new_tokens=8,
    )
    print(f"text  : {result['text']!r}")
    print(f"stack : {result['pda_stack']}")


if __name__ == "__main__":
    main()
