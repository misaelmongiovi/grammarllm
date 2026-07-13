"""
Hierarchical sentiment classification with a grammar-constrained model.

The grammar forces the output to be exactly one level-1 label followed by
one matching level-2 label, e.g. "positive happy".

Run:  uv run python examples/classification.py
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
    create_prompt,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    setup_logging()

    productions = {
        'S*': ["<<positive>> A", "<<negative>> B", "<<neutral>> C"],
        'A':  ["<< happy>>", "<< peaceful>>", "<< joyful>>"],
        'B':  ["<< gloomy>>", "<< angry>>", "<< frustrated>>"],
        'C':  ["<< calm>>", "<< indifferent>>", "<< unemotional>>"],
    }

    system_prompt = (
        "You are a hierarchical classification assistant. Classify the user "
        "input into one level-1 category (positive, negative, neutral) "
        "followed by one matching level-2 category."
    )
    examples = [
        {"role": "user", "content": "I just got a promotion!"},
        {"role": "assistant", "content": "positive joyful"},
        {"role": "user", "content": "Nothing ever goes my way."},
        {"role": "assistant", "content": "negative frustrated"},
        {"role": "user", "content": "The lake was still and quiet."},
        {"role": "assistant", "content": "neutral calm"},
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    # Phase 1 — once per (grammar, tokenizer)
    pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
    # Phase 2 — once per session
    pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)

    prompt = create_prompt(
        prompt_input="It's raining and I feel a bit down.",
        system_prompt=system_prompt,
        examples=examples,
    )

    # Phase 3 — every call
    result = generate_text(
        # no chat template: the conversation is rendered with the model's own
        model, tokenizer, prompt, pdas, streamer,
        max_new_tokens=8,
    )

    print(f"text        : {result['text']!r}")
    print(f"probability : {result['probability']:.4f}")
    print(f"final stack : {result['pda_stack']}   ([] = grammar satisfied)")


if __name__ == "__main__":
    main()
