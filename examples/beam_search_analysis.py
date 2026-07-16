"""
Beam search with multiple returned sequences + constraint-impact analysis.

Shows:
  - num_beams / num_return_sequences usage
  - the result-dict format (probabilities, per-step PDA stack history)
  - preserved-mass analysis of how much the grammar forced the model

Run:  uv run python examples/beam_search_analysis.py
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
)
from grammarllm.utils.generation_analysis import (
    compute_generation_analysis,
    print_analysis_summary,
    plot_generation_analysis,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    setup_logging()

    productions = {
        'S*': ["<<positive>> A", "<<negative>> B"],
        'A':  ["<< happy>>", "<< peaceful>>"],
        'B':  ["<< gloomy>>", "<< angry>>"],
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
    pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)

    prompt = "Classify the sentiment: 'I love sunny days.' Answer:"

    results = generate_text(
        model, tokenizer, prompt, pdas, streamer,
        max_new_tokens=8,
        num_beams=3,
        num_return_sequences=3,
        output_scores=True,        # needed for the analysis below
    )

    # results is a list of dicts, sorted by probability (descending)
    for i, r in enumerate(results):
        print(f"beam {i}: {r['text']!r:<22} p={r['probability']:.4f}")
        for step, stack in enumerate(r["pda_history"]):
            print(f"    stack after token {step}: {stack}")

    # Constraint impact for the best sequence: preserved_mass ~ 1.0 means the
    # model already agreed with the grammar, ~ 0.0 means it was forced.
    analysis = compute_generation_analysis(results[0], tokenizer, label="best beam")
    print_analysis_summary(analysis)

    fig = plot_generation_analysis(analysis, title="Grammar constraint impact")
    fig.savefig("analysis.png")
    print("saved plot -> analysis.png")


if __name__ == "__main__":
    main()
