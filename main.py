import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm.utils.toolbox import create_prompt, chat_template
from grammarllm.utils.generation_analysis import (
    compute_generation_analysis, 
    plot_generation_analysis,
    print_analysis_summary,  
    compare_analyses         
)
from grammarllm.generate_with_constraints import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    setup_logging,
    generate_text
)


    
def main():
    setup_logging()
    
    ######## HIERARCHICAL CLASSIFICATION EXAMPLE ##########
    # Define grammar productions

    productions = { 'S*': ["<<positive>> A", "<<negative>> B", "<<neutral>> C"],
                    'A': ["<< happy>>", "<< peaceful>>", "<< joyful>>"],
                    'B': ['<< gloomy>>', '<< angry>>', '<< frustrated>>'],
                    'C': ['<< calm>>', '<< indifferent>>', '<< unemotional>>']
                }
    # Define system prompt and examples
    system_prompt = """You are a hierarchical classification assistant. Your task is to classify the user input 
                        into one of the following hierarchical categories as shown in the followig examples\n\n 
                        # 1-level categories
                        categories: positive, negative, neutral
                        # 2-level categories
                        positive: happy, peaceful, joyful
                        negative: gloomy, angry, frustrated
                        neutral: calm, indifferent, unemotional"""
                        

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
    # Create prompt
    # BATCH PROMPTING EXAMPLE
    prompt_inputs = [
        "It's raining and I feel a bit down.",
        #"The sun is shining and I am winning."
    ]
    # Define the Grammar Productions
#     productions = {
#             'S*': ["SUBJ PRED OBJ . S*"],
#             'SUBJ': ["IRI", "BLANKNODE"],
#             'PRED': ["IRI"],
#             'OBJ': ["IRI", "BLANKNODE", "LITERAL"],
#             'IRI': ["< URI >"],
#             'BLANKNODE': ["<<_:>> NAME"],
#             'LITERAL': ["\" STRING \" DESCRIPTION_LANG"],
#             'DESCRIPTION_LANG': ["^^ IRI", "@ LANGTAG", "ε"],

#             'URI': [
#                 # People
#                 "<<http://example.org/people/MarioRossi>>",
#                 "<<http://example.org/people/LuisaVerdi>>",
#                 "<<http://example.org/people/GiovanniBianchi>>",

#                 # Properties
#                 "<<http://example.org/properties/hasAge>>",
#                 "<<http://example.org/properties/hasProfession>>",
#                 "<<http://example.org/properties/hasSalary>>",

#                 # Other types or datatypes
#                 "<<http://www.w3.org/2001/XMLSchema#decimal>>",
#                 "<<http://www.w3.org/2001/XMLSchema#integer>>",
#                 "<<http://www.w3.org/2001/XMLSchema#string>>"
#             ],

#             'STRING':["alfanum STRING", "ε"],
#             'NAME': ["ids NAME_C"],
#             'NAME_C': ["idc NAME_C", "ε"],

#             'LANGTAG': ['<<it >>', '<<en >>', '<<fr >>', '<<sp >>']
#     }
#     # Define regular expressions for terminal tokens
#     regex_alfanum = re.compile(r"[a-zA-Z0-9]+")  # es. "abc123"
#     regex_right_round_bracket = re.compile(r"\)$")  # match only ')'
#     regex_left_round_bracket = re.compile(r"\($")  # match only '('
#     regex_less_than = re.compile(r"^<$") # match only '<'
#     regex_greater_than = re.compile(r"^>$") # match only '>'
#     regex_double_quote = re.compile(r'^\"$') # match only '"'
#     regex_datatype = re.compile(r"^\^\^$")   # match only '^^'
#     regex_langtag = re.compile(r"^@$")       # match only '@'
#     regex_dot = re.compile(r"^\.$")  # match only '.'

#     # Starting identifier: must start with a letter or an underscore
#     regex_ids = re.compile(r'[A-Za-z_][A-Za-z0-9_-]*')
#     # Continuation identifier: cannot start with a letter or an underscore
#     regex_idc = re.compile(r'(?![A-Za-z_])[0-9_-][A-Za-z0-9_-]*')


#     regex_dict = {
#         'regex_alfanum': regex_alfanum,
#         'regex_)': regex_right_round_bracket,
#         'regex_(': regex_left_round_bracket,
#         'regex_<': regex_less_than,
#         'regex_>': regex_greater_than,
#         'regex_"': regex_double_quote,
#         'regex_^^': regex_datatype,
#         'regex_@': regex_langtag,
#         'regex_.': regex_dot,

#         'regex_ids':regex_ids,
#         'regex_idc':regex_idc

#         }

#     # Define the system prompt and examples for the classification task
#     system_prompt = """You are an assistant that converts natural language sentences into RDF triples syntax.

#     Follow these rules:

#     1. Use URIs (`<...>`) for:
#     - Identifiable entities such as people, properties, or concepts.
#     - Example:
#         <http://example.org/people/MarioRossi> <http://example.org/properties/hasFriend> <http://example.org/people/LuisaVerdi> .

#     2. Use literals (`"..."`) for:
#     - Plain values such as professions, cities, names, numbers, dates, or booleans.
#     - Add datatypes (`^^<...>`) or language tags (`@lang`) if needed.
#     - Examples:
#         "engineer"@en  
#         "40"^^<http://www.w3.org/2001/XMLSchema#integer>

#     3. Use blank nodes (`_:`) only if:
#     - The object is anonymous and has internal structure (i.e., it has its own properties).
#     - Example:
#         <http://example.org/people/MarioRossi> <http://example.org/properties/hasAddress> _:b1 .
#         _:b1 <http://example.org/properties/street> "Via Roma" .
#         _:b1 <http://example.org/properties/city> "Milano" .

#     Never use a blank node (`_:`) for simple values like "engineer" or "teacher". Use a literal (`"..."`) instead.

#     Now use the following examples to generate clean and correct RDF triples from user input."""

#     examples = [
#     {"role": "user", "content": "Mario Rossi is 40 years old."},
#     {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasAge> \"40\" ^^<http://www.w3.org/2001/XMLSchema#integer> ."},

#     {"role": "user", "content": "Luisa Verdi is an engineer."},
#     {"role": "assistant", "content": "<http://example.org/people/LuisaVerdi> <http://example.org/properties/hasProfession> \"engineer\" @en ."},

#     {"role": "user", "content": "Giovanni Bianchi earns 55000."},
#     {"role": "assistant", "content": "<http://example.org/people/GiovanniBianchi> <http://example.org/properties/hasSalary> \"55000\" ^^<http://www.w3.org/2001/XMLSchema#decimal> ."},

#     {"role": "user", "content": "Mario Rossi has an anonymous node as a contact."},
#     {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasContact> _:ids ."},

#     {"role": "user", "content": "Mario Rossi has the profession of teacher."},
#     {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasProfession> \"teacher\" @en ."}
# ]
#     # BATCH PROMPTING EXAMPLE
#     prompt_inputs = [
#         "Giovanni Bianchi was born 30 years ago.",
#         #"Giovanni bianchi is a doctor."
#     ]

    prompts = []
    for p in prompt_inputs:
        prompts.append(create_prompt(
            prompt_input=p,
            system_prompt=system_prompt,
            examples=examples
        ))
    
    # Pass LIST of prompts to generate_text
    prompt = prompts


    # Initialize model and tokenizer
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    model.to("mps")



    # Generate grammar parameters
    pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
        tokenizer, 
        productions=productions, 
        #regex_dict=regex_dict, # uncomment this line to use regex for terminal token mapping, unneeded for HIERARCHICAL CLASSIFICATION EXAMPLE
    )

    # Generate LogitProcessor and Streamer
    pdas, Streamer = generate_grammar_parameters(tokenizer, pars_table, map_terminal_tokens, num_return_sequences=2)
    
    output = generate_text(
        model, tokenizer, prompt, pdas, Streamer, chat_template, 
        do_sample=True, 
        num_return_sequences=3, 
        max_new_tokens=10, # Single token generation
        num_beams=3,  # Enable Beam Search
        temperature=1.0,
        output_scores=True, # Verify stack is still available even if scores are disabled
    )
    for i, out in enumerate(output):
        print(f"\n--- Result {i} ---")
        if isinstance(out, dict):
            print(f"Generated text: {out['text']}")
            if out.get('pda_history'):
                print("PDA Stack History (step by step):")
                for step, stack in enumerate(out['pda_history']):
                    print(f"  Stack {step+1}: {stack}")
            else:
                print(f"Final PDA Stack: {out.get('pda_stack')}")
        else:
            print(out)
        

    # # Prerequisito: generate_text() con output_scores=True
    # result = generate_text(
    #     model, tokenizer, prompt, pdas, streamer,
    #     output_scores=True,        # ← indispensabile
    #     max_new_tokens=10,
    #     num_beams=4,
    # )



    analysis = compute_generation_analysis(output[0], tokenizer, label="Impatto vincolo grammaticale")
    print_analysis_summary(analysis)
    fig = plot_generation_analysis(analysis, title="Impatto vincolo grammaticale")
    fig.savefig("analysis.png")

    # # Confronto tra due configurazioni
    # analysis_generic  = compute_generation_analysis(result_generic,  tokenizer, label="Prompt generico")
    # analysis_specific = compute_generation_analysis(result_specific, tokenizer, label="Prompt specifico")

    # fig = compare_analyses(
    #     [analysis_generic, analysis_specific],
    #     metric="preserved_mass"
    # )


if __name__ == "__main__":
    main()

