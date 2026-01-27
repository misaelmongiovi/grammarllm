import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm.utils.toolbox import create_prompt, chat_template 
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
    # Create prompt
    # BATCH PROMPTING EXAMPLE
    prompt_inputs = [
        "It's raining and I feel a bit down.",
        "The sun is shining and I am winning."
    ]
    
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
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


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
        do_sample=False, 
        #num_return_sequences=1, 
        max_new_tokens=2,
        num_beams=2,  # Enable Beam Search
        temperature=1.2,
        output_scores=False,
    )
    for out in output:
        print(out)
        



if __name__ == "__main__":
    main()

