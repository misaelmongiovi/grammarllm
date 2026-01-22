from grammarllm.generate_with_constraints import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    setup_logging,
    generate_text
)

import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm.utils.toolbox import create_prompt, chat_template 



def plot_invalid_trajectory(points):
    """
    Plotta la traiettoria dei punti raccolti: 
    x = entropia normalizzata, y = massa cumulativa dei token invalidi.
    Aggiunge anche il centroide medio di tutti i punti con marker verde,
    e mostra la distanza euclidea dal centroide all'origine con linea tratteggiata.
    """
    if not points:
        print("Nessun punto da plottare")
        return

    x = [pt[0] for pt in points]
    y = [pt[1] for pt in points]

    # Calcola centroide
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    # Calcola distanza euclidea dall'origine
    centroid_distance = np.sqrt(centroid_x**2 + centroid_y**2)

    plt.figure(figsize=(8, 6))
    
    # Linea tratteggiata che collega i punti in ordine
    plt.plot(x, y, '--', color='blue', linewidth=1, alpha=0.7, label='Traiettoria')

    # Punti rossi
    plt.scatter(x, y, color='red', s=40, label='Punti generazione')

    # Numeri progressivi
    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        plt.text(xi + 0.01, yi + 0.01, str(i), fontsize=9, color='black')

    # Centroide verde
    plt.scatter(centroid_x, centroid_y, color='green', s=100, marker='o', label='Centroide')

    # Linea tratteggiata dal centroide all'origine
    plt.plot([0, centroid_x], [0, centroid_y], '--', color='black', linewidth=1.2, alpha=0.7)
    # Testo della distanza sopra la linea
    mid_x = centroid_x / 2
    mid_y = centroid_y / 2
    plt.text(mid_x, mid_y + 0.02, f"{centroid_distance:.4f}", fontsize=10, color='black', fontweight='bold')

    plt.xlabel("Normalized invalid entropy")
    plt.ylabel("Cumulative invalid mass")
    plt.title("Traiettoria: Entropy vs Invalid Mass con Centroide")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()



    
def main():
    setup_logging()
    
    ######## HIERARCHICAL CLASSIFICATION EXAMPLE ##########
    # Define grammar productions
    # productions = { 'S*': ["<<positive >> A", "<<negative >> B", "<<neutral >> C"],
    #                 'A': ["<<happy>>", "<<peaceful>>", "<<joyful>>"],
    #                 'B': ['<<sad>>', '<<angry>>', '<<frustrated>>'],
    #                 'C': ['<<calm>>', '<<indifferent>>', '<<unemotional>>']
    #               }
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
    for p_in in prompt_inputs:
        prompts.append(create_prompt(
            prompt_input=p_in,
            system_prompt=system_prompt,
            examples=examples
        ))
    
    # Pass LIST of prompts to generate_text
    prompt = prompts


    # Initialize tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    #model = AutoModelForCausalLM.from_pretrained("gpt2")
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Generate grammar parameters
    pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
        tokenizer, 
        productions=productions, 
        #regex_dict=regex_dict, # uncomment this line to use regex for terminal token mapping, unneeded for HIERARCHICAL CLASSIFICATION EXAMPLE
    )
    # Generate LogitProcessor and Streamer
    num_return_sequences = 3
    LogitProcessor, Streamer = generate_grammar_parameters(tokenizer, pars_table, map_terminal_tokens, num_return_sequences=num_return_sequences)
    
    # Set temperature for LogitProcessor
    LogitProcessor.temperature = 3.0 
    #output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template, do_sample=False)
    output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template, do_sample=True, num_return_sequences=num_return_sequences, max_new_tokens=2)
    print(output) # Example output: "negative sad"

    #print(f"Passaggi generati: {len(LogitProcessor.original_scores_history)}")
    # Primo step dei logit originali
    if len(LogitProcessor.original_scores_history) > 0:
        primi_logit = LogitProcessor.original_scores_history[0]
        print(primi_logit)
    #Plotta la traiettoria usando i punti raccolti
    #plot_invalid_trajectory(LogitProcessor.points)
    #a = LogitProcessor.preserved_mass
    #print(a)#[0.9560056328773499, 0.046188708394765854, 0.0]


if __name__ == "__main__":
    main()
