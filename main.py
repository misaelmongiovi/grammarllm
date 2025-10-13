from grammarllm.scripts.grammar_generation import ProductionRuleProcessor
from grammarllm.scripts.map_terminal_tokens import generate_token_maps
from grammarllm.scripts.generate_LL1_parsing_table import parsing_table

from grammarllm.modules.BaseStreamer import BaseStreamer
from grammarllm.modules.PushdownAutomaton import PushdownAutomaton
from grammarllm.modules.SimpleLogitProcessor import MaskLogitsProcessor

import logging
import os
import re

#from grammarllm.utils.common_regex import regex_dict
#from grammarllm.utils.examples import *
#from grammarllm.utils.gloss_class import classes
from grammarllm.utils.toolbox import create_prompt, chat_template 

from transformers import AutoTokenizer, AutoModelForCausalLM



def get_parsing_table_and_map_tt(tokenizer, productions, regex_dict=None):

    processor = ProductionRuleProcessor(tokenizer=tokenizer)
    # Process the grammar productions
    final_grammar, tag_mapping = processor.process_full_grammar(productions)

    #add eos token to the grammar
    final_grammar[('S*','RULE')].append([tokenizer.eos_token])
    # Generate parsing table
    pars_tab = parsing_table(final_grammar)

    # Generate token maps
    if regex_dict:
        map_terminal_tokens = generate_token_maps(tokenizer, pars_tab, regex_dict)
    else:
        map_terminal_tokens = generate_token_maps(tokenizer, pars_tab)

    # uncomment the following lines to log the parsing table and terminal token mappings
    # logging.info("\nMap Terminal Tokens:\n")
    # for key, values in map_terminal_tokens.items():
    #     logging.info(f"{key} -> {values}")

    return pars_tab, map_terminal_tokens

def generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens):
    # Create Pushdown Automaton and initialize processors and streamer
    pda = PushdownAutomaton(grammar=pars_tab, startSymbol='S*', map=map_terminal_tokens)
    return MaskLogitsProcessor(tokenizer, pda), BaseStreamer(tokenizer, pda)

def setup_logging():
    """Setup logging configuration."""
    log_dir = 'grammarllm/temp'
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    
    logging.basicConfig(
        filename=os.path.join(log_dir, 'GRAM-GEN.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w+'  # Overwrites the file every time
    )

def generate_text(model, tokenizer, text, logit_processor, streamer, chat_template = None, max_new_tokens=400, do_sample=False, top_p=None, **kwargs):
    """
    Generate text using the provided model and tokenizer with grammar constraints.

    Args:
        model: pre-trained model.
        tokenizer: model tokenizer.
        text: input text or list of messages (if chat_template is used).
        logit_processor: processor parameter
        streamer: Streamer parameter
        max_new_tokens: maximum number of new tokens to generate.
        do_sample: if True, enables sampling; otherwise, uses greedy decoding.
        top_p: nucleus sampling parameter (used if do_sample is True).
        **kwargs: additional generation parameters.
    """
    
    try:
        # TO USE WHEN CREATE PROMPT IS USED AND PROMPT IS A LIST
        if isinstance(text,list):
            if chat_template is None:
                raise ValueError("Chat template must be specified")
            tokenizer.chat_template = chat_template
            tokenized_input = tokenizer.apply_chat_template(text, 
                                                        tokenize=True,
                                                        add_generation_prompt=True,
                                                        return_dict=True,
                                                        return_tensors="pt").to(model.device)
        else:
            tokenized_input = tokenizer(text, return_tensors="pt")

        # Safe defaults
        kwargs.setdefault("num_beams", 1)  # beam search disabled by default
        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        # num_beams safety
        if kwargs["num_beams"] != 1:
            logging.warning("⚠️ num_beams > 1 is not compatible with grammar-constrained generation. Automatically set to num_beams=1.")
            kwargs["num_beams"] = 1


        # Sampling parameters
        if do_sample:
            if top_p is not None:
                kwargs["top_p"] = top_p
        else:
            # Rimuovi parametri di sampling se presenti
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)

        # Device compatibility
        device = model.device
        input_ids = tokenized_input["input_ids"].to(device)
        if input_ids.device != model.device:

            logging.warning(f"Error: 'input_ids' are on device {input_ids.device}, while the model is on device {model.device}. Moving 'input_ids' to the same device as the model.")
            
        attention_mask = tokenized_input["attention_mask"].to(device)
        if attention_mask.device != model.device:
            logging.warning(f"Error: 'attention_mask' is on device {attention_mask.device}, while the model is on device {model.device}. Moving 'attention_mask' to the same device as the model.")


        start = input_ids.shape[1]

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            logits_processor=[logit_processor],
            **kwargs
        )

        answer = tokenizer.decode(output[0][start:], skip_special_tokens=True)

        return answer

    except Exception as e:
        raise RuntimeError(f"Errore nella generazione del testo: {e}")
    
def main():
    setup_logging()
    
    ######## HIERARCHICAL CLASSIFICATION EXAMPLE ##########
    # Define grammar productions
    productions = { 'S*': ["<<positive >> A", "<<negative >> B", "<<neutral >> C"],
                    'A': ["<<happy>>", "<<peaceful>>", "<<joyful>>"],
                    'B': ['<<sad>>', '<<angry>>', '<<frustrated>>'],
                    'C': ['<<calm>>', '<<indifferent>>', '<<unemotional>>']
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
    prompt=create_prompt(
        prompt_input="It's raining and I feel a bit down.",
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
        #regex_dict=regex_dict, # uncomment this line to use regex for terminal token mapping, unneeded for HIERARCHICAL CLASSIFICATION EXAMPLE
    )
    # Generate LogitProcessor and Streamer
    LogitProcessor, Streamer = generate_grammar_parameters(tokenizer, pars_table, map_terminal_tokens)
    
    # Set temperature for LogitProcessor
    LogitProcessor.temperature = 1.0 
    output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template, do_sample=True, top_k=10)
    print(output) # Example output: "negative sad"

if __name__ == "__main__":
    main()
