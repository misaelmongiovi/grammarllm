from .scripts.grammar_generation import ProductionRuleProcessor
from .scripts.map_terminal_tokens import generate_token_maps
from .scripts.generate_LL1_parsing_table import parsing_table

from .modules.BaseStreamer import BaseStreamer
from .modules.PushdownAutomaton import PushdownAutomaton
from .modules.SimpleLogitProcessor_ import MaskLogitsProcessor

import logging
import os
import copy

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



def generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens, num_return_sequences=1):
    # Create Pushdown Automaton based on num_return_sequences
    # We need independent PDA instances for each sequence because they maintain state
    
    pdas = []
    base_pda = PushdownAutomaton(grammar=pars_tab, startSymbol='S*', map=map_terminal_tokens)
    
    # If num_return_sequences is 1, we can just use one.
    # If >1, we need to clone it or create new ones.
    # Cloning is safer if PushdownAutomaton initialization is heavy, but here it's light.
    # We'll just append independent deepcopies or new instances.
    
    pdas.append(base_pda)
    
    for _ in range(num_return_sequences - 1):
        # Deepcopy to ensure independent stack and state
        pdas.append(copy.deepcopy(base_pda))

    # LogitsProcessor and Streamer now accept a LIST of PDAs
    return MaskLogitsProcessor(tokenizer, pdas, return_original_dist=True), BaseStreamer(tokenizer, pdas)

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

def generate_text(model, tokenizer, text, logit_processor, streamer, chat_template = None, max_new_tokens=400, do_sample=False, top_p=None, num_return_sequences=1, **kwargs):
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
        num_return_sequences: number of sequences to return.
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
        
        # Check compatibility between num_return_sequences and do_sample
        if num_return_sequences > 1 and not do_sample:
             logging.warning("⚠️ num_return_sequences > 1 requires do_sample=True. Automatically setting do_sample=True.")
             do_sample = True

        # Device compatibility
        device = model.device
        input_ids = tokenized_input["input_ids"].to(device)
        if input_ids.device != model.device:

            logging.warning(f"Error: 'input_ids' are on device {input_ids.device}, while the model is on device {model.device}. Moving 'input_ids' to the same device as the model.")
            
        attention_mask = tokenized_input["attention_mask"].to(device)
        if attention_mask.device != model.device:
            logging.warning(f"Error: 'attention_mask' is on device {attention_mask.device}, while the model is on device {model.device}. Moving 'attention_mask' to the same device as the model.")


        start = input_ids.shape[1]
        
        # Reset dello stato per garantire pulizia, specialmente se la generazione precedente
        # è terminata prematuramente (max_new_tokens)
        logit_processor.reset()
        streamer.is_first_call = True

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            logits_processor=[logit_processor],
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        answers = []
        for i in range(len(output)):
            decoded_text = tokenizer.decode(output[i][start:], skip_special_tokens=True)
            answers.append(decoded_text)
            logging.info(f"Generated Text {i+1}: {decoded_text}\n\n")

        if num_return_sequences == 1:
            return answers[0]
        else:
            return answers

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Errore nella generazione del testo: {e}")
