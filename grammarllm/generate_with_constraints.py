from .scripts.grammar_generation import ProductionRuleProcessor
from .scripts.map_terminal_tokens import generate_token_maps
from .scripts.generate_LL1_parsing_table import parsing_table

from .modules.BaseStreamer import BaseStreamer
from .modules.PushdownAutomaton import PushdownAutomaton
from .modules.SimpleLogitProcessor import MaskLogitsProcessor

import logging
import os

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

def generate_text(model, tokenizer, text, logit_processor, streamer, chat_template = None, max_new_tokens=400, do_sample=False, temperature=None, top_p=None, **kwargs):
    """
    Genera testo vincolato dalla grammatica, con configurazione dei parametri di generazione sicura.

    Args:
        model: Il modello pre-addestrato.
        tokenizer: Il tokenizer del modello.
        text: Input text iniziale.
        logit_processor: Processor dei logit basato sulla grammatica.
        streamer: Streamer per l'output live.
        max_new_tokens: Numero massimo di nuovi token da generare.
        do_sample: Se True, abilita la generazione stocastica.
        temperature: Controlla la casualità (usato solo se do_sample=True).
        top_p: Top-p (nucleus sampling), usato solo se do_sample=True.
        **kwargs: Parametri aggiuntivi opzionali per model.generate().
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
            #logging.info(tokenized_input) #DEBUG
        else:
            tokenized_input = tokenizer(text, return_tensors="pt")

        # Safe defaults
        kwargs.setdefault("num_beams", 1)  # beam search disattivato
        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        # Sicurezza num_beams
        if kwargs["num_beams"] != 1:
            logging.warning("⚠️ num_beams > 1 non è compatibile con la generazione vincolata da grammatica. Impostato automaticamente a num_beams=1.")
            kwargs["num_beams"] = 1

        # Sampling parameters
        if do_sample:
            if temperature is not None:
                kwargs["temperature"] = temperature
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
            logging.warning("Errore: gli 'input_ids' sono sulla device {input_ids.device}, mentre il modello è sulla device {model.device}. Spostando 'input_ids' sulla stessa device del modello.")
        
        attention_mask = tokenized_input["attention_mask"].to(device)
        if attention_mask.device != model.device:
            logging.warning(f"Errore: l'attention_mask è sulla device {attention_mask.device}, mentre il modello è sulla device {model.device}. Spostando 'attention_mask' sulla stessa device del modello.")
        

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
