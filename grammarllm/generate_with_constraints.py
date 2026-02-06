from .scripts.grammar_generation import ProductionRuleProcessor
from .scripts.map_terminal_tokens import generate_token_maps
from .scripts.generate_LL1_parsing_table import parsing_table

from .modules.streamer import BaseStreamer
from .modules.automaton import PushdownAutomaton
from .modules.logits_processor import StatelessLogitsProcessor

import logging
import os
import copy
import torch

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
    # For compatibility, we return base PDAs which can be used by the new processor
    return pdas, BaseStreamer(tokenizer, pdas)

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
    
    # Define detailed logger
    detail_logger = logging.getLogger("grammarllm.detail")
    detail_logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicates if re-run
    if detail_logger.hasHandlers():
        detail_logger.handlers.clear()
        
    detail_handler = logging.FileHandler(os.path.join(log_dir, 'GRAM-DETAIL.log'), mode='w+')
    detail_handler.setFormatter(logging.Formatter('%(message)s')) # Raw message for rich output
    detail_logger.addHandler(detail_handler)
    detail_logger.propagate = False # Do not propagate to root logger (avoid double logging)

def generate_text(model, tokenizer, text, logit_processor, streamer, chat_template = None, max_new_tokens=400, do_sample=False, top_p=None, num_return_sequences=1, return_pda_stack=True, **kwargs):
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
        # Pre-execution checks for tokenizer 
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                logging.warning("Tokenizer has no pad_token or eos_token to use as pad.")
                raise ValueError("Tokenizer has no pad_token or eos_token to use as pad.")
        
        # Enforce left padding for decoder-only generation
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"

        # TO USE WHEN CREATE PROMPT IS USED AND PROMPT IS A LIST
        if isinstance(text,list):
            if chat_template is not None:
                # LIST WITH CHAT TEMPLATE -> CONVERSATION
                tokenizer.chat_template = chat_template
                tokenized_input = tokenizer.apply_chat_template(text, 
                                                            tokenize=True,
                                                            add_generation_prompt=True,
                                                            return_dict=True,
                                                            padding=True,
                                                            return_tensors="pt").to(model.device)
            else:
                # LIST WITHOUT CHAT TEMPLATE -> BATCH OF PROMPTS
                # Se l'utente passa una lista di stringhe ["prompt1", "prompt2"], lo trattiamo come batch
                # Assicuriamoci che siano stringhe
                if all(isinstance(t, str) for t in text):
                     # Padding è necessario per batch input
                     tokenized_input = tokenizer(text, return_tensors="pt", padding=True)
                else:
                    raise ValueError("Se `text` è una lista e `chat_template` è None, deve essere una lista di stringhe (batch prompts).")
        else:
            tokenized_input = tokenizer(text, return_tensors="pt")

        # Safe defaults
        kwargs.setdefault("num_beams", 1)  # beam search disabled by default
        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        # Sampling logic was simplified/removed in previous edit but we should probably keep safe defaults or cleanup.
        # Since I'm using Stateless Processor, I should just trust the kwargs.
        # Removing the dangling line.
        
        # Determine num_beams (default 1 if not passed)
        num_beams = kwargs.get("num_beams", 1)
        
        # Check compatibility between num_return_sequences and do_sample
        # If num_beams > 1, we can return multiple sequences WITHOUT sampling (returning top beams).
        if num_return_sequences > 1 and not do_sample and num_beams == 1:
             logging.warning("⚠️ num_return_sequences > 1 with num_beams=1 requires do_sample=True. Automatically setting do_sample=True.")
             do_sample = True

        # Device compatibility
        device = model.device
        input_ids = tokenized_input["input_ids"].to(device)
        if input_ids.device != model.device:

            logging.warning(f"Error: 'input_ids' are on device {input_ids.device}, while the model is on device {model.device}. Moving 'input_ids' to the same device as the model.")
            
        attention_mask = tokenized_input["attention_mask"].to(device)
        if attention_mask.device != model.device:
            logging.warning(f"Error: 'attention_mask' is on device {attention_mask.device}, while the model is on device {model.device}. Moving 'attention_mask' to the same device as the model.")


        # Determine effective batch size
        batch_prompts = input_ids.shape[0]
        # For Beam Search, the processor sees (batch_prompts * num_beams) sequences
        # BUT: LogitsProcessor in HF often gets the 'expanded' input_ids automatically
        
        start_len = input_ids.shape[1]
        
        # Ensure we have enough base_pdas (templates) for the PROMPTS
        # `logit_processor` in arguments is actually just the list of PDAs now (from generate_grammar_parameters return change)
        # Rename for clarity
        base_pdas = logit_processor if isinstance(logit_processor, list) else logit_processor.pdas
        
        if len(base_pdas) < batch_prompts:
             logging.info(f"Expanding Base PDAs from {len(base_pdas)} to {batch_prompts}")
             base_template = base_pdas[0]
             while len(base_pdas) < batch_prompts:
                base_pdas.append(base_template.clone())

        # Calculate effective sequences per prompt (expansion factor)
        sequences_per_prompt = max(num_beams, num_return_sequences)

        # Instantiate the Stateless Processor
        # prompt_len = start_len (length of context before generation)
        temperature = kwargs.get("temperature", 1.0)
        
        stateless_processor = StatelessLogitsProcessor(
            tokenizer=tokenizer,
            base_pdas=base_pdas,
            sequences_per_prompt=sequences_per_prompt,
            prompt_len=start_len,
            temperature=temperature
        )
        
        streamer.is_first_call = True

        # Prepare kwargs for generate
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "logits_processor": [stateless_processor],
            "num_return_sequences": num_return_sequences,
            **kwargs 
        }
        
        if top_p is not None:
            generate_kwargs["top_p"] = top_p

        # HF Transformers does not support Streamer with Beam Search
        if num_beams == 1:
            generate_kwargs["streamer"] = streamer
        
        # Enable output scores
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True

        outputs = model.generate(**generate_kwargs)
        
        # Calculate transition scores
        # normalize_logits=True means we get log_softmax probs
        transition_scores = model.compute_transition_scores(
            outputs.sequences, 
            outputs.scores, 
            beam_indices=getattr(outputs, "beam_indices", None),  #Beam indices are only available when num_beams > 1
            normalize_logits=True
        )

        # Organize answers by prompt
        # Transformers generate() returns batch_size * num_return_sequences items
        # Indices [0...num_return_sequences-1] belong to prompt 0, etc.
        n_ret = num_return_sequences
        batch_answers = []
        
        for p_idx in range(batch_prompts):
            prompt_results = []
            for s_idx in range(n_ret):
                i = p_idx * n_ret + s_idx
                sequence = outputs.sequences[i]
                
                # Calculate metrics
                gen_log_prob = transition_scores[i].sum().item()
                prob = torch.exp(torch.tensor(gen_log_prob)).item()
                
                # Get PDA stack history (always available if requested)
                new_tokens = sequence[start_len:].tolist()
                decoded_text = tokenizer.decode(sequence[start_len:], skip_special_tokens=True)
                
                result_item = {
                    "text": decoded_text,
                    "probability": prob,
                    "log_prob": gen_log_prob
                }
                
                if return_pda_stack:
                    stack_history = []
                    # For each step of generation, get the stack state
                    for t in range(1, len(new_tokens) + 1):
                        prefix = new_tokens[:t]
                        pda_at_t = stateless_processor.get_pda_for_sequence(prefix, prompt_idx=p_idx)
                        stack_history.append(list(pda_at_t.stack))
                    
                    result_item["pda_history"] = stack_history
                    # pda_stack remains the final state for backward compatibility
                    result_item["pda_stack"] = stack_history[-1] if stack_history else list(base_pdas[p_idx].stack)
                if kwargs.get("output_scores", False):
                    result_item["transition_scores"] = transition_scores[i].tolist()
                    result_item["scores"] = [score[i].tolist() for score in outputs.scores]
                    result_item["original_scores"] = [score[i].tolist() for score in stateless_processor.original_scores_history]
                
                prompt_results.append(result_item)
                
                logging.info(f"Prompt {p_idx+1}, Seq {s_idx+1}: {decoded_text}")
                logging.info(f"Metrics: Prob={prob:.6f}, LogProb={gen_log_prob:.4f}")
                if "pda_stack" in result_item:
                    logging.info(f"PDA Stack: {result_item['pda_stack']}\n")
                else:
                    logging.info("\n")
            
            # Sort individual prompt results by probability descending
            prompt_results.sort(key=lambda x: x["probability"], reverse=True)
            
            # Final Return Logic
            # If output_scores is False AND return_pda_stack is False, simplify to just text if only one sequence requested
            if not kwargs.get("output_scores", False) and not return_pda_stack:
                if n_ret == 1:
                    batch_answers.append(prompt_results[0]["text"])
                else:
                    batch_answers.append([r["text"] for r in prompt_results])
            else:
                # Return the full result_item dicts (containing at least text and stack)
                if n_ret == 1:
                    batch_answers.append(prompt_results[0])
                else:
                    batch_answers.append(prompt_results)

        # Final Return Logic
        if batch_prompts > 1:
            return batch_answers
        else:
            # Single prompt: return the list of completions (or single string if n_ret=1)
            return batch_answers[0]

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Errore nella generazione del testo: {e}")
