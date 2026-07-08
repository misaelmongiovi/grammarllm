"""
generate_with_constraints.py
==============================
Punto di ingresso pubblico di GrammarLLM. Espone le funzioni che l'utente
chiama direttamente per vincolare la generazione di testo a una grammatica.

Pipeline completa
-----------------

  Fase 1 — Setup (una volta per grammatica)
  ------------------------------------------
  pars_tab, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
      │
      ├─► ProductionRuleProcessor.process_full_grammar(productions)
      │       Converte la grammatica utente (<<tag>>) in grammatica LL(1)
      │
      ├─► parsing_table(final_grammar)
      │       Calcola FIRST/FOLLOW e costruisce la tabella LL(1)
      │
      └─► generate_token_maps(tokenizer, pars_tab)
              Mappa terminali → token ID del vocabolario

  Fase 2 — Preparazione parametri (una volta per sessione di generazione)
  ------------------------------------------------------------------------
  pdas, streamer = generate_grammar_parameters(tokenizer, pars_tab, map_tt)
      │
      └─► PushdownAutomaton(pars_tab, 'S*', map_tt)  × num_return_sequences

  Fase 3 — Generazione (ogni chiamata)
  -------------------------------------
  result = generate_text(model, tokenizer, text, pdas, streamer, ...)
      │
      ├─► StatelessLogitsProcessor(tokenizer, pdas, ...)
      ├─► model.generate(..., logits_processor=[stateless_processor])
      └─► post-processing: prob, pda_history, decoded text

Utilizzo tipico
---------------
    pars_tab, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
    pdas, streamer   = generate_grammar_parameters(tokenizer, pars_tab, map_tt)
    result = generate_text(model, tokenizer, "Classify:", pdas, streamer,
                           max_new_tokens=10, num_beams=4)
    print(result["text"])        # es. "positive"
    print(result["probability"]) # es. 0.89
"""
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
    """
    Costruisce la tabella di parsing LL(1) e la mappa terminale→token_ID.

    Prima funzione da chiamare. Il risultato è stabile per una data grammatica
    e tokenizer, e va riutilizzato per tutte le generazioni successive.

    Step interni
    ------------
    1. ProductionRuleProcessor.process_full_grammar(productions)
       Converte la grammatica <<tag>> in grammatica LL(1) formale.
    2. Aggiunge tokenizer.eos_token come produzione alternativa di S*,
       in modo che il PDA raggiunga lo stato finale (stack vuoto) quando
       il modello genera EOS.
    3. parsing_table(final_grammar) → tabella LL(1) con FIRST/FOLLOW.
    4. generate_token_maps(tokenizer, pars_tab) → terminale → [token_id].

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
    productions : dict[str, list[str]]
        Grammatica utente: { 'S*': ['<<a>> A', '<<b>>'], 'A': ['<<x>>'] }
    regex_dict : dict[str, re.Pattern], optional
        Pattern regex per terminali aperti (es. numeri interi).
        Chiavi nel formato 'regex_<nome_terminale>'.

    Returns
    -------
    tuple[dict, dict]
        (pars_tab, map_terminal_tokens)

    Raises
    ------
    ValueError
        Se la grammatica non è LL(1) o se un terminale non ha token nel vocab.
    """

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



def generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens,
                                num_return_sequences=1, token_lookahead=True):
    """
    Istanzia i PDA base e lo Streamer per la sessione di generazione.

    Crea num_return_sequences PDA indipendenti. Il primo viene costruito
    direttamente, i successivi tramite deepcopy. Questa funzione si chiama
    una volta per sessione; i PDA vengono poi passati a generate_text()
    che li usa come template (clonandoli per ogni beam/step).

    Perché deepcopy invece di clone()
    -----------------------------------
    I PDA base devono essere completamente indipendenti tra loro poiché
    sono template di riferimento per tutta la sessione. clone() è usato
    invece per le copie effimere durante la generazione in beam search.
    Inconsistenza documentata come BUG-20.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
    pars_tab : dict
        Tabella LL(1) da get_parsing_table_and_map_tt().
    map_terminal_tokens : dict
        Mappa terminale→token_ID da get_parsing_table_and_map_tt().
    num_return_sequences : int
        Numero di sequenze per prompt (default 1).
    token_lookahead : bool
        Default True — masks are computed with the g_t_r lookahead engine,
        allowing merged tokens across terminal boundaries (canonical
        tokenization). Set False for the legacy boundary-strict engine,
        used as the A/B baseline in preserved-mass measurements.

    Returns
    -------
    tuple[list[PushdownAutomaton], BaseStreamer]
        (pdas, streamer) da passare a generate_text().
    """
    # Create Pushdown Automaton based on num_return_sequences
    # We need independent PDA instances for each sequence because they maintain state
    
    pdas = []
    base_pda = PushdownAutomaton(grammar=pars_tab, startSymbol='S*', map=map_terminal_tokens)
    # Token-boundary lookahead (spec L3): ON by default. Pass
    # token_lookahead=False for the boundary-strict A/B baseline.
    base_pda.lookahead = token_lookahead

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
    """
    Configura il sistema di logging di GrammarLLM su due file.

    File prodotti
    -------------
    grammarllm/temp/GRAM-GEN.log
        Log principale (INFO). Flusso di elaborazione grammatica, FIRST/FOLLOW,
        produzioni, metriche di generazione. Sovrascritto ad ogni chiamata.

    grammarllm/temp/GRAM-DETAIL.log
        Log di dettaglio per le distribuzioni logit. Contiene le Rich Table
        top-10 di StatelessLogitsProcessor.log_comparison() (solo se DEBUG).
        Non propagato al root logger per evitare duplicazione.

    Note
    ----
    I file vengono sovrascritti (mode='w+') ad ogni chiamata.
    Chiamare una sola volta all'inizio della sessione.
    """
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
    # detail_logger.setLevel(logging.INFO)
    detail_logger.setLevel(logging.DEBUG)
    # Clear existing handlers to avoid duplicates if re-run
    if detail_logger.hasHandlers():
        detail_logger.handlers.clear()
        
    detail_handler = logging.FileHandler(os.path.join(log_dir, 'GRAM-DETAIL.log'), mode='w+')
    detail_handler.setFormatter(logging.Formatter('%(message)s')) # Raw message for rich output
    detail_logger.addHandler(detail_handler)
    detail_logger.propagate = False # Do not propagate to root logger (avoid double logging)

def generate_text(model, tokenizer, text, logit_processor, streamer, chat_template = None, max_new_tokens=400, do_sample=False, top_p=None, num_return_sequences=1, return_pda_stack=True, **kwargs):
    """
    Genera testo vincolato alla grammatica LL(1) usando model.generate().

    Funzione principale dell'API pubblica. Gestisce tokenizzazione, setup del
    processor, chiamata a model.generate(), calcolo metriche e ricostruzione
    della pda_history.

    Modalità supportate
    -------------------
    Greedy:       do_sample=False, num_beams=1 (default)
    Sampling:     do_sample=True, top_p=0.9
    Beam search:  kwargs={'num_beams': 4}  (Streamer disabilitato da HF)

    Input text
    ----------
    str:                     prompt singolo
    list[dict] + template:  conversazione chat
    list[str]:              batch di prompt (richiede un PDA per prompt)

    Formato risultato
    -----------------
    return_pda_stack=True, num_return_sequences=1:
        {"text": "positive", "probability": 0.89, "log_prob": -0.12,
         "pda_history": [...], "pda_stack": []}

    num_return_sequences=3:
        [result_0, result_1, result_2]  (ordinati per probability desc)

    return_pda_stack=False, output_scores=False, n_ret=1:
        "positive"  (solo stringa)

    Parameters
    ----------
    model : transformers.PreTrainedModel
    tokenizer : transformers.PreTrainedTokenizer
    text : str | list[str] | list[dict]
    logit_processor : list[PushdownAutomaton]
        Lista di PDA base da generate_grammar_parameters().
    streamer : BaseStreamer
    chat_template : str, optional
    max_new_tokens : int
    do_sample : bool
    top_p : float, optional
    num_return_sequences : int
    return_pda_stack : bool
        Se True: include pda_history e pda_stack. Se False: solo text/prob.
    **kwargs
        Passati a model.generate() (num_beams, temperature, ecc.).

    Returns
    -------
    str | dict | list
        Vedi "Formato risultato" sopra.

    Raises
    ------
    RuntimeError
        Wrappa qualsiasi eccezione interna. BUG-21: usare `raise ... from e`
        preserverebbe il tipo originale per i caller.
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
            temperature=temperature,
            # Score history costs one (batch, vocab) tensor per step —
            # only track it when the caller actually asked for scores.
            track_score_history=bool(kwargs.get("output_scores", False))
        )

        # BUG FIX: reset score history and PDA cache before every generation.
        # original_scores_history and filtered_scores_history accumulate one
        # (batch * beams, vocab_size) tensor per step and are never cleared
        # automatically.  Without reset(), repeated calls to generate_text
        # with the same processor instance would accumulate GBs of tensors.
        # The new StatelessLogitsProcessor is created fresh each call so the
        # cache starts empty, but the explicit reset() call makes the
        # invariant visible and protects against future refactors that reuse
        # the processor instance across calls.
        stateless_processor.reset()

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
                    "token_ids": new_tokens,
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
                    
                    # Fix Beam Search Indexing:
                    # When num_beams > 1, HF reorders the beams at each step. 
                    # scores[t] contains (batch * beams) distributions.
                    # outputs.beam_indices[i, t] tells us which beam produced the t-th token for sequence i.
                    has_beams = hasattr(outputs, "beam_indices") and outputs.beam_indices is not None
                    
                    mapped_scores = []
                    mapped_orig_scores = []
                    for t in range(len(outputs.scores)):
                        # beam_idx for sequence i at step t
                        idx = outputs.beam_indices[i, t].item() if has_beams else i
                        # BUG FIX: beam_indices is -1 for steps after the
                        # sequence finished; -1 would silently index the LAST
                        # row of scores, appending another beam's distribution.
                        if idx < 0:
                            break

                        mapped_scores.append(outputs.scores[t][idx].tolist())
                        if t < len(stateless_processor.original_scores_history):
                            mapped_orig_scores.append(stateless_processor.original_scores_history[t][idx].tolist())
                    
                    result_item["scores"] = mapped_scores
                    result_item["original_scores"] = mapped_orig_scores
                
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
        # `from e` preserves the original exception type and traceback chain
        # for callers that need to distinguish ValueError (grammar) from
        # infrastructure errors.
        raise RuntimeError(f"Errore nella generazione del testo: {e}") from e