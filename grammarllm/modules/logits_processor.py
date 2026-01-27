
import logging
from transformers import LogitsProcessor
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from io import StringIO
import os

class StatelessLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, base_pdas, sequences_per_prompt=1, prompt_len=0, temperature=1.0):
        """
        Args:
            tokenizer: The model tokenizer.
            base_pdas: List of template PDAs (one per prompt in the batch).
            sequences_per_prompt: Number of sequences generated for each input prompt (max of num_beams and num_return_sequences).
            prompt_len: Length of the prompt (to skip during re-simulation).
            temperature: Softmax temperature (default 1.0).
        """
        self.tokenizer = tokenizer
        self.base_pdas = base_pdas
        self.sequences_per_prompt = sequences_per_prompt
        self.prompt_len = prompt_len
        self.temperature = temperature
        
        # History tracking (if requested)
        self.original_scores_history = []
        self.filtered_scores_history = []
        
        # Cache for PDA states: { tuple(token_ids): pda_state }
        # Key: tuple of tokens (history)
        # Value: PDA object (cloned and advanced)
        self.pda_cache = {} 
        
        # Logging limiter
        self.log_counter = 0
        
        # Detail Logger
        self.detail_logger = logging.getLogger("grammarllm.detail")

    #currently not used but could be useful
    def reset(self):
        """Resets the history, cache and log counter for a new generation."""
        self.original_scores_history = []
        self.filtered_scores_history = []
        self.pda_cache = {}
        self.log_counter = 0

    def log_comparison(self, orig_probs, filt_probs, beam_idx, step):
        """
        Log Top 10 distribution Comparison using Rich Table.
        """
        # Get Top 10 for Original
        top_orig_val, top_orig_ind = torch.topk(orig_probs, 10)
        orig_tokens = self.tokenizer.convert_ids_to_tokens(top_orig_ind.tolist())
        orig_vals = top_orig_val.tolist()

        # Get Top 10 for Filtered
        top_filt_val, top_filt_ind = torch.topk(filt_probs, 10)
        filt_tokens = self.tokenizer.convert_ids_to_tokens(top_filt_ind.tolist())
        filt_vals = top_filt_val.tolist()
        
        table = Table(title=f"Sequence {beam_idx} - Step {step} (Comparison)", show_lines=True)
        table.add_column("Original Token", style="cyan")
        table.add_column("Orig Prob", justify="right", style="green")
        table.add_column("Filtered Token", style="magenta")
        table.add_column("Filt Prob", justify="right", style="yellow")
        
        for i in range(10):
            o_tok = str(orig_tokens[i]) if i < len(orig_tokens) else ""
            o_prob = f"{orig_vals[i]:.6f}" if i < len(orig_vals) else ""
            
            f_tok = str(filt_tokens[i]) if i < len(filt_tokens) else ""
            f_prob = f"{filt_vals[i]:.6f}" if i < len(filt_vals) else ""
            
            table.add_row(o_tok, o_prob, f_tok, f_prob)
            
        # Capture rich output to string
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        console.print(table)
        
        self.detail_logger.info(buf.getvalue())

    def __call__(self, input_ids, scores):
        """
        Args:
            input_ids: (batch_size * num_beams, seq_len)
            scores: (batch_size * num_beams, vocab_size)
        """
        batch_size = scores.shape[0]
        
        # Debug only first call or sparse logging
        # We assume prompt is skipped, so shape[1] is absolute length including prompt
        current_len = input_ids.shape[1]
        

        if self.temperature != 1.0:
            scores = scores / self.temperature
        
        # Log Logic: Log every step? The user asked for "distribution before and after"
        # and "different sentences". For Beam Search, logging every step for every beam is verbose but requested.
        # We will log ALL beams.
        
        # Calculate Original Distribution (Before Masking)
        raw_scores = scores.clone() # Capturing original logits
        original_probs = F.softmax(raw_scores, dim=-1)
        
        # Log Logic: Log every step? The user asked for "distribution before and after"
        # and "different sentences". For Beam Search, logging every step for every beam is verbose but requested.
        # We will log ALL beams.
        
        # We deferred logging 'original' to combine it with 'filtered' at the end.

        for i in range(batch_size):
            # 1. Identify Parent Prompt
            # If batch_size=6 and num_beams=3 -> Prompts: 0,0,0, 1,1,1
            # Index i maps to prompt: i // num_beams
            # Note: We must clamp index just in case of mismatch, though standard HF behavior guarantees this.
            prompt_idx = i // self.sequences_per_prompt
            
            if prompt_idx >= len(self.base_pdas):
                logging.warning(f"Batch index {i} maps to prompt {prompt_idx} but only {len(self.base_pdas)} PDAs available. Wrapping mod.")
                prompt_idx = prompt_idx % len(self.base_pdas)

            base_pda = self.base_pdas[prompt_idx]

            # 2. Extract Generation History (Skip Prompt)
            # input_ids[i] includes prompt + new tokens
            current_seq = input_ids[i]
            history_tokens = current_seq[self.prompt_len:].tolist()
            history_tuple = tuple(history_tokens)
            # Use (prompt_idx, history_tuple) as key to avoid collisions between different prompts in a batch
            cache_key = (prompt_idx, history_tuple)

            # 3. Retrieve or Re-Simulate PDA
            if cache_key in self.pda_cache:
                # Cache Hit
                pda = self.pda_cache[cache_key]
            else:
                # Cache Miss - Needs Re-simulation
                # Optimization: Can we find a prefix in cache?
                # Ideally, history = prefix + [new_token].
                # We check cache[prefix].
                
                # Try finding closest ancestor in cache
                found_ancestor = False
                prefix_tuple = history_tuple[:-1]
                
                if len(history_tokens) > 0 and (prompt_idx, prefix_tuple) in self.pda_cache:
                     # Linear Advance: Clone ancestor and step once
                     ancestor_pda = self.pda_cache[(prompt_idx, prefix_tuple)]
                     pda = ancestor_pda.clone()
                     try:
                         pda.next_state(history_tokens[-1])
                         found_ancestor = True
                     except Exception as e:
                         # Invalid transition in history (shouldn't happen if masked correctly before)
                         # Fallback to base
                         logging.error(f"Error advancing cached PDA: {e}. Falling back to base.")
                         pda = base_pda.clone()
                else:
                    # Full Re-simulation from Base
                    pda = base_pda.clone()
                    for token in history_tokens:
                        try:
                            pda.next_state(token)
                        except Exception as e:
                             # This catches cases where history is invalid relative to grammar
                             if do_log: logging.debug(f"History mismatch (likely EOS or forced): {e}")
                             break 
                
                # Store in cache
                self.pda_cache[cache_key] = pda

            # 4. Get Valid Tokens & Mask
            if pda.eos():
                # Stack empty -> Allow only EOS or Pad
                # Assuming eos_token_id is available in tokenizer
                # Mask EVERYTHING except EOS
                scores[i, :] = -float("inf")
                scores[i, self.tokenizer.eos_token_id] = 0
            else:
                valid_tokens = pda.get_tokens()
                
                if not valid_tokens:
                     # No valid tokens but stack not empty? Dead end logic
                     if do_log: logging.warning(f"PDA {i} Dead End. Stack: {pda.stack}")
                     # Force EOS to exit gracefully
                     scores[i, :] = -float("inf")
                     scores[i, self.tokenizer.eos_token_id] = 0
                else:
                    # Convert terminals to token IDs
                    # Note: pda.get_tokens() returns TERMINALS or TOKENS?
                    # Looking at PushdownAutomaton.py: get_tokens() returns TOKENS (ids).
                    # "tokens.update(self.map_terminals_tokens[terminal])"
                    
                    valid_ids = list(valid_tokens)
                    mask = torch.ones_like(scores[i], dtype=torch.bool)
                    mask[valid_ids] = False # False = Do not mask (Keep)
                    

                    scores[i] = scores[i].masked_fill(mask, -float('inf'))

        # Log Comparison (Original vs Filtered)
        filtered_probs = F.softmax(scores, dim=-1)
        for i in range(batch_size):
             self.log_comparison(original_probs[i], filtered_probs[i], beam_idx=i, step=current_len)

        # Save history if needed
        self.original_scores_history.append(raw_scores)
        self.filtered_scores_history.append(scores.clone())
        
        return scores
