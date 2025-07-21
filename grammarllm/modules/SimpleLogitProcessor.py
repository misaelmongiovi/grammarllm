
import logging
from transformers import LogitsProcessor
import torch

class MaskLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, pda):
        self.tokenizer = tokenizer
        self.pda = pda

    def log_top_10_scores(self, filtered_probabilities, prefix):
        top_probs, top_indices = torch.topk(filtered_probabilities, 10, dim=1)
        top_token_ids = top_indices[0].tolist()
        top_probs = top_probs[0].tolist()
        top_token_labels = self.tokenizer.convert_ids_to_tokens(top_token_ids)

        log_message = f"{prefix}:\nTop 10 Tokens!!!\n"
        for token, prob in zip(top_token_labels, top_probs):
            log_message += f"Token: {token}, Probability: {prob:.6f}\n"
            
        logging.info(log_message)

    def __call__(self, input_ids, scores):
        logging.info(f"Stack: {self.pda.stack[::-1]}") 
        
        valid_tokens = self.pda.get_tokens()
        valid_tokens_ids = valid_tokens

        if valid_tokens_ids:
            logging.info("\n\nLogitsProcessor attivato!")  
            original_probabilities = torch.softmax(scores, dim=-1)
            self.log_top_10_scores(original_probabilities, prefix="Original")

            filtered_scores = scores.clone()

            filtered_scores = torch.full_like(scores, -float('inf'))
            filtered_scores[:, valid_tokens_ids] = scores[:, valid_tokens_ids]
            filtered_probabilities = torch.softmax(filtered_scores, dim=-1)


            self.log_top_10_scores(filtered_probabilities, prefix="Filtered")

            return filtered_scores

        else:
            logging.info(f"Valid tokens è vuoto!{valid_tokens}")
            if self.pda.eos():
                logging.info("stack vuoto quindi eos True")
                valid_tokens_ids = [self.tokenizer.eos_token_id]

                logging.info("\n\nposso generare solo eos perché stack vuoto!")
                logging.info("LogitsProcessor attivato!")  
                original_probabilities = torch.softmax(scores, dim=-1)
                self.log_top_10_scores(original_probabilities, prefix="Original")

                # # Applica la stessa logica per EOS
                filtered_scores = scores.clone()
                filtered_scores = torch.full_like(scores, -float('inf'))
                filtered_scores[:, valid_tokens_ids] = scores[:, valid_tokens_ids]

                filtered_probabilities = torch.softmax(filtered_scores, dim=-1)
                self.log_top_10_scores(filtered_probabilities, prefix="Filtered")

                return filtered_scores
            else:
                logging.info("Valid tokens è vuoto e eos() è False, nessun filtro applicato.")
                return scores