import logging
from transformers import LogitsProcessor
import torch
from scipy.stats import entropy
import numpy as np

#LOGIT PROCESSOR CHE FILTRA NON UFFICIALE

class MaskLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, pda):
        self.tokenizer = tokenizer
        self.pda = pda
        self.points = []  # Lista per memorizzare i punti (x, y)
        self.preserved_mass = []  # nuovo attributo per salvare l'ultima massa valida

    def log_top_10_scores(self, filtered_probabilities, prefix):
        top_probs, top_indices = torch.topk(filtered_probabilities, 10, dim=1)
        top_token_ids = top_indices[0].tolist()
        top_probs = top_probs[0].tolist()
        top_token_labels = self.tokenizer.convert_ids_to_tokens(top_token_ids)

        log_message = f"{prefix}:\nTop 10 Tokens!!!\n"
        for token, prob in zip(top_token_labels, top_probs):
            log_message += f"Token: {token}, Probability: {prob:.6f}\n" 
        logging.info(log_message)


    def log_valid_tokens_prob_mass(self, probabilities, valid_tokens, prefix):
        """
        Log each valid token's probability and the cumulative probability mass.
        
        Args:
            probabilities (torch.Tensor): Tensor of shape (batch_size, vocab_size) with probabilities.
            valid_tokens (List[int]): List of valid token IDs.
            prefix (str): String prefix for logging.
        """
        if not valid_tokens:
            logging.info(f"{prefix} - No valid tokens available.")
            # Ritorna 0 per valid e 1 per invalid
            return 0.0, 1.0
        
        # Estrai le probabilità dei token validi
        valid_probs = probabilities[:, valid_tokens]
        
        # # Log individual token probabilities
        # log_message = f"{prefix} - Valid Tokens and Their Probability Mass:\n"
        # for token_id, prob in zip(valid_tokens, valid_probs[0].tolist()):
        #     token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        #     log_message += f"Token: {token_str}, Probability: {prob:.6f}\n"
        # logging.info(log_message)
        #self.log_top_10_scores(valid_probs, prefix=f"{prefix} - Valid Tokens Top 10")

        # Log cumulative probability mass
        cumulative_prob_mass_valid = valid_probs.sum().item()
        cumulative_prob_mass_invalid = 1 - cumulative_prob_mass_valid

        logging.info(f"{prefix} - Cumulative Probability Mass of Valid Tokens: {cumulative_prob_mass_valid:.6f}")
        logging.info(f"{prefix} - Cumulative Probability Mass of Invalid Tokens: {cumulative_prob_mass_invalid:.6f}")

        return cumulative_prob_mass_valid, cumulative_prob_mass_invalid

    def log_invalid_tokens_entropy(self, probabilities, valid_tokens, prefix):
        """
        Calcola l'entropia normalizzata (0..1) della distribuzione dei token non validi.
        Usa scipy.stats.entropy che normalizza automaticamente pk.
        """
        batch_size, vocab_size = probabilities.shape
        device = probabilities.device

        # Maschera: True = invalid token
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        if valid_tokens:
            mask[valid_tokens] = False

        invalid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if invalid_indices.numel() == 0:
            logging.info(f"{prefix} - Normalized invalid entropy: 0.000000 (no invalid tokens)")
            return 0.0

        # Prendiamo il primo elemento del batch (come gli altri log)
        invalid_probs = probabilities[0, invalid_indices].cpu().numpy()

        if invalid_probs.sum() <= 0.0:
            logging.info(f"{prefix} - Normalized invalid entropy: 0.000000 (deleted mass zero)")
            return 0.0

        # Calcola l'entropia di Shannon (scipy normalizza pk da solo)
        H = entropy(invalid_probs)  # in nats (base e)

        # Numero di token invalidi con prob > 0
        k = np.count_nonzero(invalid_probs)

        if k <= 1:
            normalized_entropy = 0.0
        else:
            H_max = np.log(k)
            normalized_entropy = float(H / H_max)
            normalized_entropy = max(0.0, min(1.0, normalized_entropy))  # clamp

        logging.info(f"{prefix} - Normalized invalid entropy: {normalized_entropy:.6f}")
        return normalized_entropy

    

    def __call__(self, input_ids, scores):

        # Applica la temperatura se specificata
        temperature = getattr(self, "temperature", 1.0)
        scores = scores / temperature
        logging.info(f"Stack: {self.pda.stack[::-1]}") 
        
        valid_tokens = self.pda.get_tokens()
        valid_tokens_ids = valid_tokens

        if valid_tokens_ids:
            logging.info("\n\nLogitsProcessor attivato!")  
            original_probabilities = torch.softmax(scores, dim=-1)
            self.log_top_10_scores(original_probabilities, prefix="Original")
            cumulative_prob_mass_valid , cumulative_prob_mass_invalid = self.log_valid_tokens_prob_mass(original_probabilities, valid_tokens, prefix="Original Valid Tokens")
            
            self.preserved_mass.append(cumulative_prob_mass_valid)
            # NUOVO: Calcola e logga l'entropia dei token non validi
            normalized_entropy = self.log_invalid_tokens_entropy(original_probabilities, valid_tokens, prefix="Original")

            # Aggiungi il punto (x, y) alla lista dei punti
            self.points.append((normalized_entropy, cumulative_prob_mass_invalid))

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
                cumulative_prob_mass_valid, _ = self.log_valid_tokens_prob_mass(original_probabilities, valid_tokens, prefix="Original Valid Tokens")
                
                self.preserved_mass.append(cumulative_prob_mass_valid)
                # NUOVO: Calcola e logga l'entropia dei token non validi
                self.log_invalid_tokens_entropy(original_probabilities, valid_tokens_ids, prefix="Original")

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