import logging
from transformers import LogitsProcessor
import torch
from scipy.stats import entropy
import numpy as np

class MaskLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor che filtra token basandosi su un PDA.
    Gestisce correttamente la generazione di EOS e raccoglie metriche.
    """
    def __init__(self, tokenizer, pda, return_original_dist=False):
        self.tokenizer = tokenizer
        self.pda = pda
        self.return_original_dist = return_original_dist
        self.generation_ended = False  # Flag per terminazione generazione
        self.points = []  # Metriche: (entropy, invalid_mass)
        self.preserved_mass = []  # Storia della massa preservata
        self.temperature = 1.0  # Temperatura di default
        
        # History dei logits (se return_original_dist=True)
        self.original_scores_history = []
        self.filtered_scores_history = []

    def reset(self):
        """Resetta lo stato per una nuova generazione."""
        self.generation_ended = False
        self.points = []
        self.preserved_mass = []
        self.original_scores_history = []
        self.filtered_scores_history = []

    def log_top_10_scores(self, filtered_probabilities, prefix):
        """Log dei top 10 token con le loro probabilità."""
        top_probs, top_indices = torch.topk(filtered_probabilities, 10, dim=1)
        top_token_ids = top_indices[0].tolist()
        top_probs = top_probs[0].tolist()
        top_token_labels = self.tokenizer.convert_ids_to_tokens(top_token_ids)

        log_message = f"{prefix}:\nTop 10 Tokens:\n"
        for token, prob in zip(top_token_labels, top_probs):
            log_message += f"  {token}: {prob:.6f}\n" 
        logging.info(log_message)

    def log_valid_tokens_prob_mass(self, probabilities, valid_tokens, prefix):
        """
        Calcola e logga la massa di probabilità dei token validi/invalidi.
        
        Returns:
            tuple: (valid_mass, invalid_mass)
        """
        if not valid_tokens:
            logging.info(f"{prefix} - Nessun token valido disponibile.")
            return 0.0, 1.0
        
        valid_probs = probabilities[:, valid_tokens]
        cumulative_prob_mass_valid = valid_probs.sum().item()
        cumulative_prob_mass_invalid = 1.0 - cumulative_prob_mass_valid

        logging.info(f"{prefix} - Massa valida: {cumulative_prob_mass_valid:.6f}, "
                    f"Massa invalida: {cumulative_prob_mass_invalid:.6f}")

        return cumulative_prob_mass_valid, cumulative_prob_mass_invalid

    def log_invalid_tokens_entropy(self, probabilities, valid_tokens, prefix):
        """
        Calcola l'entropia normalizzata (0..1) della distribuzione dei token invalidi.
        
        Returns:
            float: Entropia normalizzata tra 0 e 1
        """
        batch_size, vocab_size = probabilities.shape
        device = probabilities.device

        # Crea maschera per token invalidi
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        if valid_tokens:
            mask[valid_tokens] = False

        invalid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if invalid_indices.numel() == 0:
            logging.info(f"{prefix} - Entropia normalizzata invalidi: 0.000000 (nessun token invalido)")
            return 0.0

        invalid_probs = probabilities[0, invalid_indices].cpu().numpy()

        if invalid_probs.sum() <= 0.0:
            logging.info(f"{prefix} - Entropia normalizzata invalidi: 0.000000 (massa nulla)")
            return 0.0

        # Calcola entropia di Shannon
        H = entropy(invalid_probs)  # in nats
        k = np.count_nonzero(invalid_probs)

        if k <= 1:
            normalized_entropy = 0.0
        else:
            H_max = np.log(k)
            normalized_entropy = float(H / H_max)
            normalized_entropy = max(0.0, min(1.0, normalized_entropy))

        logging.info(f"{prefix} - Entropia normalizzata invalidi: {normalized_entropy:.6f}")
        return normalized_entropy

    def __call__(self, input_ids, scores):
        """
        Filtra i logits basandosi sui token validi dal PDA.
        
        Args:
            input_ids: Sequenza di token generati finora
            scores: Logits non normalizzati per il prossimo token
            
        Returns:
            torch.Tensor or tuple: Logits filtrati, o (logits filtrati, logits originali) se return_original_dist è True
        """

        # Applica temperatura
        scores = scores / self.temperature

        original_scores = scores.clone()
        
        # Se la generazione è già terminata, lascia passare tutto
        if self.generation_ended:
            if self.return_original_dist:
                self.original_scores_history.append(original_scores)
                self.filtered_scores_history.append(scores)
            return scores
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Stack PDA: {self.pda.stack[::-1]}")
        
        # Ottieni token validi dal PDA
        valid_tokens = self.pda.get_tokens()
        
        # CASO 1: Ci sono token validi - Applica filtro normale
        if valid_tokens:
            logging.info(f"Token validi disponibili: {len(valid_tokens)}")
            
            # Calcola metriche originali
            original_probabilities = torch.softmax(scores, dim=-1)
            self.log_top_10_scores(original_probabilities, prefix="ORIGINALE")
            
            valid_mass, invalid_mass = self.log_valid_tokens_prob_mass(
                original_probabilities, valid_tokens, prefix="ORIGINALE"
            )
            self.preserved_mass.append(valid_mass)
            
            normalized_entropy = self.log_invalid_tokens_entropy(
                original_probabilities, valid_tokens, prefix="ORIGINALE"
            )
            self.points.append((normalized_entropy, invalid_mass))
            
            # Applica filtro
            filtered_scores = torch.full_like(scores, -float('inf'))
            filtered_scores[:, valid_tokens] = scores[:, valid_tokens]
            
            filtered_probabilities = torch.softmax(filtered_scores, dim=-1)
            self.log_top_10_scores(filtered_probabilities, prefix="FILTRATO")
            
            if self.return_original_dist:
                self.original_scores_history.append(original_scores)
                self.filtered_scores_history.append(filtered_scores)
                
            return filtered_scores
        
        # CASO 2: Nessun token valido - Controlla se stack è vuoto
        else:
            logging.info("Nessun token valido dal PDA")
            
            if self.pda.eos():
                # Stack vuoto: forza generazione EOS
                logging.info("Stack PDA vuoto -> Forzando generazione EOS")
                
                eos_token_id = self.tokenizer.eos_token_id
                original_probabilities = torch.softmax(scores, dim=-1)
                
                self.log_top_10_scores(original_probabilities, prefix="ORIGINALE (pre-EOS)")
                
                valid_mass, _ = self.log_valid_tokens_prob_mass(
                    original_probabilities, [eos_token_id], prefix="ORIGINALE (pre-EOS)"
                )
                self.preserved_mass.append(valid_mass)
                
                self.log_invalid_tokens_entropy(
                    original_probabilities, [eos_token_id], prefix="ORIGINALE (pre-EOS)"
                )
                
                # Forza EOS
                filtered_scores = torch.full_like(scores, -float('inf'))
                filtered_scores[:, eos_token_id] = scores[:, eos_token_id]
                
                filtered_probabilities = torch.softmax(filtered_scores, dim=-1)
                self.log_top_10_scores(filtered_probabilities, prefix="FILTRATO (EOS)")
                
                if self.return_original_dist:
                    self.original_scores_history.append(original_scores)
                    self.filtered_scores_history.append(filtered_scores)
                    
                self.generation_ended = True  # Segnala terminazione
                return filtered_scores
            else:
                # ATTENZIONE: Questo è uno stato di errore!
                logging.error("ERRORE: Stack non vuoto ma nessun token valido disponibile!")
                logging.error(f"Stack corrente: {self.pda.stack}")
                
                # Opzione sicura: forza comunque EOS per terminare
                logging.warning("Forzando EOS per sicurezza")
                eos_token_id = self.tokenizer.eos_token_id
                filtered_scores = torch.full_like(scores, -float('inf'))
                filtered_scores[:, eos_token_id] = scores[:, eos_token_id]
                if self.return_original_dist:
                    self.original_scores_history.append(original_scores)
                    self.filtered_scores_history.append(filtered_scores)

                self.generation_ended = True
                return filtered_scores