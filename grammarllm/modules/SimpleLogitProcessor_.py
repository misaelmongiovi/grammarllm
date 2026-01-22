import logging
from transformers import LogitsProcessor
import torch
from scipy.stats import entropy
import numpy as np

class MaskLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor che filtra token basandosi su una lista di PDA (uno per sequenza nel batch).
    Gestisce correttamente la generazione di EOS e raccoglie metriche.
    """
    def __init__(self, tokenizer, pda, return_original_dist=False):
        self.tokenizer = tokenizer
        # Supporta sia singolo PDA che lista di PDA
        self.pdas = pda if isinstance(pda, list) else [pda]
        self.return_original_dist = return_original_dist
        self.generation_ended = False  # Flag per terminazione generazione (globale o per-batch da gestire meglio se servisse)
        self.points = []  # Metriche: (entropy, invalid_mass) - Solo per la prima sequenza
        self.preserved_mass = []  # Storia della massa preservata - Solo per la prima sequenza
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
        # Non resettiamo i PDA qui, lo fa lo streamer o l'utente esternamente

    def log_top_10_scores(self, filtered_probabilities, prefix):
        """Log dei top 10 token con le loro probabilità (Solo per la prima sequenza)."""
        # Prende solo la prima riga del batch
        top_probs, top_indices = torch.topk(filtered_probabilities[0].unsqueeze(0), 10, dim=1)
        top_token_ids = top_indices[0].tolist()
        top_probs = top_probs[0].tolist()
        top_token_labels = self.tokenizer.convert_ids_to_tokens(top_token_ids)

        log_message = f"{prefix} (Seq 0):\nTop 10 Tokens:\n"
        for token, prob in zip(top_token_labels, top_probs):
            log_message += f"  {token}: {prob:.6f}\n" 
        logging.info(log_message)

    def log_valid_tokens_prob_mass(self, probabilities, valid_tokens, prefix):
        """
        Calcola e logga la massa di probabilità dei token validi/invalidi per la prima sequenza.
        Returns: tuple: (valid_mass, invalid_mass)
        """
        if not valid_tokens:
            logging.info(f"{prefix} (Seq 0) - Nessun token valido disponibile.")
            return 0.0, 1.0
        
        # probabilities[0] è la prob dist della prima sequenza
        valid_probs = probabilities[0, valid_tokens]
        cumulative_prob_mass_valid = valid_probs.sum().item()
        cumulative_prob_mass_invalid = 1.0 - cumulative_prob_mass_valid

        logging.info(f"{prefix} (Seq 0) - Massa valida: {cumulative_prob_mass_valid:.6f}, "
                    f"Massa invalida: {cumulative_prob_mass_invalid:.6f}")

        return cumulative_prob_mass_valid, cumulative_prob_mass_invalid

    def log_invalid_tokens_entropy(self, probabilities, valid_tokens, prefix):
        """
        Calcola l'entropia normalizzata (0..1) della distribuzione dei token invalidi per la prima sequenza.
        """
        # probabilities shape (batch, vocab), usiamo solo row 0
        vocab_size = probabilities.shape[1]
        device = probabilities.device

        # Crea maschera per token invalidi
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        if valid_tokens:
            mask[valid_tokens] = False

        invalid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if invalid_indices.numel() == 0:
            logging.info(f"{prefix} (Seq 0) - Entropia normalizzata invalidi: 0.000000 (nessun token invalido)")
            return 0.0

        invalid_probs = probabilities[0, invalid_indices].cpu().numpy()

        if invalid_probs.sum() <= 0.0:
            logging.info(f"{prefix} (Seq 0) - Entropia normalizzata invalidi: 0.000000 (massa nulla)")
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

        logging.info(f"{prefix} (Seq 0) - Entropia normalizzata invalidi: {normalized_entropy:.6f}")
        return normalized_entropy

    def __call__(self, input_ids, scores):
        """
        Filtra i logits basandosi sui token validi dai PDA.
        Supporta batchsize > 1. Ogni riga del batch usa il suo PDA corrispondente.
        
        Args:
            input_ids: Sequenza di token generati finora (batch, seq_len)
            scores: Logits non normalizzati per il prossimo token (batch, vocab_size)
        """

        # Applica temperatura
        scores = scores / self.temperature
        original_scores = scores.clone()

        # Prepariamo i filtered scores inizializzati a -inf
        filtered_scores = torch.full_like(scores, -float('inf'))
        
        # Batch size
        batch_size = scores.shape[0]
        
        # Se abbiamo meno PDA del batch size (es. broadcasting o errore), adattiamo
        # Ma l'architettura prevede 1 PDA per sequenza se diverse.
        # Se c'è solo 1 PDA e batch > 1, assume che condividano lo stesso stato (rischioso se divergono)
        # Qui assumiamo len(self.pdas) == batch_size
        if len(self.pdas) != batch_size:
            # Fallback o errore? Se è 1, lo usiamo per tutti (ma non funzionerà se divergono)
            if len(self.pdas) == 1:
                 # Usa lo stesso PDA per tutti (ok solo se generano identico o non aggiornano stato qui - ma stato è aggiornato dallo Streamer)
                 # ATTENZIONE: Questo processor legge solo lo stato. Lo stato è scritto dallo Streamer.
                 # Se lo Streamer diverged, i pda divergeranno.
                 # Se l'utente ha passato 1 solo PDA ma chiede 3 sequenze, deve averne clonato 3 prima.
                 # Se non lo ha fatto, è un problema.
                 logging.warning(f"Mismatch: Batch size {batch_size} ma solo {len(self.pdas)} PDA. Uso pda[0] per tutti (potrebbe causare errori se sequenze divergono).")
                 current_pdas = [self.pdas[0]] * batch_size
            else:
                 raise ValueError(f"Batch size {batch_size} non compatibile con {len(self.pdas)} PDA.")
        else:
            current_pdas = self.pdas

        # Per logging della prima sequenza
        log_first_seq = True

        for i in range(batch_size):
            pda = current_pdas[i]
            
            # --- LOGIC PER SINGOLO PDA (Simile a prima, ma applicato alla riga i) ---
            
            # Loggo solo per la prima sequenza per non intasare
            do_log = (i == 0) and log_first_seq
            
            if do_log:
                logging.info(f"\n{'='*50}")
                logging.info(f"Stack PDA (Seq {i}): {pda.stack[::-1]}")

            valid_tokens = pda.get_tokens()

            # CASO 1: Ci sono token validi
            if valid_tokens:
                if do_log:
                    logging.info(f"Token validi disponibili (Seq {i}): {len(valid_tokens)}")
                    # Metriche sulla PROPRIA distribuzione (riga i)
                    # Per riusare le funzioni di log che aspettano (batch, vocab), passiamo unsqueeze
                    probs_slice = torch.softmax(scores[i].unsqueeze(0), dim=-1)
                    
                    self.log_top_10_scores(probs_slice, prefix="ORIGINALE")
                    valid_mass, invalid_mass = self.log_valid_tokens_prob_mass(probs_slice, valid_tokens, prefix="ORIGINALE")
                    self.preserved_mass.append(valid_mass)
                    normalized_entropy = self.log_invalid_tokens_entropy(probs_slice, valid_tokens, prefix="ORIGINALE")
                    self.points.append((normalized_entropy, invalid_mass))

                # Copia i punteggi validi
                filtered_scores[i, valid_tokens] = scores[i, valid_tokens]

                if do_log:
                    post_probs_slice = torch.softmax(filtered_scores[i].unsqueeze(0), dim=-1)
                    self.log_top_10_scores(post_probs_slice, prefix="FILTRATO")

            # CASO 2: Nessun token valido
            else:
                if do_log:
                    logging.info(f"Nessun token valido dal PDA (Seq {i})")

                if pda.eos():
                    if do_log:
                        logging.info("Stack PDA vuoto -> Forzando generazione EOS")
                    
                    eos_token_id = self.tokenizer.eos_token_id
                    
                    # Log metriche pre-eos
                    if do_log:
                         probs_slice = torch.softmax(scores[i].unsqueeze(0), dim=-1)
                         valid_mass, _ = self.log_valid_tokens_prob_mass(probs_slice, [eos_token_id], prefix="ORIGINALE (pre-EOS)")
                         self.preserved_mass.append(valid_mass)
                         normalized_entropy = self.log_invalid_tokens_entropy(probs_slice, [eos_token_id], prefix="ORIGINALE (pre-EOS)")
                         self.preserved_mass.append(valid_mass) # Doppia appesa? manteniamo logica originale per ora

                    # Forza EOS
                    filtered_scores[i, eos_token_id] = scores[i, eos_token_id]
                    
                else:
                    if do_log:
                        logging.error(f"ERRORE (Seq {i}): Stack non vuoto ma nessun token valido!")
                        logging.error(f"Stack corrente: {pda.stack}")
                        logging.warning("Forzando EOS per sicurezza")

                    eos_token_id = self.tokenizer.eos_token_id
                    filtered_scores[i, eos_token_id] = scores[i, eos_token_id]

        # Salva history
        if self.return_original_dist:
            self.original_scores_history.append(original_scores)
            self.filtered_scores_history.append(filtered_scores)
            
        return filtered_scores