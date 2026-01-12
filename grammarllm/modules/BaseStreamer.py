import logging
class BaseStreamer:
    #Stereamer has the functionality of updating PDA
    """
    Base class from which `.generate()` streamers should inherit.
    """
    def __init__(self, tokenizer, pda):
        self.tokenizer = tokenizer
        self.pda = pda
        self.is_first_call = True
        
    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        # Se il PDA ha già finito, esce subito
        if self.pda.eos() and self.is_first_call:
            logging.warning(
            "⚠️ PDA è già in stato finale (stack vuoto) PRIMA dell'inizio di una nuova generazione. "
            "Questo indica che manca un `pda.reset()` o che la grammatica è stata consumata interamente "
            "nella generazione precedente e il pda non è stato resettao -> stack vuoto []"
        )
            raise  "ERROR ON PDA RESET"
        
        if self.pda.eos():
            self.is_first_call = True
            return

        # Se è la prima chiamata, contiene il prompt iniziale
        if self.is_first_call:
            generated_token_id = value[0]
            #TO UNCOMMENT ONLY IF YOU WANT TO SEE THE ID TOKENS OF YOUR PROMPT 
            #logging.info(f"Valore ricevuto in put: {generated_token_id}") #DEBUG
            logging.info(f"Valore ricevuto in put:{generated_token_id}") #DEBUG
            logging.info(f"detokenizzato: {self.tokenizer.decode(generated_token_id)}") #debug

            self.is_first_call = False
            return  # non processa i token del prompt

        # Da qui in poi arrivano i token generati uno per volta
        token_id = value.item() if hasattr(value, "item") else value
        logging.info(f"Token generato: {token_id} ({self.tokenizer.decode([token_id])})")

        try:
            self.pda.next_state(token_id)
            logging.info(f"Stack PDA aggiornato: {self.pda.stack[::-1]}")
        except Exception as e:
            logging.error(f"Errore durante aggiornamento PDA con token {token_id}: {e}")
            logging.error(f"Stack corrente: {self.pda.stack}")
            raise


    def end(self):
        """
        Chiamato da .generate() per segnalare fine generazione.
        Resetta lo stato per permettere nuove generazioni.
        """
        logging.info("=== Fine generazione ===")
        
        # Verifica finale consistenza
        if not self.pda.eos():
            logging.warning(f"⚠ Generazione terminata ma stack PDA non vuoto: {self.pda.stack[::-1]}")
        else:
            logging.info("✓ Stack PDA correttamente vuoto")
        
        # Reset completo dello stato per la prossima generazione
        # Reset per la prossima generazione
        self.is_first_call = True
        self.pda.reset()  # questo resetta anche il PDA

        logging.info("Streamer e PDA resettati per prossima generazione")

        