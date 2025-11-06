import logging
class BaseStreamer:
    #Stereamer has the functionality of updating PDA
    """
    Base class from which `.generate()` streamers should inherit.
    """
    def __init__(self, tokenizer, pda):
        self.tokenizer = tokenizer
        self.pda = pda
        self.is_first_call = True  # Variabile per evitare la chiamata iniziale con un tensore di più elementi.

    def put(self, value):

        if self.pda.eos():
            logging.info("STACK vuoto, eos generato! Interrompendo la generazione.")
            return  # Esce dalla funzione, interrompendo l'elaborazione
        
        """Function that is called by `.generate()` to push new tokens"""
        generated_token_id = value[0]
        #TO UNCOMMENT ONLY IF YOU WANT TO SEE THE ID TOKENS OF YOUR PROMPT 
        #logging.info(f"Valore ricevuto in put: {generated_token_id}") #DEBUG
        logging.info(f"Valore ricevuto in put:{generated_token_id}") #DEBUG

        if not self.is_first_call:
            # CORREZIONE: Estrai il valore scalare PRIMA del confronto
            token = generated_token_id.item()
            if token == self.tokenizer.eos_token_id:
                logging.info("eos generato! Interrompendo la generazione.")
                return

            self.pda.next_state(token)  # Esegui il next_state del PDA
            
        self.is_first_call = False
        

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        logging.info("end generation")
        self.is_first_call = True  # Reset per la prossima generazione
        