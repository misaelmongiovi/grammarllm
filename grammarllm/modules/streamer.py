"""
streamer.py
===========
Implementa il BaseStreamer per la generazione in tempo reale (greedy/sampling).

Ruolo nel sistema
-----------------
Lo Streamer è il componente che riceve i token mano a mano che vengono
generati da model.generate() e li gestisce in tempo reale. In GrammarLLM,
il suo ruolo principale è duplice:

1. **Logging in tempo reale**: decodifica e logga ogni token generato nel
   momento esatto in cui viene prodotto, utile per monitorare la generazione.

2. **Reset dei PDA a fine generazione**: quando generate() segnala il
   completamento (end()), resetta tutti i PDA allo stato iniziale per
   permettere una nuova generazione senza reinstanziare gli oggetti.

Relazione con StatelessLogitsProcessor
---------------------------------------
In precedenza lo Streamer aggiornava anche lo stato del PDA (next_state per
ogni token generato). Con l'introduzione di StatelessLogitsProcessor, lo
stato del PDA è gestito interamente tramite re-simulation a partire da
input_ids — lo Streamer non deve più aggiornare lo stato perché:
  a) il LogitsProcessor ricalcola lo stato da zero ad ogni step
  b) il beam search rimescola le sequenze tra beam, rendendo l'aggiornamento
     incrementale del PDA non sicuro

Il codice di aggiornamento PDA è quindi commentato (linee 79-83) e mantenuto
solo per documentazione storica.

Limitazione: solo greedy/sampling
-----------------------------------
HuggingFace Transformers non supporta lo Streamer con Beam Search
(num_beams > 1). Per questo generate_with_constraints.generate_text() passa
lo Streamer a model.generate() solo se num_beams == 1.

Pipeline di integrazione
------------------------
    generate_grammar_parameters()
        └─► BaseStreamer(tokenizer, pdas)
              └─► passato a generate_text() come argomento `streamer`
                    └─► model.generate(streamer=streamer)  [solo greedy]
                          ├─► streamer.put(token) ad ogni step
                          └─► streamer.end() a fine generazione
"""

import logging


class BaseStreamer:
    """
    Streamer per la generazione token-by-token in modalità greedy/sampling.

    Riceve i token da model.generate() in tempo reale, li logga, e al
    termine resetta i PDA per permettere generazioni successive.

    Attributi
    ---------
    tokenizer : transformers.PreTrainedTokenizer
        Usato per decodificare i token ID in stringhe leggibili nel log.
    pdas : list[PushdownAutomaton]
        Lista dei PDA da resettare al termine della generazione.
        Accetta sia un singolo PDA che una lista (normalizzato a lista in __init__).
    is_first_call : bool
        Flag per distinguere la prima chiamata a put() (che contiene il prompt)
        dalle chiamate successive (che contengono i token generati).
        Viene resettato a True in end() per preparare la prossima generazione.

    Nota sul BUG-10
    ---------------
    Esiste un edge case: se tutti i PDA sono già in stato EOS prima che
    inizi la generazione (manca un reset()), put() imposta is_first_call=True
    e ritorna immediatamente, entrando in un loop permanente di discard.
    L'unica protezione attuale è un logging.warning. In pratica non si
    manifesta perché generate_text() chiama streamer.is_first_call = True
    prima di ogni generazione.
    """

    def __init__(self, tokenizer, pda):
        """
        Inizializza lo Streamer con il tokenizer e i PDA da gestire.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer del modello. Usato solo per decodificare i token nel log.
        pda : PushdownAutomaton | list[PushdownAutomaton]
            PDA o lista di PDA. Se singolo, viene normalizzato a lista [pda].
            I PDA vengono resettati in end() ma NON aggiornati in put()
            (l'aggiornamento è delegato a StatelessLogitsProcessor).

        Integrazione
        ------------
        Istanziato da generate_grammar_parameters() insieme alla lista di
        PDA base, e restituito come secondo elemento della tupla:
            pdas, streamer = generate_grammar_parameters(tokenizer, pars_tab, map_tt)
        """
        self.tokenizer = tokenizer
        self.pdas = pda if isinstance(pda, list) else [pda]
        self.is_first_call = True

    def put(self, value):
        """
        Riceve un batch di token da model.generate() e li logga.

        Chiamata da HuggingFace ad ogni step di generazione con i token
        appena prodotti per tutte le sequenze nel batch.

        Comportamento per chiamata
        --------------------------
        Prima chiamata (is_first_call=True):
            Contiene il prompt iniziale (input_ids completo). Viene loggato
            e scartato — non è un token generato dal modello.
            is_first_call viene impostato a False.

        Chiamate successive:
            Contengono i token generati, uno per sequenza nel batch.
            Solo la sequenza 0 viene loggata per ridurre la verbosità.
            L'aggiornamento del PDA (next_state) è disabilitato: lo stato
            è gestito interamente da StatelessLogitsProcessor.

        Gestione del batch
        ------------------
        value può essere un tensor PyTorch di shape (batch_size,) o (1,)
        oppure uno scalare. Viene normalizzato a una lista Python di interi.

        Guardia EOS
        -----------
        Se tutti i PDA sono già in stato EOS all'inizio di put(), viene
        emesso un warning (manca il reset()) e il metodo ritorna senza
        processare i token. Questo evita crash ma segnala la condizione
        anomala.

        Parameters
        ----------
        value : torch.Tensor | int
            Token ID o batch di token ID appena generati.
        """
        def all_pdas_eos():
            return all(p.eos() for p in self.pdas)

        if all_pdas_eos() and self.is_first_call:
            logging.warning(
                "⚠️ PDA sono già in stato finale (stack vuoto) PRIMA dell'inizio "
                "di una nuova generazione. Questo indica che manca un `pda.reset()` "
                "o che la grammatica è stata consumata interamente nella generazione "
                "precedente e il pda non è stato resettato -> stack vuoto []"
            )

        if all_pdas_eos():
            # BUG FIX: this used to also set is_first_call = True, which made
            # EVERY subsequent put() treat its tokens as prompt tokens and
            # discard them permanently (silent infinite-discard loop).
            # Just skip this call without touching the first-call flag.
            return

        # Normalizza il tensor/scalare a lista Python
        if hasattr(value, "shape") and len(value.shape) > 0:
            tokens_batch = value.view(-1).tolist()
        else:
            t = value.item() if hasattr(value, "item") else value
            tokens_batch = [t]

        # Prima chiamata: contiene il prompt, non i token generati → scarta
        if self.is_first_call:
            if len(tokens_batch) > 0:
                logging.info(f"Prompt input_ids (First Call): {tokens_batch}")
                logging.info(f"Prompt decodificato (First Call): {self.tokenizer.decode(tokens_batch)}")
            self.is_first_call = False
            return

        # Chiamate successive: token generati dal modello
        for i, token_id in enumerate(tokens_batch):
            if i == 0:  # log solo prima sequenza per ridurre verbosità
                logging.info(
                    f"Token generato (Seq {i}): {token_id} "
                    f"({self.tokenizer.decode([token_id])})"
                )

            # STATE UPDATE DISABLED
            # Lo stato del PDA è ora gestito da StatelessLogitsProcessor
            # tramite re-simulation a partire da input_ids. Lo Streamer
            # non chiama più pda.next_state() per due motivi:
            # 1. Il beam search rimescola le sequenze tra beam, rendendo
            #    l'aggiornamento incrementale non affidabile.
            # 2. StatelessLogitsProcessor garantisce la consistenza dello
            #    stato ricalcolandolo da zero ad ogni step.

    def end(self):
        """
        Segnala la fine della generazione e resetta lo stato per il prossimo ciclo.

        Chiamata automaticamente da model.generate() al completamento.
        Esegue due operazioni:

        Reset: imposta is_first_call=True e chiama pda.reset() su tutti i
        PDA, riportandoli allo stato iniziale (stack = [startSymbol]).
        Questo permette di riusare gli stessi oggetti nella prossima
        chiamata a generate_text() senza reinstanziarli.

        Dov'e' finita la verifica di consistenza (BUG FIX)
        ---------------------------------------------------
        Qui c'era un controllo `if not pda.eos(): logging.warning(...)`.  Era
        strutturalmente falso: ispezionava self.pdas, che put() non avanza mai
        ("STATE UPDATE DISABLED", vedi sopra) da quando lo stato e' gestito da
        StatelessLogitsProcessor via re-simulation.  Quei PDA restano a ['S*']
        per tutta la generazione, quindi il warning si accendeva SEMPRE — anche
        sulle righe perfettamente valide — e non poteva distinguere un
        fallimento vero dal caso normale.  In piu' lo streamer non e' nemmeno
        nel giro con beam search: HF non lo supporta per num_beams > 1 e
        generate_text() lo passa solo in greedy.

        L'avviso non e' stato rimosso, e' stato spostato dove lo stato reale e'
        noto: generate_text(), subito dopo aver ricostruito `pda_stack` con
        StatelessLogitsProcessor.get_pda_for_sequence().  Li' segnala solo i
        casi veri (verificato: 0 falsi allarmi su generazioni valide, 5/5 su un
        troncamento forzato a max_new_tokens).

        Nota sull'integrazione con StatelessLogitsProcessor
        ----------------------------------------------------
        Il reset dei PDA qui non influenza la cache di StatelessLogitsProcessor
        (pda_cache), perché il processor viene ricreato fresh ad ogni chiamata
        a generate_text() e reset() viene chiamato esplicitamente su di esso
        prima della generazione. I due reset sono quindi indipendenti.
        """
        logging.info("=== Fine generazione ===")

        self.is_first_call = True
        for pda in self.pdas:
            pda.reset()

        logging.info("Streamer e tutti i PDA resettati per prossima generazione")