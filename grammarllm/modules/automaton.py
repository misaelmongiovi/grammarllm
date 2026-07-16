"""
automaton.py
============
Implementa il Pushdown Automaton (PDA) deterministico che guida la generazione
vincolata a grammatica.

Posizione nella pipeline
------------------------
È il motore centrale dell'intera libreria.  Viene istanziato da
generate_with_constraints.py (tramite generate_grammar_parameters) e poi
clonato per ogni beam/sequenza all'interno di StatelessLogitsProcessor.

Flusso dati:
    parsing_table()           →  grammar dict  (input di __init__)
    generate_token_maps()     →  map dict       (input di __init__)
    StatelessLogitsProcessor  →  clone() + next_state() + get_tokens()
"""

import logging

try:
    from .lookahead import REGEX_TERMINALS_KEY
except ImportError:                      # bare-import test path
    from lookahead import REGEX_TERMINALS_KEY


class PushdownAutomaton:
    """
    Automa a pila deterministico (PDA) che traccia lo stato di parsing
    durante la generazione vincolata a grammatica LL(1).

    Struttura interna
    -----------------
    - stack : list[str]
        La pila del parser.  All'inizio contiene solo il simbolo iniziale S*.
        Man mano che i token vengono generati, i non-terminali vengono espansi
        e i terminali vengono consumati.  La pila vuota segnala che la
        grammatica è stata completamente soddisfatta (EOS).

    - grammar : dict  { NT: { lookahead_terminal: production_list } }
        La parsing table LL(1) prodotta da parsing_table().  È condivisa
        (read-only) tra tutti i cloni per risparmiare memoria.

    - map_terminals_tokens : dict  { terminal_string: [token_id, ...] }
        Mappa ogni terminale della grammatica all'insieme dei token IDs del
        vocabolario che lo rappresentano.  Prodotta da generate_token_maps().

    - map_tokens_terminals : dict  { token_id: [terminal_string, ...] }
        Inverso del precedente, costruito automaticamente in __init__.
        Permette a next_state() di trovare rapidamente il terminale
        corrispondente a un token generato.

    - current_terminals : list[str]
        Cache dei terminali validi per lo step corrente.  Aggiornata da
        get_tokens() dopo ogni next_state().  Letta da next_state() per
        verificare che il token generato sia ammissibile.

    Integrazione con StatelessLogitsProcessor
    -----------------------------------------
    Il processor mantiene un dizionario-cache { (prompt_idx, history_tuple): PDA }.
    Ad ogni step di generazione:
      1. Recupera o ri-simula il PDA corrispondente alla history del beam.
      2. Chiama get_tokens() per ottenere i token IDs validi.
      3. Maschera tutti gli altri token mettendo il loro logit a -inf.
      4. Dopo che il modello sceglie un token, chiama next_state() per
         avanzare il PDA di uno step.

    Il PDA non viene mai modificato in-place nel processor — viene sempre
    clonato prima dell'uso per garantire l'isolamento tra beam diversi.
    """

    def __init__(self, grammar, startSymbol, map):
        """
        Inizializza il PDA a partire dalla parsing table e dalla mappa terminali→token.

        Parametri
        ---------
        grammar : dict  { NT: { lookahead_terminal: production_list } }
            La parsing table LL(1) restituita da parsing_table().
            Le chiavi sono i non-terminali; i valori sono dizionari che mappano
            ogni terminale di lookahead alla produzione da applicare.
            La produzione epsilon è rappresentata come lista vuota [].

        startSymbol : str
            Il simbolo iniziale della grammatica (convenzionalmente 'S*').
            Viene messo in cima alla pila all'avvio.

        map : dict  { terminal_string: [token_id, ...] }
            La mappa terminali→token IDs restituita da generate_token_maps().
            Ogni terminale della grammatica è associato a uno o più token IDs
            del vocabolario del tokenizer.

        Effetti collaterali
        -------------------
        - Costruisce map_tokens_terminals (inverso di map) in O(|vocab|).
        - Chiama get_tokens() per pre-calcolare i terminali validi allo stato
          iniziale, così current_terminals è già popolato.

        Collegamento
        ------------
        Chiamato da generate_grammar_parameters() in generate_with_constraints.py
        dopo che parsing_table() e generate_token_maps() hanno prodotto i loro
        output.
        """
        self.stack = [startSymbol]
        self.start_symbol = startSymbol
        self.grammar = grammar
        self.map_terminals_tokens = map
        self.map_tokens_terminals = {}
        # ── lookahead state (spec L1) ─────────────────────────────────
        self.residue = ""            # unconsumed suffix of a partially-covered terminal
        self.lookahead = False       # engine flag, set by generate_grammar_parameters
        self.regex_terminals = set(map.get(REGEX_TERMINALS_KEY, []))

        for non_terminal, value in map.items():
            if non_terminal == REGEX_TERMINALS_KEY:
                continue             # metadata, not a terminal→tokens entry
            if isinstance(value, dict):
                for terminal, tokens in value.items():
                    if isinstance(tokens, list):
                        for token in tokens:
                            if token not in self.map_tokens_terminals:
                                self.map_tokens_terminals[token] = []
                            self.map_tokens_terminals[token].append(terminal)
            elif isinstance(value, list):
                for token in value:
                    if token not in self.map_tokens_terminals:
                        self.map_tokens_terminals[token] = []
                    self.map_tokens_terminals[token].append(non_terminal)

        self.get_tokens()

    def clone(self):
        """
        Crea una copia leggera del PDA, condividendo le strutture read-only
        e copiando solo lo stato mutabile.

        Uso principale
        --------------
        Chiamato da StatelessLogitsProcessor ogni volta che:
          - Si recupera un PDA dalla cache (cache hit): il clone è usato per
            lo step corrente senza modificare l'entry cachata.
          - Si avanza un PDA antenato dalla cache (Case A): si clona l'antenato
            e si avanza di un token.
          - Si ri-simula da base_pda (Case B): si clona il template pulito.

        Anche chiamato da generate_with_constraints.py per espandere la lista
        di base_pdas quando il batch ha più prompt del previsto.

        Strutture condivise (read-only, sicure)
        ----------------------------------------
        - grammar : la parsing table non viene mai modificata dopo la costruzione.
        - map_terminals_tokens : idem.
        - map_tokens_terminals : idem.

        Strutture copiate (mutabili, indipendenti)
        ------------------------------------------
        - stack : ogni clone ha la propria pila indipendente.
        - current_terminals : lista mutabile — DEVE essere copiata per valore.
          Se condivisa, get_tokens() su un clone sovrascrive current_terminals
          del padre, corrompendo il beam search silenziosamente (bug fixato).

        Complessità
        -----------
        O(|stack| + |current_terminals|) — costante rispetto alla dimensione
        della grammatica, che può essere grande.
        """
        new_pda = PushdownAutomaton.__new__(PushdownAutomaton)

        new_pda.start_symbol = self.start_symbol
        new_pda.grammar = self.grammar
        new_pda.map_terminals_tokens = self.map_terminals_tokens
        new_pda.map_tokens_terminals = self.map_tokens_terminals
        new_pda.current_terminals = list(getattr(self, 'current_terminals', []))
        new_pda.stack = list(self.stack)
        new_pda.residue = self.residue
        new_pda.lookahead = self.lookahead
        new_pda.regex_terminals = self.regex_terminals   # read-only, shared

        return new_pda

    def reset(self):
        """
        Riporta il PDA allo stato iniziale (pila = [start_symbol]).

        Uso
        ---
        Chiamato da BaseStreamer.end() al termine di una generazione, per
        riportare il PDA al suo stato pulito prima della generazione successiva.
        Non svuota la cache di StatelessLogitsProcessor — quella viene resettata
        separatamente tramite stateless_processor.reset() in generate_text().

        BUG FIX: reset() lasciava current_terminals = [] senza ricalcolarlo.
        Questi PDA sono i template (base_pdas) riusati dalla generazione
        successiva: un clone con current_terminals vuoto fa fallire il primo
        next_state() con ValueError se il chiamante non passa prima da
        get_tokens().  Ricalcoliamo subito, come fa __init__.
        """
        self.stack = [self.start_symbol]
        self.residue = ""
        self.get_tokens()
        logging.info(f"PDA resettato: stack = {self.stack}")

    def recursive_get_tokens(self, stack, visited=None):
        """
        Calcola i terminali validi come prossimo token, data la configurazione
        corrente della pila.

        Logica (scan iterativo FIRST-of-stack)
        ---------------------------------------
        Il prossimo terminale valido è FIRST(α) dove α è il contenuto della
        pila letto dalla cima verso il fondo.  La parsing table LL(1) codifica
        già i FIRST set: per ogni NT, le chiavi delle entry NON-epsilon sono
        esattamente FIRST(NT) - {ε} (compute_parsing_table inserisce la regola
        A → α sotto ogni t ∈ FIRST(α) - {ε}), mentre la presenza di un'entry
        epsilon ([]) indica che il NT è nullable.

        Quindi basta scorrere la pila dall'alto:
          - simbolo terminale → è l'unico consumabile qui; aggiungi e stop.
          - NT → aggiungi le chiavi delle sue entry non-epsilon;
                 se ha un'entry epsilon (nullable) continua col simbolo
                 sottostante, altrimenti stop.

        Bug fixati
        ----------
        BUG 1/2: vecchia versione pre-espansione (vedi storia git).

        BUG 3 (visited over-pruning + costo esponenziale): la versione
        ricorsiva usava un set `visited` per bloccare la ricorsione infinita,
        ma il set bloccava anche la rivisita dello stesso NT presente più in
        basso nella pila, escludendo continuazioni valide.  (Per tabelle che
        superano la validazione LL(1) stretta quella configurazione implica
        un conflitto FIRST/FOLLOW già rifiutato a monte, quindi il pruning
        era per lo più latente — ma la ricorsione clonava pila e visited per
        OGNI produzione a OGNI livello: costo esponenziale nel caso peggiore,
        pagato ad ogni step di generazione.)  Lo scan iterativo è esatto,
        fa un solo passaggio O(|stack| + |row|) e non può divergere.

        Parametri
        ---------
        stack : list[str]
            Copia della pila corrente (la cima è l'ULTIMO elemento,
            coerentemente con stack.pop() usato altrove).
        visited : set[str] | None
            Ignorato — mantenuto solo per compatibilità di firma con i
            call-site legacy.

        Ritorna
        -------
        list[str]
            Lista deduplicata dei terminali validi come prossimo token.

        Collegamento
        ------------
        Chiamata solo da get_tokens().  Non è parte dell'API pubblica.
        """
        terminals = []
        seen = set()

        for symbol in reversed(stack):
            if symbol not in self.grammar:
                # Terminale concreto in cima: è l'unico token consumabile qui.
                if symbol not in seen:
                    terminals.append(symbol)
                break

            nullable = False
            for lookahead, production in self.grammar[symbol].items():
                if production == []:
                    nullable = True
                elif lookahead not in seen:
                    seen.add(lookahead)
                    terminals.append(lookahead)

            if not nullable:
                break

        return terminals

    def get_tokens(self):
        """
        Calcola e aggiorna l'insieme dei token IDs validi come prossimo token.

        Logica
        ------
        1. Chiama recursive_get_tokens() sulla copia della pila corrente per
           ottenere i terminali validi (stringhe grammaticali).
        2. Per ogni terminale, recupera i token IDs corrispondenti da
           map_terminals_tokens e li accumula in un set.
        3. Verifica che i token IDs dei diversi terminali siano disgiunti
           (invariante LL(1): un token ID deve corrispondere a un solo
           terminale nello stato corrente).
        4. Aggiorna self.current_terminals per uso da next_state().

        Ritorna
        -------
        list[int]
            Lista dei token IDs del vocabolario che il modello può generare
            allo step corrente, nel rispetto della grammatica.

        Uso nel sistema
        ---------------
        Chiamata da StatelessLogitsProcessor.__call__() per ottenere la maschera
        da applicare ai logit.  Tutti i token IDs NON in questa lista vengono
        settati a -inf prima del sampling.

        Chiamata anche da next_state() dopo ogni avanzamento per aggiornare
        current_terminals in vista del prossimo step.

        Eccezioni
        ---------
        ValueError : se due terminali diversi mappano agli stessi token IDs.
            Indica un conflitto nella grammatica che non è stato rilevato da
            check_tokens_conflicts() in map_terminal_tokens.py (ad esempio,
            un conflitto cross-state non verificabile staticamente).
        """
        terminals = self.recursive_get_tokens(self.stack.copy())
        tokens = set()

        for terminal in terminals:
            if not set(self.map_terminals_tokens[terminal]).isdisjoint(tokens):
                raise ValueError(
                    f"Token conflict: terminal '{terminal}' maps to token IDs that overlap "
                    f"with already-collected terminals {terminals}. "
                    f"Intersection: {set(self.map_terminals_tokens[terminal]) & tokens}"
                )
            tokens.update(self.map_terminals_tokens[terminal])

        self.current_terminals = terminals
        return list(tokens)

    def next_state(self, token_gen):
        """
        Avanza il PDA di uno step, consumando il token appena generato.

        Logica
        ------
        1. Verifica che token_gen sia tra i token validi dello stato corrente
           (usando map_tokens_terminals e current_terminals).
        2. Identifica il terminale grammaticale corrispondente.
        3. Delega a next_state_terminal() per aggiornare la pila.
        4. Aggiorna current_terminals chiamando get_tokens().

        Parametri
        ---------
        token_gen : int
            Il token ID appena generato dal modello.  Deve essere nell'insieme
            restituito dall'ultimo get_tokens().

        Eccezioni
        ---------
        ValueError : se token_gen non è valido nello stato corrente (0 terminali
            corrispondenti) o è ambiguo (più di 1 terminale corrispondente).
            Nella versione fixata questo errore NON viene mai catturato
            silenziosamente da StatelessLogitsProcessor: si propaga immediatamente,
            segnalando un bug nel masking upstream.

        Collegamento
        ------------
        Chiamata da StatelessLogitsProcessor nei path di re-simulation (Case A e B)
        e da get_pda_for_sequence() per ricostruire la pda_history post-generazione.
        NON viene chiamata direttamente durante il beam search: il processor ricrea
        lo stato PDA dalla history dei token, senza tenere un PDA "live" per beam.
        """
        logging.info(f"current terminals is:{self.current_terminals}")
        token_terminals = self.map_tokens_terminals.get(token_gen, [])
        check_terminals = set(token_terminals).intersection(set(self.current_terminals))
        logging.info(f"check_terminals for token {token_gen} is: {check_terminals} (Associated terminals: {token_terminals})")

        if len(check_terminals) != 1:
            raise ValueError(
                f"Token '{token_gen}' is invalid or ambiguous: found {len(check_terminals)} "
                f"matching terminals {check_terminals} among current valid terminals {self.current_terminals}"
            )
        terminal = list(check_terminals)[0]
        self.next_state_terminal(terminal)
        self.get_tokens()

    def next_state_terminal(self, terminal):
        """
        Aggiorna la pila del PDA consumando il terminale dato.

        Logica (ricorsiva)
        ------------------
        1. Fa pop del top della pila.
        2. Se il top è un non-terminale (presente in self.grammar):
           a. Verifica che esista un'entry per questo terminal nella parsing table.
              Se non esiste → ValueError con messaggio diagnostico (bug fixato:
              prima si otteneva un KeyError generico senza contesto).
           b. Pusha in ordine inverso i simboli della produzione selezionata.
              Per produzioni epsilon ([]) non pusha nulla.
           c. Richiama ricorsivamente next_state_terminal(terminal) per consumare
              il terminale ora che la pila è stata espansa.
        3. Se il top è un terminale: verifica che coincida con terminal.
           Se non coincide → ValueError (stack mismatch).

        Parametri
        ---------
        terminal : str
            Il terminale grammaticale da consumare (stringa, non token ID).
            Deriva da next_state() dopo la risoluzione del token ID.

        Nota sulla distinzione epsilon vs entry mancante
        -------------------------------------------------
        La parsing table usa [] (lista vuota) per le produzioni epsilon.
        Un'entry mancante (token non nella tabella) è un errore di parsing.
        Il codice distingue i due casi esplicitamente:
            grammar[NT][terminal] == []  →  epsilon, non pusha nulla (corretto)
            terminal not in grammar[NT]  →  ValueError con contesto diagnostico

        Collegamento
        ------------
        Chiamata solo da next_state().  Non è parte dell'API pubblica.
        """
        token = terminal
        stack = self.stack
        top = stack.pop()

        if top in self.grammar:
            if token not in self.grammar[top]:
                raise ValueError(
                    f"Parse error: no LL(1) table entry for non-terminal '{top}' "
                    f"with lookahead terminal '{token}'. "
                    f"Valid lookaheads for '{top}': {list(self.grammar[top].keys())}. "
                    f"Current stack: {self.stack}"
                )
            for symbol in reversed(self.grammar[top][token]):
                stack.append(symbol)
            self.next_state_terminal(token)
            return

        if top != token:
            raise ValueError(
                f"PDA stack mismatch: expected terminal '{token}' but top of stack is '{top}'. "
                f"Remaining stack: {stack}"
            )

    def apply_lookahead_path(self, fragments, chars_into_last):
        """
        Consume a merged token described by its lookahead path.

        fragments : tuple[str, ...]
            Terminal strings the token covers, in order. If self.residue is
            non-empty, fragments[0] IS the residue (grammar already advanced
            for it — only its characters remain to be spelled).
        chars_into_last : int
            How many characters of fragments[-1] the token covers.
            == len(fragments[-1]) → fully consumed, residue becomes "".

        Grammar-advance strategy: a terminal is consumed from the stack the
        moment the token ENTERS it (next_state_terminal); the unspelled
        suffix lives in self.residue. eos() stays False until the residue
        is spelled out.
        """
        for i, frag in enumerate(fragments):
            last = (i == len(fragments) - 1)
            if i == 0 and self.residue:
                if frag != self.residue:
                    raise ValueError(
                        f"Lookahead path expected residue {self.residue!r}, got {frag!r}"
                    )
                self.residue = frag[chars_into_last:] if last else ""
                continue
            self.next_state_terminal(frag)
            self.residue = frag[chars_into_last:] if last else ""
        self.get_tokens()

    def eos(self):
        """
        Indica se la grammatica è stata completamente soddisfatta.

        Ritorna True se e solo se la pila è vuota, ovvero tutti i non-terminali
        sono stati espansi e tutti i terminali sono stati consumati.

        Uso nel sistema
        ---------------
        - StatelessLogitsProcessor.__call__(): se eos() è True, forza il token
          EOS settando tutti i logit a -inf tranne tokenizer.eos_token_id.
        - StatelessLogitsProcessor nei loop di re-simulation: interrompe l'avanzamento
          della history quando la grammatica è già stata soddisfatta.
        - BaseStreamer.end(): verifica che tutti i PDA siano in stato EOS al
          termine della generazione.

        Nota
        ----
        La condizione è stack vuota, non semplicemente "ha generato EOS".
        Se il token EOS è nella grammatica come terminale ma la sua produzione
        non porta la pila a zero, eos() restituisce False anche dopo aver
        generato EOS — questo è il BUG-4 documentato, ancora aperto.
        """
        return not self.stack and not self.residue

    def get_stack_debug_info(self):
        """
        Ritorna una stringa leggibile della pila corrente per il debugging.

        Uso
        ---
        Chiamata nei messaggi di log di StatelessLogitsProcessor quando si
        verificano errori durante la re-simulation della history.
        """
        return f"Stack: {self.stack}"