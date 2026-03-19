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

        for non_terminal, value in map.items():
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

        Nota: dopo reset(), current_terminals è impostato a [] e non viene
        ricalcolato.  La prima chiamata a get_tokens() lo popolerà correttamente.
        """
        self.stack = [self.start_symbol]
        self.current_terminals = []
        logging.info(f"PDA resettato: stack = {self.stack}")

    def recursive_get_tokens(self, stack, visited=None):
        """
        Calcola ricorsivamente i terminali validi come prossimo token, data
        la configurazione corrente della pila.

        Logica
        ------
        Partendo dalla cima della pila, espande i non-terminali sostituendoli
        con le loro produzioni (usando la parsing table) e raccoglie i terminali
        che compaiono in testa a qualsiasi produzione applicabile.

        La parsing table ha la struttura:
            grammar[NT] = { lookahead_terminal: production_list }

        Iterando sui VALORI (le produzioni) invece che sulle chiavi (i lookahead),
        si ottiene l'insieme dei terminali che possono comparire come prossimo
        token.  Le produzioni epsilon ([]) richiedono un trattamento speciale:
        non pushano nulla e ricorrono sul resto della pila, in modo che il
        terminale valido sia il simbolo che si trova SOTTO il NT epsilon nella pila.

        Parametri
        ---------
        stack : list[str]
            Copia della pila corrente (non self.stack — questa funzione è
            chiamata su copie per non modificare lo stato del PDA).

        visited : set[str] | None
            Insieme dei non-terminali già visitati nel percorso di espansione
            corrente.  Usato per prevenire la ricorsione infinita su grammatiche
            con cicli.  Ogni ramo riceve una copia indipendente di visited.

        Ritorna
        -------
        list[str]
            Lista dei terminali (stringhe) validi come prossimo token.
            Può contenere duplicati se più produzioni portano allo stesso
            terminale; get_tokens() li deduplicerà tramite il mapping a token IDs.

        Bug fixati
        ----------
        BUG 1: il vecchio codice iterava su grammar[top].keys() (i lookahead)
               invece che sui valori (le produzioni), restituendo i lookahead
               stessi come "terminali validi".  Per D → ε con tabella
               {'1':[], '2':[], '3':[]}, restituiva ['1','2','3'] invece di
               ricorrere sulla pila sottostante.

        BUG 2: le produzioni epsilon [] non venivano gestite — non si pushava
               nulla e si restituiva [], ignorando il resto della pila.

        Collegamento
        ------------
        Chiamata solo da get_tokens().  Non è parte dell'API pubblica.
        """
        if visited is None:
            visited = set()

        if not stack:
            return []

        top = stack.pop()

        if top in visited:
            return []

        visited.add(top)

        if top not in self.grammar:
            return [top]

        seen_productions = set()
        tokens = []

        for production in self.grammar[top].values():
            prod_key = tuple(production)
            if prod_key in seen_productions:
                continue
            seen_productions.add(prod_key)

            if len(production) == 0:
                tokens += self.recursive_get_tokens(list(stack), set(visited))
            else:
                new_stack = list(stack)
                for sym in reversed(production):
                    new_stack.append(sym)
                tokens += self.recursive_get_tokens(new_stack, set(visited))

        return tokens

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
        return True if not self.stack else False

    def get_stack_debug_info(self):
        """
        Ritorna una stringa leggibile della pila corrente per il debugging.

        Uso
        ---
        Chiamata nei messaggi di log di StatelessLogitsProcessor quando si
        verificano errori durante la re-simulation della history.
        """
        return f"Stack: {self.stack}"