"""
generate_LL1_parsing_table.py
=============================
Costruisce la parsing table LL(1) a partire dalla grammatica processata.

Posizione nella pipeline
------------------------
È il secondo passo della fase di setup:

    grammar_generation.py   →   parsing_table()   →   generate_token_maps()
         (produzioni)            (parsing table)         (mappa token)

La parsing table prodotta viene passata sia a generate_token_maps() (per
costruire la mappa terminali→token) sia a PushdownAutomaton (come grammar).

Struttura della parsing table prodotta
---------------------------------------
    { NT: { lookahead_terminal: production_list, ... }, ... }

dove:
  - NT è un non-terminale della grammatica
  - lookahead_terminal è un terminale del vocabolario grammaticale
  - production_list è la lista di simboli da pushare sulla pila del PDA
    quando si incontra quel lookahead; [] significa produzione epsilon.

Esempio:
    { 'S*': { 'positive': ['positive'], 'negative': ['negative'] },
      'A':  { 'happy': ['happy'], 'sad': ['sad'] } }
"""

import os
import json
import logging
from collections import defaultdict


def compute_first_of_string(symbols, first_sets):
    """
    Calcola FIRST(α) per una sequenza di simboli α = [s1, s2, ..., sn].

    Logica
    ------
    Scorre la sequenza da sinistra:
    - Se si incontra un non-terminale X già in first_sets, aggiunge
      FIRST(X) - {ε} al risultato.  Se ε ∈ FIRST(X), continua al simbolo
      successivo (X può "scomparire").
    - Se si incontra un terminale t, aggiunge {t} e si ferma.
    - Se tutti i simboli possono derivare ε, aggiunge ε al risultato.

    Differenza da compute_first_of_sequence
    ----------------------------------------
    Questa funzione assume che tutti i non-terminali siano già in first_sets
    (viene usata DOPO che compute_all_first_sets() ha completato il calcolo).
    compute_first_of_sequence() invece gestisce i non-terminali non ancora
    calcolati (used durante l'iterazione fixed-point).

    Parametri
    ---------
    symbols : list[str]
        La sequenza di simboli (corpo di una produzione o suffisso di essa).
    first_sets : dict  { NT: set[str] }
        I FIRST set già calcolati.  Letto ma non modificato.

    Ritorna
    -------
    set[str]
        L'insieme FIRST della sequenza.

    Uso nel sistema
    ---------------
    - Chiamata da compute_parsing_table() per ogni produzione, per determinare
      sotto quali lookahead inserirla nella parsing table.
    - Chiamata da follow() per calcolare FIRST dei suffissi nelle produzioni.
    """
    first_result = set()
    for symbol in symbols:
        if symbol in first_sets:
            first_result |= first_sets[symbol] - {"ε"}
            if "ε" not in first_sets[symbol]:
                break
        else:
            first_result.add(symbol)
            break
    else:
        first_result.add("ε")
    return first_result


def compute_first_of_sequence(symbols, productions, first_sets):
    """
    Versione di compute_first_of_string tollerante ai FIRST set parziali.

    Logica
    ------
    Identica a compute_first_of_string, ma accetta non-terminali che non sono
    ancora in first_sets (li tratta come aventi FIRST={} provvisoriamente).
    Questo è necessario durante l'iterazione fixed-point di compute_all_first_sets(),
    dove i FIRST set vengono raffinati iterazione dopo iterazione.

    Parametri
    ---------
    symbols : list[str]
    productions : dict  { NT: [production_list, ...] }
        Usato per distinguere non-terminali (presenti in productions) da
        terminali (assenti).
    first_sets : dict  { NT: set[str] }
        I FIRST set parziali dell'iterazione corrente.

    Ritorna
    -------
    set[str]

    Uso nel sistema
    ---------------
    Chiamata solo da compute_all_first_sets() durante l'iterazione fixed-point.
    """
    first_result = set()
    for symbol in symbols:
        if symbol in productions:
            sym_first = first_sets.get(symbol, set())
            first_result |= sym_first - {"ε"}
            if "ε" not in sym_first:
                break
        else:
            first_result.add(symbol)
            break
    else:
        first_result.add("ε")
    return first_result


def compute_all_first_sets(productions):
    """
    Calcola i FIRST set per tutti i non-terminali con algoritmo a punto fisso.

    Logica (fixed-point iteration)
    -------------------------------
    1. Inizializza FIRST(NT) = {} per ogni NT.
    2. Ripete finché nessun set cambia:
       Per ogni NT e ogni sua produzione:
         - Se la produzione è epsilon: aggiunge ε a FIRST(NT).
         - Altrimenti: calcola FIRST della produzione con i set parziali
           correnti e aggiunge il contributo a FIRST(NT).
    3. Termina quando un'intera iterazione non modifica nessun set.

    Correttezza sulla ricorsione mutua (BUG-12 FIX)
    -------------------------------------------------
    L'algoritmo precedente usava un DFS one-shot con un sentinel vuoto per
    rompere i cicli.  Questo produceva FIRST set incompleti per grammatiche
    con ricorsione mutua destra:

        A → B 'y' | 'x'
        B → A 'w' | 'z'

    Il DFS partiva da A, settava il sentinel first_sets['A'] = {}, espandeva
    B → A 'w', trovava il sentinel {} e restituiva {} per A — lasciando
    FIRST(B) = {'z'} invece del corretto {'z', 'x'}.

    Con l'algoritmo fixed-point:
      Iterazione 1: FIRST(A) = {'x'}, FIRST(B) = {'z'}  (solo contributi diretti)
      Iterazione 2: FIRST(A) = {'x','z'}, FIRST(B) = {'z','x'}  (propagazione mutua)
      Iterazione 3: nessun cambiamento → terminazione.

    Proprietà che garantiscono la correttezza
    ------------------------------------------
    - Monotonia: i FIRST set crescono ma non decrescono.
    - Terminazione: i set sono limitati dall'alfabeto terminale finito.
    - Correttezza: ogni terminale raggiungibile tramite qualsiasi catena di
      produzioni viene eventualmente aggiunto al FIRST del NT appropriato.

    Complessità
    -----------
    O(|NT| * |P| * iterazioni), dove |P| è il numero totale di produzioni.
    In pratica converge in 2-3 iterazioni per grammatiche LL(1) tipiche.

    Parametri
    ---------
    productions : dict  { NT: [production_list, ...] }
        La grammatica in forma classica (NT → lista di produzioni).
        Nota: questo è il formato 'grammar' prodotto da parsing_table() con
        defaultdict(list), non il formato con chiavi tuple di final_rules.

    Ritorna
    -------
    dict  { NT: set[str] }
        I FIRST set completi per tutti i non-terminali.

    Collegamento
    ------------
    Chiamata da parsing_table() come primo passo del calcolo della parsing table.
    L'output viene passato a follow() e compute_parsing_table().
    """
    first_sets = {nt: set() for nt in productions}

    changed = True
    while changed:
        changed = False
        for nt, rules in productions.items():
            old_size = len(first_sets[nt])
            for production in rules:
                if production == ["ε"] or production == []:
                    first_sets[nt].add("ε")
                else:
                    contrib = compute_first_of_sequence(production, productions, first_sets)
                    first_sets[nt] |= contrib
            if len(first_sets[nt]) > old_size:
                changed = True

    return first_sets


def find_first(symbol, productions, first_sets):
    """
    Thin wrapper di compatibilità verso il vecchio call-site.

    Logica
    ------
    Se il simbolo è già in first_sets (calcolato da compute_all_first_sets),
    ritorna il valore già presente.  Altrimenti, chiama compute_all_first_sets()
    per calcolare tutti i FIRST set insieme (garantendo la correttezza sulla
    ricorsione mutua) e aggiorna first_sets in-place.

    Uso
    ---
    Non viene più chiamata direttamente da parsing_table() (che ora usa
    compute_all_first_sets() direttamente).  Mantenuta per eventuali
    chiamanti esterni o test che usano l'API legacy.

    Parametri
    ---------
    symbol : str
        Il non-terminale di cui si vuole FIRST, o un terminale (in quel caso
        ritorna {symbol} direttamente).
    productions : dict  { NT: [production_list, ...] }
    first_sets : dict  { NT: set[str] }
        Dizionario da aggiornare in-place se necessario.

    Ritorna
    -------
    set[str]
    """
    if symbol in first_sets:
        return first_sets[symbol]
    if symbol in productions:
        all_sets = compute_all_first_sets(productions)
        first_sets.update(all_sets)
        return first_sets.get(symbol, set())
    else:
        return {symbol}


def follow(productions, first_sets, start_symbol):
    """
    Calcola i FOLLOW set per tutti i non-terminali con algoritmo a punto fisso.

    Definizione
    -----------
    FOLLOW(A) = insieme dei terminali t tali che esiste una derivazione
                S* →* α A t β per qualche α, β.
    Aggiunge anche $ (end-of-input) a FOLLOW del simbolo iniziale.

    Logica (fixed-point iteration)
    -------------------------------
    Per ogni produzione X → ... A β:
      - Se β non è vuoto: aggiunge FIRST(β) - {ε} a FOLLOW(A).
        Se ε ∈ FIRST(β), aggiunge anche FOLLOW(X) a FOLLOW(A)
        (A può essere l'ultimo simbolo effettivo se β deriva ε).
      - Se β è vuoto (A è alla fine): aggiunge FOLLOW(X) a FOLLOW(A).
    Ripete finché nessun set cambia.

    Uso nel sistema
    ---------------
    I FOLLOW set sono necessari per le produzioni epsilon: quando una
    produzione A → ε è applicabile, la inserisce nella parsing table
    per ogni terminale t ∈ FOLLOW(A) (sotto quei lookahead, A "scompare").

    Parametri
    ---------
    productions : dict  { NT: [production_list, ...] }
    first_sets : dict  { NT: set[str] }
        I FIRST set già calcolati da compute_all_first_sets().
    start_symbol : str
        Il simbolo iniziale ('S*' per convenzione).
        Riceve $ nel suo FOLLOW set come marker di end-of-input.

    Ritorna
    -------
    dict  { NT: set[str] }
        I FOLLOW set per tutti i non-terminali.

    Collegamento
    ------------
    Chiamata da parsing_table() dopo compute_all_first_sets().
    L'output viene passato a compute_parsing_table() (funzione interna).
    """
    follow_sets = {nt: set() for nt in productions}
    follow_sets[start_symbol].add("$")

    changed = True
    while changed:
        changed = False
        for lhs, rhs_list in productions.items():
            for rhs in rhs_list:
                for i, symbol in enumerate(rhs):
                    if symbol in productions:
                        old_size = len(follow_sets[symbol])
                        if i + 1 < len(rhs):
                            next_symbols = rhs[i + 1:]
                            first_of_next = compute_first_of_string(next_symbols, first_sets)
                            follow_sets[symbol] |= first_of_next - {"ε"}
                            if "ε" in first_of_next:
                                follow_sets[symbol] |= follow_sets[lhs]
                        else:
                            follow_sets[symbol] |= follow_sets[lhs]
                        if len(follow_sets[symbol]) > old_size:
                            changed = True

    return follow_sets


def parsing_table(final_rules):
    """
    Costruisce la parsing table LL(1) a partire dalle regole di produzione
    processate da grammar_generation.py.

    Logica in quattro passi
    -----------------------
    1. Conversione formato: converte final_rules (con chiavi tuple (NT, 'RULE'))
       in un grammar dict classico { NT: [production_list, ...] } tramite
       defaultdict.  Produzioni multiple per lo stesso NT vengono accumulate.

    2. FIRST sets: chiama compute_all_first_sets() per calcolare i FIRST set
       di tutti i non-terminali con algoritmo fixed-point (corretto per
       ricorsione mutua destra — BUG-12 FIX).

    3. FOLLOW sets: chiama follow() per calcolare i FOLLOW set con algoritmo
       fixed-point (necessari per le produzioni epsilon).

    4. Parsing table: per ogni produzione A → α:
       - Per ogni t ∈ FIRST(α) - {ε}: inserisce A → α nella cella [A][t].
       - Se ε ∈ FIRST(α): per ogni t ∈ FOLLOW(A): inserisce A → [] (epsilon)
         nella cella [A][t].
       - Se una cella ha già un'entry → conflitto LL(1) → ValueError.
       Rimuove le entry con $ (end-of-input) dalla tabella finale.

    Formato di input (final_rules)
    --------------------------------
    { (NT, 'RULE'): [production_list, ...], ... }

    Ogni chiave è una coppia (NT, tag) dove il tag è sempre 'RULE'.
    Le produzioni con prefix-grouping e fattorizzazione sono già state
    risolte da grammar_generation.py.

    Formato di output
    -----------------
    { NT: { lookahead_terminal: production_list, ... }, ... }

    La production_list è:
      - [] per produzioni epsilon
      - ['sym1', 'sym2', ...] per produzioni non-epsilon

    Questo formato viene usato direttamente come self.grammar in
    PushdownAutomaton e come table_parsing in generate_token_maps().

    Effetti collaterali
    -------------------
    Salva la tabella in grammarllm/temp/table_parsing.json tramite
    save_table_parsing_as_txt() per debug e ispezione.

    Eccezioni
    ---------
    ValueError : se viene rilevato un conflitto LL(1) (due produzioni dello
        stesso NT con lo stesso lookahead terminale).  Il messaggio include
        il NT, il terminale conflittuale e le due produzioni alternative.

    Parametri
    ---------
    final_rules : dict  { (NT, 'RULE'): [production_list, ...] }
        L'output di ProductionRuleProcessor.process_full_grammar() in
        grammar_generation.py, con l'aggiunta della produzione EOS:
            final_grammar[('S*','RULE')].append([tokenizer.eos_token])
        aggiunta da get_parsing_table_and_map_tt() in generate_with_constraints.py.

    Ritorna
    -------
    dict  { NT: { lookahead_terminal: production_list } }
        La parsing table LL(1) pronta per PushdownAutomaton e generate_token_maps().

    Collegamento
    ------------
    - Riceve l'input da get_parsing_table_and_map_tt() in generate_with_constraints.py.
    - Il suo output viene passato sia a generate_token_maps() che a
      PushdownAutomaton.__init__() (come argomento 'grammar').
    """

    grammar = defaultdict(list)
    for (nt, _), rules in final_rules.items():
        grammar[nt].extend(rules)

    def save_table_parsing_as_txt(table):
        """
        Serializza la parsing table in JSON leggibile per debug.

        Salva in <package>/temp/table_parsing.json.  Il file viene
        sovrascritto ad ogni chiamata.  Utile per ispezionare la tabella
        generata e diagnosticare conflitti o entry mancanti.

        Il percorso è ancorato alla directory del package (non alla cwd):
        un path relativo creava una directory 'grammarllm/' nella cwd del
        chiamante, che faceva shadowing del package installato ai run
        successivi (ImportError da namespace package).
        """
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_grammar_file = os.path.join(package_dir, 'temp', 'table_parsing.json')
        os.makedirs(os.path.dirname(output_grammar_file), exist_ok=True)
        with open(output_grammar_file, "w", encoding="utf-8") as f:
            f.write("{\n")
            items = list(table.items())
            for i, (nt, rules) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                f.write(f"    {json.dumps(nt)}: {json.dumps(rules)}{comma}\n")
            f.write("}\n")
        logging.info(f"\nTable Parsing saved to {output_grammar_file}")

    def compute_parsing_table(productions, first_sets, follow_sets):
        """
        Costruisce la tabella di parsing LL(1) a partire da FIRST e FOLLOW set.

        Per ogni produzione A → α:
          - Calcola FIRST(α) usando i FIRST set già calcolati.
          - Per ogni terminale t ∈ FIRST(α) - {ε}: inserisce A → α in [A][t].
          - Se ε ∈ FIRST(α): per ogni t ∈ FOLLOW(A), inserisce A → [] in [A][t]
            (A può essere saltata con epsilon quando il lookahead è t ∈ FOLLOW).
          - Conflitto LL(1): se una cella è già occupata, lancia ValueError.

        Le entry con '$' (end-of-input) vengono rimosse dalla tabella finale
        perché il PDA non li incontra mai come lookahead durante la generazione
        (la generazione termina con il token EOS del modello, non con $).

        Parametri
        ---------
        productions : dict  { NT: [production_list, ...] }
        first_sets : dict  { NT: set[str] }
        follow_sets : dict  { NT: set[str] }

        Ritorna
        -------
        dict  { NT: { lookahead_terminal: production_list } }

        Eccezioni
        ---------
        ValueError : conflitto LL(1).
        """
        table = {nt: {} for nt in productions}
        for non_terminal, rules in productions.items():
            for rule in rules:
                first_alpha = compute_first_of_string(rule, first_sets)
                for terminal in first_alpha - {'ε'}:
                    if terminal in table[non_terminal]:
                        raise ValueError(
                            f"Conflict: {non_terminal} → {terminal} "
                            f"{table[non_terminal][terminal]}!\n"
                            f"Regola attuale: {terminal} {rule}!"
                        )
                    table[non_terminal][terminal] = rule
                if 'ε' in first_alpha:
                    for terminal in follow_sets[non_terminal]:
                        if terminal in table[non_terminal]:
                            raise ValueError(
                                f"Conflict: {non_terminal} → {terminal} "
                                f"{table[non_terminal][terminal]}!\n"
                                f"Regola attuale: {terminal} {rule}!"
                            )
                        table[non_terminal][terminal] = []
        for key in table:
            table[key].pop('$', None)
        return table

    logging.info("\nProcessed grammar:\n")
    logging.info(final_rules)

    first_sets = compute_all_first_sets(grammar)
    logging.info("\nFirst sets:\n")
    logging.info(first_sets)

    follow_sets = follow(grammar, first_sets, 'S*')
    logging.info("\nFollow sets:\n")
    logging.info(follow_sets)

    table = compute_parsing_table(grammar, first_sets, follow_sets)
    save_table_parsing_as_txt(table)

    return table