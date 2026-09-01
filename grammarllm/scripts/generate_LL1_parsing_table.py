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


# End-of-input marker used while computing FOLLOW sets, then stripped from the
# finished table (the PDA never sees it: generation ends on the model's EOS
# token, not on an end-of-input terminal).
#
# It must be a string that can never be a real grammar symbol. This used to be
# the plain "$", which silently broke any grammar using "$" as a terminal — the
# strip step at the end of compute_parsing_table deleted the user's own entries
# and the terminal simply stopped being derivable, with no error raised.
# ("$" is the quadruple bond in SMILES, which is how this was found.)
# NUL bytes appear neither in tokenizer vocabularies nor in hand-written
# grammars, so this name cannot collide.
EOF_MARKER = "\x00__grammarllm_eof__\x00"


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
    follow_sets[start_symbol].add(EOF_MARKER)

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


# ═════════════════════════════════════════════════════════════════════════
# Riscrittura automatica in forma LL(1)
# ═════════════════════════════════════════════════════════════════════════
#
# Usata da parsing_table() SOLO quando compute_parsing_table() rileva un
# conflitto, cioè solo su grammatiche che verrebbero comunque rifiutate con
# ValueError.  Una grammatica già LL(1) non passa mai di qui e non cambia:
# questa sezione non può far regredire nulla che oggi funzioni.
#
# Perché serve
# ------------
# grammar_generation.left_factor_productions risolve i conflitti FIRST/FIRST
# (due alternative dello STESSO non-terminale che iniziano allo stesso modo).
# Non può però risolvere quelli FIRST/FOLLOW: lì le alternative in
# competizione stanno in non-terminali DIVERSI e nessuna fattorizzazione
# locale le raggiunge.  Esempio reale (OpenSMILES, sezione 2.2):
#
#     S*        -> BRANCHED_ATOM S* | BOND BRANCHED_ATOM S*
#     RINGBONDS -> RINGBOND RINGBONDS | ε
#     RINGBOND  -> digit | BOND digit
#
# Dopo un atomo un simbolo di legame può aprire una chiusura di anello
# (RINGBOND) oppure legare l'atomo successivo della catena (S*).  RINGBONDS è
# annullabile e il legame sta sia nel suo FIRST sia nel suo FOLLOW:
#
#     Conflict: RINGBONDS → = ['RINGBOND', 'RINGBONDS']!
#     Regola attuale: = ['ε']!
#
# Le tre trasformazioni, tutte conservative del linguaggio per costruzione:
#
#   T1  left_factor    raggruppa le alternative per primo simbolo e
#                      fattorizza ogni gruppo di due o più con il prefisso
#                      comune più lungo, ricorsivamente.
#   T2  specialize     per un NT annullabile A con conflitto FIRST/FOLLOW,
#                      sostituisce ogni occorrenza `A γ` con un NT fresco che
#                      denota esattamente L(A)·L(γ).  Le occorrenze
#                      ricorsive-destre di A dentro A si ripiegano sul nuovo
#                      NT: è questo che tiene finito il risultato.  Se A sta
#                      in coda a una produzione di X la sua continuazione è
#                      FOLLOW(X), non locale: si specializza prima X.
#   T3  inline_heads   sostituisce il non-terminale di testa delle
#                      alternative in conflitto con le alternative proprie di
#                      quello, portando alla luce il prefisso terminale
#                      condiviso che T1 può poi fattorizzare.
#
# Limiti, dichiarati e non nascosti
# ----------------------------------
# Non tutti i linguaggi deterministici ammettono una grammatica LL(1),
# quindi un algoritmo completo non può esistere: questo è un SEMI-algoritmo,
# limitato sia nel numero di round sia nella crescita della grammatica.  Se
# non converge non tocca nulla e lascia risalire il ValueError originale,
# che resta il messaggio diagnostico visto dall'utente.
#
# Validazione: equivalenza di linguaggio verificata per enumerazione su
# 14.000 grammatiche casuali (3.950 riscritte, zero divergenze) e sulla
# trascrizione letterale della sezione 2.2 di OpenSMILES.

#: Round massimi del ciclo trasforma-e-ricontrolla.
MAX_TRANSFORM_ROUNDS = 16

#: Fattore massimo di crescita in non-terminali rispetto all'originale:
#: difesa contro il blow-up su grammatiche con enum molto grandi.
MAX_TRANSFORM_GROWTH = 6


def normalise_epsilon(grammar):
    """
    Porta le produzioni epsilon alla forma canonica [].

    process_full_grammar emette l'alternativa epsilon come la produzione a un
    simbolo ['ε'], non come [].  Entrambe le forme sono accettate a valle, ma
    tutte le trasformazioni qui testano la lista vuota: senza normalizzazione
    un conflitto FIRST/FOLLOW non verrebbe riconosciuto come tale e T2 non
    scatterebbe mai.
    """
    return {
        nt: [[] if p == ["ε"] else [s for s in p if s != "ε"] for p in prods]
        for nt, prods in grammar.items()
    }


def find_conflicts(grammar, start_symbol="S*", first_sets=None, follow_sets=None):
    """
    Elenca le celle della parsing table con più di una produzione.

    Ritorna
    -------
    list[tuple[str, str, list[list[str]]]]
        (non-terminale, lookahead, alternative in competizione).  Lista vuota
        se e solo se la grammatica è LL(1).

    Usa gli stessi FIRST/FOLLOW di compute_parsing_table, così la diagnosi e
    la costruzione non possono divergere.
    """
    if first_sets is None:
        first_sets = compute_all_first_sets(grammar)
    if follow_sets is None:
        follow_sets = follow(grammar, first_sets, start_symbol)

    out = []
    for nt, prods in grammar.items():
        cell = {}
        for prod in prods:
            first_alpha = compute_first_of_string(prod, first_sets)
            lookaheads = first_alpha - {"ε"}
            if "ε" in first_alpha:
                lookaheads = lookaheads | follow_sets[nt]
            for terminal in lookaheads:
                cell.setdefault(terminal, []).append(prod)
        for terminal, competing in sorted(cell.items(), key=lambda kv: str(kv[0])):
            if len(competing) > 1:
                out.append((nt, terminal, competing))
    return out


def _fresh_name(base, used):
    """Nome di NT che non collide con nulla già in uso."""
    name, n = base, 1
    while name in used:
        n += 1
        name = f"{base}{n}"
    used.add(name)
    return name


def _reachable(grammar, start_symbol):
    """Scarta i non-terminali non più raggiungibili dal simbolo iniziale."""
    seen, stack = {start_symbol}, [start_symbol]
    while stack:
        for prod in grammar.get(stack.pop(), []):
            for sym in prod:
                if sym in grammar and sym not in seen:
                    seen.add(sym)
                    stack.append(sym)
    return {nt: prods for nt, prods in grammar.items() if nt in seen}


def _longest_common_prefix(prods):
    prefix = []
    for i in range(min(len(p) for p in prods)):
        if len({p[i] for p in prods}) != 1:
            break
        prefix.append(prods[0][i])
    return prefix


def left_factor(grammar, used):
    """
    T1 — fattorizzazione a sinistra per gruppo di primo simbolo.

    Gemella di grammar_generation.left_factor_productions, ma applicata alla
    grammatica GIÀ espansa (dopo la tokenizzazione dei tag), dove possono
    emergere prefissi condivisi che a monte non erano visibili.
    """
    out = {}
    for nt, prods in grammar.items():
        uniq = []
        for p in prods:
            if p not in uniq:
                uniq.append(p)

        epsilons = [p for p in uniq if not p]
        groups = {}
        for p in uniq:
            if p:
                groups.setdefault(p[0], []).append(p)

        main = []
        for group in groups.values():
            if len(group) == 1:
                main.append(group[0])
                continue
            helper = _fresh_name(f"{nt}_FACT", used)
            prefix = _longest_common_prefix(group)
            main.append(prefix + [helper])
            out[helper] = [p[len(prefix):] for p in group]
        main.extend(epsilons)
        out[nt] = main

    # I NT ausiliari appena creati possono a loro volta essere fattorizzabili.
    if any(nt not in grammar for nt in out):
        return left_factor(out, used)
    return out


def specialize(grammar, target, used):
    """
    T2 — sostituisce `target γ` con un NT fresco che denota L(target)·L(γ).

    Ritorna
    -------
    tuple[dict | None, set[str]]
        (grammatica riscritta, bloccanti).  La grammatica è None se la
        trasformazione non è applicabile qui; `bloccanti` contiene gli lhs
        alla cui coda `target` compare, da specializzare prima.
    """
    continuations, blockers = set(), set()
    for lhs, prods in grammar.items():
        for prod in prods:
            for i, sym in enumerate(prod):
                if sym != target:
                    continue
                tail = tuple(prod[i + 1:])
                if lhs == target and not tail:
                    continue        # ricorsione destra su sé stesso: si ripiega
                if not tail:
                    blockers.add(lhs)
                    continue
                continuations.add(tail)

    if blockers or not continuations:
        return None, blockers

    names = {cont: _fresh_name(f"{target}__C", used)
             for cont in sorted(continuations)}
    out = {nt: [list(p) for p in prods] for nt, prods in grammar.items()}

    for cont, name in names.items():
        rules = []
        for alt in grammar[target]:
            if not alt:
                rules.append(list(cont))            # ε · γ = γ
            elif alt[-1] == target:
                rules.append(alt[:-1] + [name])     # ripiega la ricorsione
            else:
                rules.append(alt + list(cont))
        out[name] = rules

    def rewrite(prod):
        res = []
        for i, sym in enumerate(prod):
            if sym == target and tuple(prod[i + 1:]) in names:
                res.append(names[tuple(prod[i + 1:])])
                return res
            res.append(sym)
        return res

    generated = set(names.values())
    for nt in list(out):
        if nt not in generated:
            out[nt] = [rewrite(p) for p in out[nt]]

    return out, set()


def specialize_deep(grammar, target, used, start_symbol, seen=None, depth=0):
    """specialize() che risolve prima i bloccanti, dall'esterno all'interno."""
    if depth > 4:
        return None
    seen = set() if seen is None else seen
    if target in seen:
        return None
    seen.add(target)

    out, blockers = specialize(grammar, target, used)
    if out is not None:
        return _reachable(out, start_symbol)

    for blocker in sorted(blockers):
        lifted = specialize_deep(grammar, blocker, used, start_symbol, seen, depth + 1)
        if lifted is None:
            continue
        retry, _ = specialize(lifted, target, used)
        return _reachable(retry if retry is not None else lifted, start_symbol)
    return None


def inline_heads(grammar, nt, competing, start_symbol):
    """T3 — sostituisce il NT di testa delle alternative in conflitto."""
    new_prods, changed = [], False
    for prod in grammar[nt]:
        head_is_nt = prod and prod[0] in grammar and prod[0] != nt
        if prod in competing and head_is_nt:
            for alt in grammar[prod[0]]:
                candidate = alt + prod[1:]
                if candidate not in new_prods:
                    new_prods.append(candidate)
            changed = True
        elif prod not in new_prods:
            new_prods.append(prod)

    if not changed:
        return None
    out = {k: [list(p) for p in v] for k, v in grammar.items()}
    out[nt] = new_prods
    return _reachable(out, start_symbol)


def ll1ify(grammar, start_symbol="S*", max_rounds=MAX_TRANSFORM_ROUNDS):
    """
    Prova a riscrivere `grammar` in forma LL(1) preservando il linguaggio.

    Parametri
    ---------
    grammar : dict  { NT: [production_list, ...] }
        La grammatica in forma classica costruita da parsing_table() a
        partire da final_rules.
    start_symbol : str
        Simbolo iniziale (per convenzione 'S*').
    max_rounds : int
        Limite di round del ciclo trasforma-e-ricontrolla.

    Ritorna
    -------
    dict | None
        La grammatica riscritta e priva di conflitti, oppure None se non si
        è raggiunta una forma LL(1) entro i limiti.  In quel caso il
        chiamante lascia risalire il conflitto originale.
    """
    grammar = normalise_epsilon(grammar)
    budget = max(len(grammar) * MAX_TRANSFORM_GROWTH, len(grammar) + 16)
    used = set(grammar)

    grammar = left_factor(grammar, used)

    for rnd in range(max_rounds):
        conflicts = find_conflicts(grammar, start_symbol)
        if not conflicts:
            logging.info(
                f"LL(1) auto-transform: conflitti risolti in {rnd} round, "
                f"{len(grammar)} non-terminali."
            )
            return grammar

        if len(grammar) > budget:
            logging.info(
                "LL(1) auto-transform: superato il budget di crescita "
                f"({len(grammar)} NT > {budget}), rinuncio."
            )
            return None

        nt, terminal, competing = conflicts[0]
        logging.info(
            f"LL(1) auto-transform round {rnd}: conflitto su {nt} "
            f"con lookahead {terminal!r} fra {competing}"
        )

        # I conflitti FIRST/FOLLOW vanno affrontati per primi: finché la
        # continuazione non è inlineata le alternative in competizione stanno
        # in non-terminali diversi e nulla di locale le raggiunge.
        step = None
        for c_nt, _, c_competing in conflicts:
            if any(not p for p in c_competing):
                step = specialize_deep(grammar, c_nt, used, start_symbol)
                if step is not None:
                    break
        if step is None:
            for c_nt, _, c_competing in conflicts:
                step = inline_heads(grammar, c_nt, c_competing, start_symbol)
                if step is not None:
                    break
        if step is None:
            logging.info("LL(1) auto-transform: nessuna trasformazione applicabile.")
            return None

        grammar = left_factor(step, used)

    logging.info(f"LL(1) auto-transform: non convergente in {max_rounds} round.")
    return None


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

    # Una sola rappresentazione dell'epsilon da qui in poi.  process_full_grammar
    # emette l'alternativa epsilon come ['ε'], mentre la tabella e il PDA usano
    # [];  compute_all_first_sets accettava entrambe le forme con un ramo
    # dedicato, ma compute_first_of_string trattava 'ε' come un terminale
    # qualsiasi e dava la risposta giusta solo perché il marcatore coincide con
    # il nome del simbolo.  Canonicalizzare qui rende quella coincidenza
    # irrilevante ed elimina un 'ε' residuo dentro produzioni più lunghe, che
    # sarebbe finito nella tabella e avrebbe fatto fallire il PDA come terminale
    # inesistente.  Sulla costruzione della tabella è un no-op: ['ε'] e [] danno
    # gli stessi FIRST, FOLLOW e celle.
    grammar = normalise_epsilon(dict(grammar))

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
            table[key].pop(EOF_MARKER, None)
        return table

    logging.info("\nProcessed grammar:\n")
    logging.info(final_rules)

    first_sets = compute_all_first_sets(grammar)
    logging.info("\nFirst sets:\n")
    logging.info(first_sets)

    follow_sets = follow(grammar, first_sets, 'S*')
    logging.info("\nFollow sets:\n")
    logging.info(follow_sets)

    # La grammatica in ingresso può non essere LL(1): metterla in forma LL(1)
    # è parte della costruzione della tabella, non una riparazione a
    # posteriori.  Il controllo riusa i FIRST/FOLLOW già calcolati, quindi
    # costa una sola passata sulle produzioni.
    if find_conflicts(grammar, 'S*', first_sets, follow_sets):
        repaired = ll1ify(grammar, 'S*')
        if repaired is not None:
            logging.info(
                f"Grammatica messa in forma LL(1): "
                f"{len(grammar)} → {len(repaired)} non-terminali."
            )
            grammar = repaired
            first_sets = compute_all_first_sets(grammar)
            follow_sets = follow(grammar, first_sets, 'S*')
        # Se la riscrittura non converge la grammatica resta com'è e
        # compute_parsing_table solleva il suo ValueError diagnostico.

    table = compute_parsing_table(grammar, first_sets, follow_sets)

    save_table_parsing_as_txt(table)

    return table