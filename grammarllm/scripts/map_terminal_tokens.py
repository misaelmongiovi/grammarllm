"""
map_terminal_tokens.py
=======================
Costruisce la mappa terminale → lista di token ID del vocabolario,
che è il collegamento tra la grammatica astratta e il tokenizer concreto.

Ruolo nel sistema
-----------------
La grammatica definita dall'utente usa stringhe come terminali
(es. 'positive', 'Ġhappy', oppure pattern regex come 'integer_token').
Il modello linguistico lavora con interi (token ID). Questo modulo
costituisce il ponte tra i due mondi.

Pipeline di integrazione
------------------------
    generate_LL1_parsing_table.py
        └─► parsing_table (tabella LL(1) con terminali stringa)
              └─► generate_token_maps(tokenizer, table_parsing)  [questo modulo]
                    └─► map_terminal_tokens: { terminale: [token_id, ...] }
                          └─► PushdownAutomaton(grammar, 'S*', map_terminal_tokens)
                                └─► StatelessLogitsProcessor (usa map per il masking)

Due tipi di terminali gestiti
------------------------------
1. Terminali esatti (<<exact string>> nella grammatica dell'utente):
   Vengono mappati tramite regex esatta ^{re.escape(terminal)}$.
   Un solo token ID se il terminale è un singolo token del vocabolario,
   lista vuota (con warning) se non corrisponde a nessun token.

2. Terminali regex (regex_* nel regex_dict):
   Mappati tramite pattern regex applicato a ogni stringa del vocabolario.
   Permettono di coprire insiemi aperti di token (es. tutti i numeri,
   tutte le parole alfabetiche).
"""

import re
import itertools
import logging

try:
    from ..modules.lookahead import REGEX_TERMINALS_KEY
except ImportError:                      # bare-import test path
    from lookahead import REGEX_TERMINALS_KEY


def generate_token_maps(tokenizer, table_parsing, regex_dict=None):
    """
    Costruisce la mappa { terminale: [token_id, ...] } per tutti i terminali
    della parsing table.

    Questa mappa è l'unico artefatto prodotto da questo modulo ed è
    utilizzata in due modi complementari:

    1. Da PushdownAutomaton.__init__(): viene invertita in
       map_tokens_terminals = { token_id: [terminale, ...] }
       per permettere a next_state() di identificare quale terminale
       corrisponde al token generato dal modello.

    2. Da PushdownAutomaton.get_tokens(): per ogni terminale valido
       restituito da recursive_get_tokens(), recupera i token ID
       corrispondenti da includere nella maschera dei logit.

    Algoritmo
    ---------
    Step 1 (se regex_dict fornito): per ogni entry 'regex_X' in regex_dict,
      aggiunge X → [token_id per ogni token nel vocabolario che matcha il pattern].
      Questo gestisce terminali aperti come 'integer_token' o 'string_token'.

    Step 2: per ogni terminale nella parsing table che non è un NT e non è
      già in map (non coperto da regex), aggiunge una entry con match esatto
      tramite regex ^{re.escape(terminal)}$.
      Se nessun token matcha: emette un warning (invece di errore silenzioso)
      per facilitare il debug di terminali malformati o non presenti nel vocab.

    Step 3: verifica i conflitti intra-riga della parsing table.
      Due terminali nella stessa riga (stesso NT lookahead) non possono
      mappare agli stessi token ID — questo violerebbe l'invariante LL(1)
      a livello di token.

    Nota sulla copertura dei conflitti
    -----------------------------------
    La verifica è intra-riga (stessa NT). I conflitti cross-NT vengono
    rilevati a runtime da get_tokens() in automaton.py tramite il check
    sulla disgiunzione dei set. La verifica statica qui è quindi
    necessaria ma non sufficiente: costituisce un early-warning per i
    casi più comuni.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer HuggingFace. Viene chiamato get_vocab() per ottenere
        il dizionario { token_string: token_id }.
    table_parsing : dict
        Tabella di parsing LL(1) prodotta da generate_LL1_parsing_table.py.
        Forma: { NT: { lookahead_terminal: production_list } }
    regex_dict : dict[str, re.Pattern], optional
        Mappa di pattern regex per terminali aperti.
        Le chiavi devono avere il formato 'regex_<nome_terminale>' dove
        <nome_terminale> corrisponde esattamente al simbolo terminale usato
        nelle produzioni della grammatica.
        Esempio: {'regex_integer_token': re.compile(r'\\d+')}

    Returns
    -------
    dict[str, list[int]]
        Mappa terminale → lista di token ID del vocabolario.

    Raises
    ------
    ValueError
        Se due terminali nella stessa riga della parsing table mappano agli
        stessi token ID (conflitto LL(1) a livello di tokenizer).
    """

    def check_tokens_conflicts(table_parsing, map_terminal_tokens):
        """
        Verifica che i token ID associati ai terminali in ogni riga della
        parsing table siano disgiunti.

        Logica
        ------
        Per ogni NT, controlla tutte le coppie di terminali lookahead nella
        sua riga. Se due terminali (es. 'positive' e 'pos') mappano agli
        stessi token ID, il PDA non può distinguerli e la grammatica non è
        deterministicamente parsabile con 1 token di lookahead.

        Nota: il check è intra-riga. I conflitti tra terminali di NT diversi
        (cross-row) non sono rilevati qui ma lo sono a runtime da get_tokens().

        Returns
        -------
        list[str] | None
            Lista di descrizioni dei conflitti trovati, o None se nessuno.
        """
        conflicts = []
        for lhs, rhs_list in table_parsing.items():
            for a, b in itertools.combinations(rhs_list.keys(), 2):
                # .get(): a lookahead key may legitimately be missing from the
                # map (e.g. a regex terminal whose regex_dict entry was not
                # provided) — the missing-terminal warning in Step 2 already
                # covers it; don't crash the conflict check with a KeyError.
                intersection = set(map_terminal_tokens.get(a, ())) & set(map_terminal_tokens.get(b, ()))
                if intersection:
                    logging.info(f"Conflitto tra '{a}' e '{b}': {intersection}")
                    conflicts.append(
                        f"I set di tokens associati ai terminali '{a}' e '{b}' "
                        f"non sono disgiunti. Intersezione: {list(intersection)[:5]}"
                    )
        if conflicts:
            logging.error("Conflitti trovati tra i token associati ai terminali:")
            for conflict in conflicts:
                logging.error(conflict)
            return conflicts

    map_terminal_tokens = {}
    vocab = tokenizer.get_vocab()
    non_terminal_keys = set(table_parsing.keys())

    # Step 1: terminali regex (insiemi aperti di token)
    if regex_dict:
        map_terminal_tokens = {
            name[6:]: [
                token_id
                for token_str, token_id in vocab.items()
                if regex.match(token_str)
            ]
            for name, regex in regex_dict.items()
            if name.startswith('regex_')
        }

    # Step 2: terminali esatti (match preciso con il token nel vocabolario)
    for lhs, rhs_list in table_parsing.items():
        for terminals in rhs_list.values():
            filtered_terminals = [t for t in terminals if t not in non_terminal_keys]
            for terminal in filtered_terminals:
                if terminal not in map_terminal_tokens:
                    regex = re.compile(rf"^{re.escape(terminal)}$")
                    matched = [
                        token_id
                        for token_str, token_id in vocab.items()
                        if regex.match(token_str)
                    ]
                    if not matched:
                        logging.warning(
                            f"Terminal '{terminal}' has no matching token IDs in the vocabulary. "
                            f"Check that the terminal string exactly matches a tokenizer token."
                        )
                    map_terminal_tokens[terminal] = matched

    # Step 3: verifica conflitti
    conflicts = check_tokens_conflicts(table_parsing, map_terminal_tokens)
    if conflicts:
        raise ValueError("\n".join(conflicts))

    # Metadata channel for the lookahead engine: which terminal names are
    # open regex classes (DFS must not spell them character-by-character).
    map_terminal_tokens[REGEX_TERMINALS_KEY] = (
        sorted(name[len("regex_"):] for name in regex_dict) if regex_dict else []
    )

    return map_terminal_tokens