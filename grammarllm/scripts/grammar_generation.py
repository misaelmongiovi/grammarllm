"""
grammar_generation.py
=====================
Converte le produzioni definite dall'utente (con notazione <<exact_string>>)
in una grammatica LL(1) formale, pronta per parsing_table().

Posizione nella pipeline
------------------------
È il primo passo della fase di setup:

    Utente definisce productions dict
              ↓
    ProductionRuleProcessor.process_full_grammar()
              ↓
    final_grammar  { (NT, 'RULE'): [production_list, ...] }
              ↓
    parsing_table()  →  generate_token_maps()  →  PushdownAutomaton

Notazione utente vs grammatica interna
---------------------------------------
L'utente scrive produzioni come:
    'S*': ["<<positive>> A", "<<negative>> B"]

Il processore:
  1. Estrae i tag <<exact_string>> e i simboli "other" (NT o terminali letterali).
  2. Tokenizza ogni tag con il tokenizer HuggingFace per gestire la
     subword tokenization (es. "positive" → ["pos", "itive"]).
  3. Raggruppa i tag per prefisso comune per costruire una grammatica LL(1)
     senza ambiguità (prefix-grouping / LL(prefix) → LL(1)).
  4. Genera NT ausiliari ({lhs}_TAG_NT{i}, {lhs}_POS{pos}, ecc.) per
     rappresentare le scelte alternative in modo deterministico.
  5. Applica left-factorization se più produzioni condividono un prefisso comune.

Problema della multi-subword tokenization
------------------------------------------
Una stringa come " gloomy" viene tokenizzata come ["Ġglo", "omy"].
La grammatica deve quindi gestire la sequenza di sottotokens, non la stringa
intera.  Il prefix-grouping permette di condividere il prefisso "Ġglo" tra
alternative che lo condividono, evitando ambiguità LL(1).
"""

import re
import logging
import os

class ProductionRuleProcessor:
    """
    Converte la grammatica LL(prefix) definita dall'utente in una grammatica
    LL(1) formale pronta per la costruzione della parsing table.

    La grammatica dell'utente usa la notazione <<exact string>> per i terminali
    letterali e simboli UPPERCASE per i non-terminali. Questo processore:
    1. Tokenizza ogni <<tag>> usando il tokenizer HuggingFace.
    2. Raggruppa i tag per prefisso comune (LL(prefix) → LL(1)).
    3. Assegna un NT posizionale a ogni posizione-tag in ogni produzione.
    4. Applica left-factorization se più produzioni condividono un prefisso.

    Integrazione
    ------------
    Istanziato e chiamato da generate_with_constraints.get_parsing_table_and_map_tt():
        processor = ProductionRuleProcessor(tokenizer=tokenizer)
        final_grammar, tag_mapping = processor.process_full_grammar(productions)
    Il final_grammar viene poi passato a parsing_table() in generate_LL1_parsing_table.py.

    Stato interno rilevante
    -----------------------
    tag_to_nt_mapping : dict[str, str]
        Mappa "{lhs}::{tag}::pos{n}" → stringa di simboli (es. "tok1 NT_aux").
        Costruita durante process_full_grammar() e usata per assemblare le
        produzioni finali in passo 3. È l'unico stato che sopravvive
        tra le chiamate ai metodi interni.
    rule_specific_grammars : dict[str, dict]
        Cache delle sub-grammatiche costruite per ogni NT posizionale.
        Usata per debug e ispezione.
    """

    def __init__(self, tokenizer=None):
        """
        Inizializza il processore.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer, optional
            Tokenizer HuggingFace. Se None, ogni <<tag>> viene trattato come
            un singolo token (fallback per test senza modello).
        """
        self.nt_counter = 0
        self.sub_nt_counter = {}
        self.tag_to_nt_mapping = {}  # Mappa dai tag originali ai NT creati
        self.original_rules_mapping = {}  # Mappa per mantenere il collegamento con le regole originali
        self.tokenizer = tokenizer  # Tokenizer di Hugging Face
        self.non_terminals = set()  # Traccia tutti i non terminali
        self.rule_specific_grammars = {}  # Grammatiche specifiche per ogni regola
        self.generated_nts = set()  # NT ausiliari creati dalla fattorizzazione
    
    def extract_tags_and_others(self, rhs_list):
        """
        Analizza le produzioni utente e separa i tag <<...>> dagli altri simboli.

        Ogni produzione è una stringa come "<<positive>> A <<suffix>>".
        Questa funzione la decompone in una lista ordinata di coppie (tipo, valore):
            [("tag", "positive"), ("other", "A"), ("tag", "suffix")]

        L'ordine è preservato: è fondamentale per la successiva assegnazione
        posizionale dei tag in process_full_grammar() (passo 1).

        Integrazione
        ------------
        Chiamata da process_full_grammar() come primo step per ogni regola lhs.
        Il risultato viene usato sia per estrarre i tag unici (build_tag_grammar)
        sia per assemblare le produzioni finali (passo 3).

        Parameters
        ----------
        rhs_list : list[str]
            Lista delle produzioni alternative per un NT, nella notazione utente.

        Returns
        -------
        list[list[tuple[str, str]]]
            Una lista (una per produzione alternativa) di liste di (tipo, valore).
        """
        tag_pattern = re.compile(r'<<(.+?)>>')
        result = []

        for item in rhs_list:
            matches = list(tag_pattern.finditer(item))
            last_index = 0
            parts = []

            for match in matches:
                # Testo prima del tag
                pre_text = item[last_index:match.start()]
                if pre_text.strip():
                    for word in pre_text.strip().split():
                        parts.append(("other", word))

                # Il tag stesso
                parts.append(("tag", match.group(1)))
                last_index = match.end()

            # Eventuale testo dopo l'ultimo tag
            post_text = item[last_index:]
            if post_text.strip():
                for word in post_text.strip().split():
                    parts.append(("other", word))

            result.append(parts)

        return result

    def tokenize_tag(self, tag):
        """
        Tokenizza una stringa esatta usando il tokenizer HuggingFace.

        Il risultato determina come il tag viene rappresentato nella grammatica:
        - Tag con un solo token → nessun NT ausiliario necessario.
        - Tag con più token → NT ausiliario per sequenziare i token.
        - Tag con zero token → ValueError immediato (tag non valido).

        Integrazione
        ------------
        Chiamata da get_prefix_groups_for_rule() per tokenizzare ogni tag,
        e da create_initial_grammar_for_rule() per i tag non raggruppati.

        Parameters
        ----------
        tag : str
            Stringa interna al delimitatore <<...>> (es. " happy", "positive").

        Returns
        -------
        list[str]
            Lista di sotto-token (es. ['Ġhappy'] o ['Ġun','em','otional']).
        """
        if self.tokenizer is None:
            # Fallback alla tokenizzazione semplice se non c'è tokenizer
            logging.info("ATTENZIONE: Nessun tokenizer fornito, uso tokenizzazione semplice")
            return [tag]  # Restituisce il tag come singolo token
        
        # Usa il tokenizer di Hugging Face
        tokens = self.tokenizer.tokenize(tag)
        return tokens


    def get_prefix_groups_for_rule(self, tags, rule_name):
        """
        Raggruppa i tag per primo sotto-token condiviso (prefix grouping).

        Implementa il cuore della conversione LL(prefix) → LL(1): tag che
        condividono il primo token di tokenizzazione vengono raggruppati
        insieme. Il prefix comune viene estratto come lookahead condiviso
        e i suffissi diventano alternative di un NT ausiliario.

        Esempio
        -------
        tags = ["positive", "possible"]  (entrambi iniziano con "pos")
        tokenized: positive → ["pos","itive"], possible → ["pos","sible"]
        → prefix_groups = {"pos": [("positive", ["itive"]), ("possible", ["sible"])]}
        → ungrouped_tags = []

        tags = ["happy", "sad"]  (primo token diverso)
        → prefix_groups = {}
        → ungrouped_tags = ["happy", "sad"]

        Determinismo
        ------------
        prefix_groups e ungrouped_tags vengono sortati per garantire che
        grammatiche identiche scritte in ordine diverso producano la stessa
        struttura interna (riproducibilità).

        Raises
        ------
        ValueError
            Se un tag tokenizza alla lista vuota (FIX-A: prima era silenzioso).

        Returns
        -------
        tuple[dict, list]
            (prefix_groups, ungrouped_tags) dove:
            - prefix_groups: { primo_token: [(tag, suffisso), ...] }
            - ungrouped_tags: tag senza prefisso condiviso con altri
        """
        tokenized = {tag: self.tokenize_tag(tag) for tag in tags if tag}

        # FIX A: a tag that tokenizes to [] is silently dropped from both
        # prefix_groups and ungrouped_tags, leaving it absent from
        # tag_to_nt_mapping.  process_full_grammar then falls back to the raw
        # <<tag>> literal which the LL(1) table builder cannot handle, producing
        # a confusing KeyError deep inside the PDA at generation time.
        # Raise immediately with an actionable message instead.
        for tag, tokens in tokenized.items():
            if not tokens:
                raise ValueError(
                    f"Tag '<<{tag}>>' in rule '{rule_name}' produces an empty "
                    f"token list from the tokenizer. "
                    f"This usually means the tag string contains only whitespace, "
                    f"BOS/EOS markers, or characters the tokenizer strips entirely. "
                    f"Use a tag string that maps to at least one vocabulary token."
                )

        # Trova i prefissi comuni tra più parole
        prefix_counts = {}
        for tag, tokens in tokenized.items():
            prefix = tokens[0]
            prefix_counts.setdefault(prefix, []).append((tag, tokens[1:]))

        # Prefissi condivisi da almeno 2 tag
        prefix_groups = {prefix: items for prefix, items in prefix_counts.items() if len(items) > 1}

        # FIX: sort both the groups dict and each group's items so that the
        # generated grammar is deterministic regardless of Python dict insertion
        # order and regardless of the order in which the user lists productions.
        # Without this, identical grammars written in different orders produced
        # structurally different parsing tables — a reproducibility hazard.
        prefix_groups = dict(sorted(prefix_groups.items()))
        for prefix in prefix_groups:
            prefix_groups[prefix] = sorted(prefix_groups[prefix], key=lambda x: x[0])

        # Tag non inclusi nei gruppi — sorted for the same reason
        grouped_tags = {tag for group in prefix_groups.values() for tag, _ in group}
        ungrouped_tags = sorted([tag for tag in tags if tag not in grouped_tags])

        return prefix_groups, ungrouped_tags
    
    def create_initial_grammar_for_rule(self, prefix_groups, ungrouped_tags, rule_name):
        """
        Costruisce la sub-grammatica LL(prefix) per i tag di una singola posizione.

        Questa funzione ha due responsabilità distinte che vale la pena rendere
        esplicite:

        1. Popola self.tag_to_nt_mapping con la chiave "{rule_name}::{tag}"
           per ogni tag.  Il valore è la stringa di simboli (token + NT) che
           rappresenta quel tag nella produzione padre — può essere un singolo
           token ("pos") oppure "tok1 NT_per_suffisso".

        2. Restituisce un dict con le regole della sub-grammatica per i NT
           ausiliari creati per gestire i suffissi condivisi.  NON include la
           regola "radice" del gruppo: quella viene assemblata da process_full_grammar
           usando tag_to_nt_mapping, non guardando il dict restituito.

        FIX B: la variabile start_productions era costruita ma poi ignorata
        (la riga grammar[rule_name] = start_productions era commentata).
        Era codice morto che suggeriva un'invariante rotta.  Rimossa per
        chiarezza; il commento sopra documenta perché è corretto non includerla.
        """
        grammar = {}

        # Gestione dei gruppi con prefissi condivisi
        for i, (prefix, tag_suffix_pairs) in enumerate(prefix_groups.items(), 1):
            if all(len(suffix) == 0 for _, suffix in tag_suffix_pairs):
                # Tutti i tag sono singoli token: il loro "expansion" è il
                # prefisso stesso — nessun NT ausiliario necessario.
                # FIX D: se due tag diversi hanno la stessa tokenizzazione
                # (stesso singolo token), uno diventa irraggiungibile nella
                # grammatica.  Avvisiamo esplicitamente.
                seen_prefix_tags = []
                for tag, _ in tag_suffix_pairs:
                    if seen_prefix_tags:
                        logging.warning(
                            f"Rule '{rule_name}': tags {seen_prefix_tags + [tag]} all "
                            f"tokenize to the single token '{prefix}'. "
                            f"They map to identical terminals — only one will ever be "
                            f"generated. Consider using distinct tag strings."
                        )
                    seen_prefix_tags.append(tag)
                    self.tag_to_nt_mapping[f"{rule_name}::{tag}"] = prefix
            else:
                # Crea un NT ausiliario per i suffissi
                nt = f"{rule_name}_TAG_NT{i}"
                suffixes = [suffix for _, suffix in tag_suffix_pairs]
                grammar[(nt, prefix)] = suffixes

                for tag, _ in tag_suffix_pairs:
                    self.tag_to_nt_mapping[f"{rule_name}::{tag}"] = f"{prefix} {nt}"

        # Gestione dei tag non raggruppati (tokenizzati completamente)
        for tag in ungrouped_tags:
            tokens = self.tokenize_tag(tag)
            production = " ".join(tokens)
            self.tag_to_nt_mapping[f"{rule_name}::{tag}"] = production

        return grammar


    def find_common_prefixes(self, token_lists):
        """
        Trova i primi token condivisi tra più liste di suffissi.

        Usata da process_grammar_iteration() per raffinare iterativamente
        la sub-grammatica dei suffissi: se due suffissi condividono il primo
        token, vengono ulteriormente fattorizzati creando un nuovo NT.

        Esempio
        -------
        token_lists = [["itive"], ["ible"]]  (suffissi di "pos")
        → common_prefixes = {}  (nessun prefisso comune)

        token_lists = [["a","x"], ["a","y"]]
        → common_prefixes = {"a": [["x"], ["y"]]}

        Parameters
        ----------
        token_lists : list[list[str]]

        Returns
        -------
        dict[str, list[list[str]]]
            { primo_token_comune: [suffissi_rimanenti, ...] }
            Vuoto se nessuna coppia condivide il primo token.
        """
        if len(token_lists) <= 1:
            return {}
        
        prefix_groups = {}
        for tokens in token_lists:
            if len(tokens) > 0:
                first_token = tokens[0]
                if first_token not in prefix_groups:
                    prefix_groups[first_token] = []
                prefix_groups[first_token].append(tokens[1:])
        
        common_prefixes = {k: v for k, v in prefix_groups.items() if len(v) > 1}
        logging.info(f"Prefissi comuni trovati: {common_prefixes}")
        return common_prefixes
    
    def get_next_sub_nt(self, parent_nt):
        """
        Genera un nome univoco per il prossimo NT ausiliario figlio di parent_nt.

        Usa un contatore per produrre nomi come S*_POS0_TAG_NT1_1,
        S*_POS0_TAG_NT1_2, ecc. Il contatore è separato per ogni NT padre.

        Integrazione
        ------------
        Chiamata da process_grammar_iteration() ogni volta che viene creato
        un nuovo NT per fattorizzare un prefisso comune tra suffissi.

        Returns
        -------
        str
            Nome del nuovo NT (es. "S*_POS0_TAG_NT1_1").
        """
        if parent_nt not in self.sub_nt_counter:
            self.sub_nt_counter[parent_nt] = 0
        self.sub_nt_counter[parent_nt] += 1
        return f"{parent_nt}_{self.sub_nt_counter[parent_nt]}"
    
    def process_grammar_iteration(self, grammar):
        """
        Esegue un'iterazione di raffinamento sulla sub-grammatica dei suffissi.

        Per ogni NT ausiliario (chiave tuple) nella grammatica, cerca prefissi
        comuni tra i suoi suffissi alternativi. Se ne trova, crea un nuovo NT
        ausiliario per quel prefisso e aggiorna la grammatica.

        Il ciclo in build_tag_grammar_for_rule() chiama questo metodo
        ripetutamente finché `changed` è False (punto fisso).

        Esempio di raffinamento iterativo
        -----------------------------------
        Iterazione 1: NT_aux ha suffissi [["a","x"], ["a","y"], ["b"]]
          → trova prefisso comune "a" tra prime due
          → crea NT_aux_1 con produzioni [["x"], ["y"]]
          → NT_aux diventa [["a", NT_aux_1], ["b"]]
          → changed = True

        Iterazione 2: NT_aux_1 ha [["x"], ["y"]] — nessun prefisso comune
          → changed = False → stop

        Deep copy (FIX-C)
        -----------------
        Usa deepcopy per evitare aliasing tra le liste di suffissi originali
        e quelle del nuovo NT. Prima usava grammar.copy() (shallow), che
        causava corruzione silenziosa se le sottoliste venivano mutate.

        Returns
        -------
        tuple[dict, bool]
            (new_grammar, changed) dove changed indica se sono state fatte modifiche.
        """
        # FIX C: grammar.copy() is a shallow dict copy — the values (lists of
        # token-lists) are shared references.  If new_token_lists reuses
        # sub-lists from token_lists directly and a future caller mutates them,
        # both old and new grammar entries would be corrupted silently.
        # Use a deep copy so every iteration works on fully independent data.
        import copy as _copy
        new_grammar = _copy.deepcopy(grammar)
        changed = False

        for key, token_lists in list(grammar.items()):
            if not isinstance(key, tuple):
                continue  # Salta produzioni come la regola iniziale

            nt, prefix = key
            if len(token_lists) > 1:
                common_prefixes = self.find_common_prefixes(token_lists)

                if common_prefixes:
                    new_token_lists = []

                    for common_prefix, suffixes in common_prefixes.items():
                        new_nt = self.get_next_sub_nt(nt)
                        new_token_lists.append([common_prefix, new_nt])
                        # Store a deep copy of suffixes so the new NT's
                        # productions are fully independent from the originals.
                        new_grammar[(new_nt, common_prefix)] = _copy.deepcopy(suffixes)
                        changed = True

                    for tokens in token_lists:
                        if len(tokens) > 0 and tokens[0] not in common_prefixes:
                            new_token_lists.append(_copy.deepcopy(tokens))
                        elif len(tokens) == 0:
                            new_token_lists.append([])

                    new_grammar[(nt, prefix)] = new_token_lists

        return new_grammar, changed

    
    def build_tag_grammar_for_rule(self, tags, rule_name):
        """
        Costruisce la sub-grammatica LL(1) per un insieme di tag alternativi
        a una specifica posizione di una produzione.

        Questa funzione è il nucleo della conversione LL(prefix) → LL(1) per
        un singolo gruppo di tag. Opera in 4 step:

        Step 1 — Tokenizzazione
            Ogni tag viene tokenizzato con il tokenizer HuggingFace.

        Step 2 — Raggruppamento per prefisso (get_prefix_groups_for_rule)
            Tag che condividono il primo sotto-token vengono raggruppati.

        Step 3 — Grammatica iniziale (create_initial_grammar_for_rule)
            Per i gruppi: crea NT ausiliari per i suffissi condivisi.
            Per gli ungrouped: registra il tag direttamente in tag_to_nt_mapping.

        Step 4 — Raffinamento iterativo (process_grammar_iteration)
            Applica left-factorization ricorsiva sui suffissi finché
            non ci sono più prefissi comuni (punto fisso).

        Integrazione
        ------------
        Chiamata da process_full_grammar() passo 2, una volta per ogni
        NT posizionale (es. S*_POS0, S*_POS1, A_POS0, ...).
        Il dizionario restituito viene mergiato in final_grammar.

        Side effects
        ------------
        Aggiorna self.tag_to_nt_mapping e self.rule_specific_grammars.

        Parameters
        ----------
        tags : list[str]
            Tag alternativi alla posizione corrente (già deduplicati).
        rule_name : str
            Nome del NT posizionale (es. "S*_POS0").

        Returns
        -------
        dict
            Sub-grammatica con chiavi tuple (NT_aux, prefix) e valori
            liste di produzioni (liste di suffissi).
        """
        logging.info(f"=== COSTRUZIONE GRAMMATICA PER I TAG DELLA REGOLA {rule_name} ===")
        
        # Filtra tag vuoti e None
        valid_tags = [tag for tag in tags if tag and tag.strip()]
        
        if not valid_tags:
            return {}
        
        logging.info(f"Tag da processare per {rule_name}: {valid_tags}")
        
        logging.info(f"\n=== STEP 1: Tokenizzazione dei tag per {rule_name} ===")
        for tag in valid_tags:
            tokens = self.tokenize_tag(tag)
            logging.info(f"'{tag}' -> {tokens}")
        
        logging.info(f"\n=== STEP 2: Raggruppamento per prefisso per {rule_name} ===")
        #prefix_groups = self.get_prefix_groups_for_rule(valid_tags, rule_name)
        prefix_groups, ungrouped_tags = self.get_prefix_groups_for_rule(valid_tags, rule_name)
        for prefix, tag_suffix_pairs in prefix_groups.items():
            logging.info(f"Prefisso '{prefix}': {[(tag, suffix) for tag, suffix in tag_suffix_pairs]}")
        
        logging.info(f"\n=== STEP 3: Creazione grammatica iniziale per {rule_name} ===")
        grammar = self.create_initial_grammar_for_rule(prefix_groups, ungrouped_tags, rule_name)
        
        logging.info(f"\n=== STEP 4: Iterazioni di raffinamento per {rule_name} ===")
        iteration = 1
        while True:
            logging.info(f"\n--- Iterazione {iteration} per {rule_name} ---")
            new_grammar, changed = self.process_grammar_iteration(grammar)
            
            if not changed:
                logging.info(f"Nessuna modifica per {rule_name}, algoritmo terminato.")
                break
            
            grammar = new_grammar
            logging.info(f"Grammatica aggiornata per {rule_name}:")
            iteration += 1
        
        # Salva la grammatica specifica per questa regola
        self.rule_specific_grammars[rule_name] = grammar
        return grammar
    
    @staticmethod
    def _longest_common_prefix(productions):
        """Il prefisso più lungo condiviso da TUTTE le produzioni date."""
        prefix = []
        for i in range(min(len(p) for p in productions)):
            column = {p[i] for p in productions}
            if len(column) != 1:
                break
            prefix.append(productions[0][i])
        return prefix

    def _fresh_nt(self, base):
        """Nome per un NT ausiliario che non collide con nulla già in uso."""
        name, n = base, 1
        while name in self.non_terminals or name in self.generated_nts:
            n += 1
            name = f"{base}{n}"
        self.generated_nts.add(name)
        return name

    def left_factor_productions(self, lhs, productions):
        """
        Fattorizza a sinistra le alternative di un non-terminale, PER GRUPPO.

        Raggruppa le alternative per primo simbolo e fattorizza ogni gruppo
        di due o più elementi usando il prefisso comune PIÙ LUNGO di quel
        gruppo, ricorrendo poi sui suffissi rimasti (che a loro volta
        possono condividere un prefisso più corto in un sotto-gruppo):

            A -> a b X | a b Y | a c Z | d W
              => A           -> a A_FACT | d W
                 A_FACT      -> b A_FACT_FACT | c Z
                 A_FACT_FACT -> X | Y

        BUG FIX (fattorizzazione tutto-o-niente)
        -----------------------------------------
        L'implementazione precedente calcolava UN solo prefisso comune a
        TUTTE le alternative non-epsilon e rinunciava se anche una sola non
        lo condivideva.  Il caso da manuale `A -> a X | a Y | b Z` restava
        quindi intatto e il costruttore della parsing table rifiutava una
        grammatica banalmente LL(1)-izzabile:

            Conflict: S* → a ['a', 'X_NT']!
            Regola attuale: a ['a', 'Y_NT']!

        Trattamento delle epsilon (invariato, ed essenziale)
        -----------------------------------------------------
        Una epsilon INDIPENDENTE (prod == []) non partecipa mai al calcolo
        del prefisso e resta sul padre.  Il suo lookahead è FOLLOW(A), che
        per la proprietà LL(1) è disgiunto dai FIRST delle alternative
        prefissate: non può condividere il prefisso.

            A -> 'x' B | 'x' | ε
              => A      -> 'x' A_FACT | ε
                 A_FACT -> B | ε      ← questa ε nasce dallo strip di 'x',
                                        è cosa diversa dalla ε propria di A

        Terminazione
        ------------
        Il prefisso di un gruppo fattorizzato è lungo almeno un simbolo (il
        gruppo è formato proprio sul primo simbolo condiviso), quindi ogni
        chiamata ricorsiva lavora su produzioni strettamente più corte.

        Parametri
        ---------
        lhs : str
            Non-terminale da fattorizzare; i NT ausiliari prendono il nome
            da lui.
        productions : list[list[str]]
            Le sue alternative.  Accetta anche la vecchia forma a stringa
            ("ε" oppure "a b c").

        Ritorna
        -------
        tuple[list[list[str]], dict[str, list[list[str]]]]
            (produzioni da tenere su lhs, {NT ausiliario: sue produzioni}).

        Collegamento
        ------------
        Chiamata da process_full_grammar() al passo 4.  Le regole ausiliarie
        restituite vengono inserite in final_grammar accanto a quella di lhs.
        """
        # Normalizza e rimuove i duplicati: un'alternativa ripetuta si
        # ridurrebbe a una epsilon ripetuta dentro il NT ausiliario, che il
        # costruttore della parsing table segnala come conflitto spurio.
        normalised = []
        for prod in productions:
            if isinstance(prod, list):
                p = list(prod)
            elif prod == "ε":
                p = []
            else:
                p = prod.split()
            if p not in normalised:
                normalised.append(p)

        epsilons = [p for p in normalised if not p]
        groups = {}
        for p in normalised:
            if p:
                groups.setdefault(p[0], []).append(p)

        main, extra = [], {}
        for group in groups.values():
            if len(group) == 1:
                main.append(group[0])
                continue

            new_nt = self._fresh_nt(f"{lhs}_FACT")
            prefix = self._longest_common_prefix(group)
            suffixes = [p[len(prefix):] for p in group]

            main.append(prefix + [new_nt])
            sub_main, sub_extra = self.left_factor_productions(new_nt, suffixes)
            extra[new_nt] = sub_main
            extra.update(sub_extra)

        main.extend(epsilons)
        return main, extra


    def create_final_productions_for_rule(self, lhs, ordered_elements_list, rule_specific_tag_grammar):
        """
        Assembla le produzioni finali per un NT usando tag_to_nt_mapping.

        Per ogni produzione alternativa (ordered_elements), sostituisce ogni
        tag con la sua espansione da tag_to_nt_mapping (che può essere un
        singolo token o una sequenza "tok NT_aux"), e mantiene gli "other"
        (NT utente come A, B, C) invariati.

        Nota: questa funzione era usata dal vecchio approccio (pool globale
        di tag). Nel nuovo approccio (NT posizionali), le produzioni finali
        vengono assemblate direttamente in process_full_grammar() passo 3
        usando le chiavi posizionali lhs::tag::pos{n}.
        Questa funzione rimane per compatibilità ma non è più nel percorso
        principale di esecuzione.

        Parameters
        ----------
        lhs : str
            NT sinistro della regola.
        ordered_elements_list : list[list[tuple]]
            Output di extract_tags_and_others().
        rule_specific_tag_grammar : dict
            Sub-grammatica dei tag (non usata direttamente qui).

        Returns
        -------
        list[list[str]]
            Lista di produzioni, ognuna come lista di simboli.
        """
        logging.info(f"\n=== PROCESSAMENTO REGOLA: {lhs} ===")

        productions = []

        for ordered_elements in ordered_elements_list:
            production_sublist = []

            for kind, value in ordered_elements:
                if kind == "tag":
                    tag_key = f"{lhs}::{value}"
                    tag_nt = self.tag_to_nt_mapping.get(tag_key)
                    if tag_nt:
                        production_sublist.extend(tag_nt.split())
                    else:
                        production_sublist.append(f"<<{value}>>")
                else:  # "other"
                    production_sublist.append(value)

            if production_sublist and production_sublist not in productions:
                productions.append(production_sublist)
                logging.info(f"  {lhs} -> {production_sublist}")
            elif not production_sublist:
                productions.append([])
                logging.info(f"  {lhs} -> []")

        return productions
    
    def process_full_grammar(self, grammar_dict):
        """
        Processa una grammatica completa con multiple regole di produzione.

        FIX (multi-tag ordering): il vecchio approccio raccoglieva TUTTI i tag
        di una regola in un unico pool e costruiva un singolo NT condiviso.
        Questo funzionava solo quando i tag erano alternative alla stessa
        posizione (es. S* → <<a>> | <<b>>), ma rompeva l'ordine quando la
        stessa produzione conteneva più tag in posizioni distinte
        (es. S* → <<a>> A <<b>>): entrambi i tag finivano nello stesso NT,
        quindi il PDA non imponeva nessun ordine sequenziale.

        La fix raggruppa i tag per POSIZIONE all'interno di ciascuna
        produzione alternativa.

          - «Posizione» = indice di apparizione del tag nella singola produzione
            (non l'indice nell'elenco di tutti gli elementi, ma il contatore
             che si incrementa solo sui tag).

          - Produzioni diverse che hanno un tag alla stessa posizione
            contribuiscono le loro alternative a quell'unico NT posizionale;
            il prefix-grouping interno a build_tag_grammar_for_rule rimane
            corretto e necessario per gestire queste alternative.

          - Posizioni diverse all'interno della stessa produzione producono
            NT distinti e indipendenti, garantendo l'ordine.

        Esempio — una sola produzione con due tag:
            S* → <<a>> A <<b>>

            posizione 0: tags = ["a"] → NT  S*_POS0  (→ <<a>>)
            posizione 1: tags = ["b"] → NT  S*_POS1  (→ <<b>>)
            produzione finale: [S*_POS0, A, S*_POS1]   ← ordine garantito

        Esempio — due produzioni alternative con tag condiviso in posizione 0:
            S* → <<a>> A <<x>>
            S* → <<b>> A <<x>>

            posizione 0: tags = ["a","b"] → NT  S*_POS0  (→ <<a>> | <<b>>)
            posizione 1: tags = ["x","x"] → NT  S*_POS1  (→ <<x>>)
        """
        logging.info("=== ELABORAZIONE GRAMMATICA COMPLETA (PER REGOLA) ===")
        logging.info(f"Regole originali: {grammar_dict}")

        self.non_terminals = set(grammar_dict.keys())
        # Azzerato ad ogni grammatica: i nomi dei NT ausiliari devono
        # dipendere solo dalla grammatica in ingresso, non da quante volte
        # il processore è già stato riusato.
        self.generated_nts = set()
        logging.info(f"Non terminali identificati: {self.non_terminals}")

        final_grammar = {}

        for lhs, rhs_list in grammar_dict.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"PROCESSAMENTO SEPARATO DELLA REGOLA: {lhs}")
            logging.info(f"{'='*60}")

            # Estrai gli elementi ordinati con tipo ('tag' o 'other')
            ordered_elements_list = self.extract_tags_and_others(rhs_list)

            # ── Passo 1: raggruppa i tag per posizione E per continuazione ──
            # position_groups[p][cont] = tag che compaiono nella p-esima
            # posizione-tag, raggruppati per ciò che li SEGUE nella produzione.
            #
            # BUG FIX (corruzione silenziosa della gerarchia).
            # Prima si raggruppava solo per posizione: tutti i tag di una
            # posizione finivano in UN solo NT condiviso.  Quel NT si biforca
            # al proprio interno (left-factoring dei prefissi di token), quindi
            # se le produzioni che lo usano hanno continuazioni DIVERSE la
            # corrispondenza tag→continuazione viene persa.  Esempio:
            #
            #   S* -> <<{"parent": "cs", "child": ">>  C_J
            #   S* -> <<{"parent": "ece", "child": ">> D_J
            #
            # I due tag condividono il prefisso di token '{" parent ": Ġ"',
            # quindi finivano nello stesso S*_POS0_TAG_NT1, e il passo 4
            # fattorizzava le continuazioni in un S*_FACT → C_J | D_J che non
            # sa più quale ramo è stato preso.  Effetti:
            #   - FIRST(C_J) ∩ FIRST(D_J) ≠ ∅  → ValueError: Conflict, cioè il
            #     rifiuto di una grammatica che È LL(1);
            #   - FIRST disgiunti → NESSUN errore e la gerarchia semplicemente
            #     non viene imposta: veniva accettato
            #     {"parent": "cs", "child": "electricity"}  con electricity
            #     figlio di ece.  Corruzione silenziosa.
            #
            # Due produzioni possono condividere la catena di NT a una
            # posizione solo se tutto ciò che segue quel tag è IDENTICO.
            # Altrimenti ogni continuazione ottiene la propria catena, e il
            # prefisso di token comune viene poi fattorizzato al passo 4 —
            # che così produce la forma corretta:
            #
            #   S*      -> '{"' 'parent' '":' 'Ġ"' S*_FACT
            #   S*_FACT -> 'cs'  … C_J        (FIRST = {'cs'})
            #   S*_FACT -> 'ece' … D_J        (FIRST = {'ece'})  → LL(1), corretta.
            #
            # Le grammatiche in cui i tag di una posizione hanno tutti la stessa
            # continuazione (il caso comune: enum di figli) producono un solo
            # gruppo e sono invariate rispetto a prima.
            position_groups: dict[int, dict[tuple, list[str]]] = {}
            for ordered_elements in ordered_elements_list:
                tag_pos = 0
                for idx, (kind, value) in enumerate(ordered_elements):
                    if kind == "tag":
                        continuation = tuple(ordered_elements[idx + 1:])
                        position_groups.setdefault(tag_pos, {})
                        position_groups[tag_pos].setdefault(continuation, [])
                        position_groups[tag_pos][continuation].append(value)
                        tag_pos += 1

            logging.info(f"Tag per posizione/continuazione per {lhs}: {position_groups}")

            # ── Passo 2: costruisci un NT per ogni (posizione, continuazione) ─
            # group_nt[(pos, cont)] = nome del NT da usare in quella posizione
            # per le produzioni con quella continuazione.  Con un solo gruppo
            # per posizione il nome resta {lhs}_POS{p}, identico a prima.
            group_nt: dict[tuple[int, tuple], str] = {}
            for pos, groups in sorted(position_groups.items()):
                multi = len(groups) > 1
                for gi, (continuation, tags_at_pos) in enumerate(groups.items()):
                    nt_for_pos = f"{lhs}_POS{pos}" if not multi else f"{lhs}_POS{pos}_G{gi}"
                    group_nt[(pos, continuation)] = nt_for_pos

                    # Deduplication preservando ordine (per deterministicità)
                    unique_tags = list(dict.fromkeys(tags_at_pos))
                    tag_grammar = self.build_tag_grammar_for_rule(unique_tags, nt_for_pos)

                    for key, value in tag_grammar.items():
                        final_grammar[key] = value

                    # Propaga il mapping al namespace lhs::tag::pos<n> in modo
                    # che il passo 3 lo trovi con chiave posizionale.
                    for tag in unique_tags:
                        src_key = f"{nt_for_pos}::{tag}"
                        dst_key = f"{lhs}::{tag}::pos{pos}"
                        if src_key in self.tag_to_nt_mapping:
                            self.tag_to_nt_mapping[dst_key] = self.tag_to_nt_mapping[src_key]

            # ── Passo 3: costruisci le produzioni finali ──────────────────
            productions = []
            for ordered_elements in ordered_elements_list:
                production_sublist = []
                tag_pos = 0
                for idx, (kind, value) in enumerate(ordered_elements):
                    if kind == "tag":
                        # Il NT dipende da (posizione, continuazione): due
                        # produzioni con continuazioni diverse NON condividono
                        # la catena, altrimenti si perde la corrispondenza
                        # tag→continuazione (vedi passo 1).
                        continuation = tuple(ordered_elements[idx + 1:])
                        nt_for_pos = group_nt[(tag_pos, continuation)]
                        expansion = self.tag_to_nt_mapping.get(f"{nt_for_pos}::{value}")
                        if expansion:
                            production_sublist.extend(expansion.split())
                        else:
                            # Tag a singolo token senza NT wrapper:
                            # usa direttamente i sotto-token del tag
                            production_sublist.extend(self.tokenize_tag(value))
                        tag_pos += 1
                    else:  # "other" → non-terminale o terminale letterale
                        production_sublist.append(value)

                if production_sublist and production_sublist not in productions:
                    productions.append(production_sublist)
                    logging.info(f"  {lhs} -> {production_sublist}")
                elif not production_sublist:
                    productions.append([])
                    logging.info(f"  {lhs} -> []")

            # ── Passo 4: fattorizzazione dei prefissi comuni ──────────────
            # Per gruppo (alternative con lo stesso primo simbolo) e
            # ricorsiva: vedi left_factor_productions per il bug fix.
            factored, helper_rules = self.left_factor_productions(lhs, productions)

            if helper_rules:
                logging.info(f"\n=== FATTORIZZAZIONE PER {lhs} ===")
                for helper_nt, helper_prods in helper_rules.items():
                    logging.info(f"  {helper_nt} -> {helper_prods}")

            final_grammar[(lhs, "RULE")] = factored
            for helper_nt, helper_prods in helper_rules.items():
                final_grammar[(helper_nt, "RULE")] = helper_prods

        self.save_final_grammar(final_grammar)
        return final_grammar, self.tag_to_nt_mapping

    def save_final_grammar(self, grammar, filename='final_grammar.txt'):
        """
        Serializza la grammatica finale in formato testuale per ispezione.

        Scrive in <package>/temp/final_grammar.txt. Utile per debug: permette
        di verificare che le produzioni generate corrispondano a quanto atteso.

        Nota: la docstring era misposizionata dopo il codice (BUG-15, ora
        corretta). Il path è ancorato alla directory del package: un path
        relativo alla cwd sporcava la working directory del chiamante.

        Parameters
        ----------
        grammar : dict
            Grammatica finale prodotta da process_full_grammar().
        filename : str
            Nome del file di output (default: 'final_grammar.txt').
        """
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_filename = os.path.join(package_dir, "temp", filename)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        if not grammar:
            logging.info("  Grammatica vuota")
            return

        with open(output_filename, 'w+') as f:
            #f.write("=== GRAMMATICA FINALE (PROCESSAMENTO PER REGOLA) ===\n\n")

            # Prima stampa le regole iniziali (non tuple)
            f.write("--- Regole principali ---\n")
            for key, productions in grammar.items():
                if not isinstance(key, tuple):
                    nt = key
                    unique_productions = []
                    for element in productions:
                        if isinstance(element, list) and len(element) > 2:
                            element = [element[0], element[1:]]
                        if element not in unique_productions:
                            unique_productions.append(element)

                    for prod in unique_productions:
                        if isinstance(prod, list):
                            if len(prod) > 1:
                                rhs = f"{prod[0]} {' '.join(prod[1])}" if isinstance(prod[1], list) else f"{prod[0]} {prod[1]}"
                            else:
                                rhs = prod[0]
                        else:
                            rhs = str(prod)
                        f.write(f"{nt} -> {rhs}\n")
                    logging.info(f"\nProduzioni per {nt}: {unique_productions}\n")

            #f.write("\n--- Regole per i tag (organizzate per regola originale) ---\n")
            
            # Raggruppa solo le chiavi tuple
            rules_by_origin = {}
            for key, productions in grammar.items():
                if isinstance(key, tuple):
                    nt, prefix = key
                    origin_rule = nt.split('_TAG_NT')[0] if '_TAG_NT' in nt else nt.split('_')[0]
                    rules_by_origin.setdefault(origin_rule, []).append(((nt, prefix), productions))

            # Stampa le regole dei tag
            for origin_rule, rule_group in rules_by_origin.items():
                f.write(f"\n-- Regole tag per {origin_rule} --\n")
                for (nt, prefix), productions in rule_group:
                    productions_str = []
                    for prod in productions:
                        if isinstance(prod, list):
                            if len(prod) == 0:
                                productions_str.append("ε")
                            elif len(prod) == 1 and isinstance(prod[0], str) and 'NT' in prod[0]:
                                productions_str.append(prod[0])
                            else:
                                productions_str.append(' '.join(prod))
                        else:
                            productions_str.append(str(prod))
                    if productions_str:
                        f.write(f"{nt} -> {' | '.join(productions_str)}\n")

        logging.info(f"Grammatica salvata in {output_filename}")