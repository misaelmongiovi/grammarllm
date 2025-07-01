import re
import logging

class ProductionRuleProcessor:
    def __init__(self, tokenizer=None):
        self.nt_counter = 0
        self.sub_nt_counter = {}
        self.tag_to_nt_mapping = {}  # Mappa dai tag originali ai NT creati
        self.original_rules_mapping = {}  # Mappa per mantenere il collegamento con le regole originali
        self.tokenizer = tokenizer  # Tokenizer di Hugging Face
        self.non_terminals = set()  # Traccia tutti i non terminali
        self.rule_specific_grammars = {}  # Grammatiche specifiche per ogni regola
    
    def extract_tags_and_others(self, rhs_list):
        """Estrae i tag <<...>> e gli altri elementi dalle regole di produzione"""
        tags_list = []
        others_list = []
        tag_pattern = re.compile(r'<<(.+?)>>')

        def smart_split(item):
            # Trova tutti i tag <<...>> e separa il resto del testo
            matches = list(tag_pattern.finditer(item))
            parts = []
            last_index = 0

            for match in matches:
                # Aggiungi il testo prima del tag, splittato
                pre_text = item[last_index:match.start()]
                if pre_text.strip():
                    parts.extend(pre_text.strip().split())

                # Aggiungi il tag intero come una sola unità
                parts.append(match.group(0))
                last_index = match.end()

            # Aggiungi eventuale testo dopo l'ultimo tag
            post_text = item[last_index:]
            if post_text.strip():
                parts.extend(post_text.strip().split())

            return parts

        for item in rhs_list:
            tags = []
            others = []
            if re.search(tag_pattern, item):
                words = smart_split(item)
                current_chunk = []
                for word in words:
                    match = re.fullmatch(tag_pattern, word)
                    if match:
                        tags.append(match.group(1))  # salva solo il contenuto del tag
                        # Se c'è un chunk corrente, salvalo
                        if current_chunk:
                            others.append(' '.join(current_chunk))
                            current_chunk = []
                    else:
                        current_chunk.append(word)

                # Aggiungi l'ultimo chunk se presente
                if current_chunk:
                    others.append(' '.join(current_chunk))

                # Se non ci sono altri elementi, aggiungi None
                if not others:
                    others.append(None)

                tags_list.append(tags)
                others_list.append(others)
            else:
                tags_list.append([])
                others_list.append([item])

        return tags_list, others_list
    
    def tokenize_tag(self, tag):
        """Tokenizza un tag usando il tokenizer di Hugging Face"""
        if self.tokenizer is None:
            # Fallback alla tokenizzazione semplice se non c'è tokenizer
            logging.info("ATTENZIONE: Nessun tokenizer fornito, uso tokenizzazione semplice")
            return [tag]  # Restituisce il tag come singolo token
        
        # Usa il tokenizer di Hugging Face
        tokens = self.tokenizer.tokenize(tag)
        return tokens


    def get_prefix_groups_for_rule(self, tags, rule_name):
        """Raggruppa i tag in base ai prefissi comuni effettivamente condivisi per una regola specifica"""
        tokenized = {tag: self.tokenize_tag(tag) for tag in tags if tag}
        
        # Trova i prefissi comuni tra più parole
        prefix_counts = {}
        for tag, tokens in tokenized.items():
            if tokens:
                prefix = tokens[0]
                prefix_counts.setdefault(prefix, []).append((tag, tokens[1:]))

        # Prefissi condivisi da almeno 2 tag
        prefix_groups = {prefix: items for prefix, items in prefix_counts.items() if len(items) > 1}

        # Tag non inclusi nei gruppi
        grouped_tags = {tag for group in prefix_groups.values() for tag, _ in group}
        ungrouped_tags = [tag for tag in tags if tag not in grouped_tags]

        # TO UNCOMMENT ONLY IF YOU WANT TO DEBUG
        # # Scrivi su file per debug
        # with open(f'temp/prefix_group_{rule_name}.txt', 'w+') as f:
        #     f.write(f"=== PREFISSI PER REGOLA {rule_name} ===\n")
        #     for prefix, tag_suffix_pairs in prefix_groups.items():
        #         f.write(f"Prefisso condiviso: {prefix}, Tag e Suffix: {tag_suffix_pairs}\n")
        #     f.write(f"\nTag non raggruppati:\n{ungrouped_tags}\n")

        return prefix_groups, ungrouped_tags
    
    def create_initial_grammar_for_rule(self, prefix_groups, ungrouped_tags, rule_name):
        """
        Crea la grammatica iniziale con NT per ogni gruppo di prefisso condiviso
        e aggiunge anche i tag non raggruppati (che non condividono alcun prefisso).
        """
        grammar = {}
        start_productions = []

        # Gestione dei gruppi con prefissi condivisi
        for i, (prefix, tag_suffix_pairs) in enumerate(prefix_groups.items(), 1):
            if all(len(suffix) == 0 for _, suffix in tag_suffix_pairs):
                # Tutti i tag sono singoli token: usa solo il prefisso
                for tag, _ in tag_suffix_pairs:
                    start_productions.append(prefix)
                    self.tag_to_nt_mapping[f"{rule_name}::{tag}"] = prefix
            else:
                # Crea un NT con prefisso
                nt = f"{rule_name}_TAG_NT{i}"
                start_productions.append(f"{prefix} {nt}")

                suffixes = [suffix for _, suffix in tag_suffix_pairs]
                grammar[(nt, prefix)] = suffixes

                for tag, _ in tag_suffix_pairs:
                    self.tag_to_nt_mapping[f"{rule_name}::{tag}"] = f"{prefix} {nt}"

        # Gestione dei tag non raggruppati (tokenizzati completamente)
        for tag in ungrouped_tags:
            tokens = self.tokenize_tag(tag)
            production = " ".join(tokens)
            start_productions.append(production)
            self.tag_to_nt_mapping[f"{rule_name}::{tag}"] = production

        #grammar[rule_name] = start_productions
        return grammar


    def find_common_prefixes(self, token_lists):
        """Trova prefissi comuni in una lista di liste di token"""
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
        """Genera il prossimo nome di non terminale per un NT padre"""
        if parent_nt not in self.sub_nt_counter:
            self.sub_nt_counter[parent_nt] = 0
        self.sub_nt_counter[parent_nt] += 1
        return f"{parent_nt}_{self.sub_nt_counter[parent_nt]}"
    
    def process_grammar_iteration(self, grammar):
        """Esegue una iterazione di raffinamento sulla grammatica"""
        new_grammar = grammar.copy()
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
                        new_grammar[(new_nt, common_prefix)] = suffixes
                        changed = True

                    for tokens in token_lists:
                        if len(tokens) > 0 and tokens[0] not in common_prefixes:
                            new_token_lists.append(tokens)
                        elif len(tokens) == 0:
                            new_token_lists.append(tokens)

                    new_grammar[(nt, prefix)] = new_token_lists

        return new_grammar, changed

    
    def build_tag_grammar_for_rule(self, tags, rule_name):
        """Costruisce la grammatica per i tag di una specifica regola"""
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
    
    def find_common_prefixes_in_productions(self, productions):
        """Trova prefissi comuni nelle produzioni - VERSIONE AGGIORNATA PER LISTE"""
        if len(productions) <= 1:
            return productions, {}
        
        # Le produzioni sono già liste di token, non stringhe
        tokenized_productions = []
        for prod in productions:
            if isinstance(prod, list):
                if len(prod) == 0:
                    tokenized_productions.append([])  # Produzione vuota
                else:
                    tokenized_productions.append(prod)  # Già una lista di token
            else:
                # Fallback per stringhe (se ancora presenti)
                if prod == "ε":
                    tokenized_productions.append([])
                else:
                    tokenized_productions.append(prod.split())
        
        # Trova il prefisso comune più lungo
        common_prefix = []
        if tokenized_productions:
            min_len = min(len(prod) for prod in tokenized_productions if len(prod) > 0)
            if min_len > 0:
                for i in range(min_len):
                    tokens_at_pos = [prod[i] for prod in tokenized_productions if len(prod) > i]
                    if len(set(tokens_at_pos)) == 1:  # Tutti uguali
                        common_prefix.append(tokens_at_pos[0])
                    else:
                        break
        
        if len(common_prefix) == 0:
            return productions, {}
        
        # Crea le nuove produzioni rimuovendo il prefisso comune
        new_productions = []
        suffixes = []
        factorization_info = {}
        
        for i, prod in enumerate(tokenized_productions):
            if len(prod) >= len(common_prefix):
                suffix = prod[len(common_prefix):]
                if len(suffix) == 0:
                    suffixes.append([])  # Produzione vuota come lista vuota
                else:
                    suffixes.append(suffix)  # Mantieni come lista
            else:
                # Produzione più corta del prefisso comune
                new_productions.append(productions[i])
        
        if suffixes:
            factorization_info = {
                'common_prefix': common_prefix,  # Lista di token invece che stringa
                'suffixes': suffixes             # Lista di liste invece che lista di stringhe
            }
        
        return new_productions, factorization_info

    def create_final_productions_for_rule(self, lhs, rhs_list, rule_specific_tag_grammar):
        """Crea le regole di produzione finali per una singola regola grammaticale usando la grammatica specifica"""
        logging.info(f"\n=== PROCESSAMENTO REGOLA: {lhs} ===")
        
        # Estrai tag e altri elementi
        tags_list, others_list = self.extract_tags_and_others(rhs_list)
        
        logging.info(f"RHS originali: {rhs_list}")
        logging.info(f"Tag estratti: {tags_list}")
        logging.info(f"Altri elementi: {others_list}")
        
        # Crea le produzioni con struttura simile ai tag
        productions = []
        
        for i, (tags, others) in enumerate(zip(tags_list, others_list)):
            logging.info(f"Tags: {tags}, Altri: {others}")
            
            # Crea una sottolista per questa produzione
            production_sublist = []
            
            # Processo alternato: tag, altri, tag, altri, ...
            max_len = max(len(tags), len(others))
            
            for j in range(max_len):
                # Aggiungi tag se presente
                if j < len(tags) and tags[j]:
                    # Cerca il mapping specifico per questa regola
                    tag_key = f"{lhs}::{tags[j]}"
                    tag_nt = self.tag_to_nt_mapping.get(tag_key)
                    if tag_nt:
                        tag_tokens = tag_nt.split()  # "13 TAG_NT1" -> ["13", "TAG_NT1"]
                        production_sublist.extend(tag_tokens)
                    else:
                        production_sublist.append(f"<<{tags[j]}>>")
                
                # Aggiungi altri elementi se presenti
                if j < len(others) and others[j] and others[j].strip():
                    other_tokens = others[j].split()
                    production_sublist.extend(other_tokens)
            
            if production_sublist:
                if production_sublist not in productions:
                    productions.append(production_sublist)
                    logging.info(f"  {lhs} -> {production_sublist}")
            else:
                productions.append([])
                logging.info(f"  {lhs} -> []")
        
        return productions
    
    def process_full_grammar(self, grammar_dict):
        """Processa una grammatica completa con multiple regole di produzione - VERSIONE MODIFICATA"""
        logging.info("=== ELABORAZIONE GRAMMATICA COMPLETA (PER REGOLA) ===")
        logging.info(f"Regole originali: {grammar_dict}")
        
        # Identifica tutti i non terminali
        self.non_terminals = set(grammar_dict.keys())
        logging.info(f"Non terminali identificati: {self.non_terminals}")
        
        # Processa ogni regola separatamente
        final_grammar = {}
        
        for lhs, rhs_list in grammar_dict.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"PROCESSAMENTO SEPARATO DELLA REGOLA: {lhs}")
            logging.info(f"{'='*60}")
            
            # Estrai i tag specifici di questa regola
            tags_list, others_list = self.extract_tags_and_others(rhs_list)
            rule_tags = []
            
            for tags in tags_list:
                for tag in tags:
                    if tag and tag not in rule_tags:
                        rule_tags.append(tag)
            
            logging.info(f"Tag specifici per {lhs}: {rule_tags}")
            
            # Costruisci la grammatica per i tag di questa regola specifica
            if rule_tags:
                rule_tag_grammar = self.build_tag_grammar_for_rule(rule_tags, lhs)
                
                # Aggiungi la grammatica dei tag di questa regola alla grammatica finale
                for key, value in rule_tag_grammar.items():
                    final_grammar[key] = value
            else:
                rule_tag_grammar = {}
            
            # Crea le produzioni finali per questa regola
            productions = self.create_final_productions_for_rule(lhs, rhs_list, rule_tag_grammar)
            
            # Applica fattorizzazione dei prefissi comuni
            factorized_productions, factorization_info = self.find_common_prefixes_in_productions(productions)
            
            if factorization_info:
                # Crea nuovo non terminale per i suffissi
                new_nt = f"{lhs}_FACT"
                logging.info(f"\n=== FATTORIZZAZIONE PER {lhs} ===")
                logging.info(f"Prefisso comune: {factorization_info['common_prefix']}")
                logging.info(f"Suffissi: {factorization_info['suffixes']}")
                
                # Regola principale fattorizzata
                main_production = factorization_info['common_prefix'] + [new_nt]
                final_grammar[(lhs, "RULE")] = [main_production]
                
                # Regola per i suffissi
                final_grammar[(new_nt, "RULE")] = factorization_info['suffixes']
                
            else:
                final_grammar[(lhs, "RULE")] = productions
        
        self.save_final_grammar(final_grammar)
        return final_grammar, self.tag_to_nt_mapping

    def save_final_grammar(self, grammar, filename='temp/final_grammar.txt'):
        """Salva la grammatica finale in formato leggibile"""
        if not grammar:
            logging.info("  Grammatica vuota")
            return

        with open(filename, 'w+') as f:
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

        logging.info(f"Grammatica salvata in {filename}")

    