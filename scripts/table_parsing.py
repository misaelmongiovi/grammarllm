import os
import logging
import json
from copy import deepcopy

def find_first(symbol, productions, first_sets):

    def calculate_first_of_sequence(symbols, productions, first_sets):
        """Calcola i first di una sequenza di simboli."""
        if not symbols:
            return {"ε"}
        
        first_symbol = symbols[0]
        
        # Se il primo simbolo è un non terminale
        if first_symbol in productions:
            if first_symbol not in first_sets:
                find_first(first_symbol, productions, first_sets)
            
            # Prendi i first del primo simbolo (tranne epsilon)
            result = first_sets[first_symbol].copy() - {"ε"}
            
            # Se il primo simbolo può produrre epsilon, considera il resto della sequenza
            if "ε" in first_sets[first_symbol]:
                # Calcola i first del resto della sequenza
                rest_first = calculate_first_of_sequence(symbols[1:], productions, first_sets)
                result |= rest_first
            
            return result
        else:
            # Se il primo simbolo è un terminale, restituisci un set contenente solo quel simbolo
            return {first_symbol}
    
    # Se il simbolo è già stato calcolato, restituisci il risultato
    if symbol in first_sets:
        return first_sets[symbol]
    
    # Se il simbolo è un non terminale (presente nelle produzioni)
    if symbol in productions:
        # Inizializza il set dei first per questo simbolo
        first_sets[symbol] = set()
        
        for production in productions.get(symbol, []):
            if production == "ε":
                first_sets[symbol].add("ε")
            else:
                # Dividi la produzione in simboli
                symbols = production.split()
                
                # Calcola i first della sequenza di simboli
                first_of_sequence = calculate_first_of_sequence(symbols, productions, first_sets)
                first_sets[symbol] |= first_of_sequence
        
        return first_sets[symbol]
    else:
        # Se il simbolo è un terminale, restituisci un set contenente solo quel simbolo
        return {symbol}

def compute_first_of_string(string,first_sets): #To use inside the parsing_table function
            first_result = set()
            for symbol in string.split():
                if symbol in first_sets:
                    first_result |= first_sets[symbol] - {'ε'}
                    if 'ε' not in first_sets[symbol]:
                        break
                else:
                    first_result.add(symbol)
                    break
            else:
                first_result.add('ε')
            return first_result  

def follow(productions, first_sets, start_symbol): #To use inside the parsing_table function
    """Computes Follow sets iteratively."""
    follow_sets = {nt: set() for nt in productions}
    follow_sets[start_symbol].add("$")  # Start symbol gets '$'
    
    changed = True
    while changed:
        changed = False
        for lhs, rhs_list in productions.items():
            for rhs in rhs_list:
                # Dividiamo la produzione in parole
                symbols = rhs.split()
                
                for i, symbol in enumerate(symbols):
                    if symbol in productions:  # It's a non-terminal
                        old_size = len(follow_sets[symbol])
                        
                        # Case 1: If there's something after `symbol`, add First(next) - {ε}
                        if i + 1 < len(symbols):
                            next_symbol = symbols[i + 1]
                            
                            if next_symbol in productions:
                                follow_sets[symbol] |= first_sets[next_symbol] - {"ε"}
                                if "ε" in first_sets[next_symbol]:
                                    # If next_symbol can be ε, add Follow(lhs)
                                    follow_sets[symbol] |= follow_sets[lhs]
                            else:
                                follow_sets[symbol].add(next_symbol)  # Terminal
                        
                        # Case 2: If symbol is at the end, inherit Follow(lhs)
                        if i + 1 == len(symbols) or ("ε" in first_sets.get(symbols[i + 1], set())):
                            follow_sets[symbol] |= follow_sets[lhs]
                        
                        # Check if anything changed to avoid infinite loop
                        if len(follow_sets[symbol]) > old_size:
                            changed = True
    
    return follow_sets

def remove_left_recursion_old(rules):
    def replace_sublist(lst, sublist, replacement):
        for i in range(len(lst) - len(sublist) + 1):
            if lst[i:i+len(sublist)] == sublist:
                return lst[:i] + [replacement] + lst[i+len(sublist):]
        return lst
    
    new_rules = deepcopy(rules)
    new_non_terminals = {}

    for non_terminal, productions in rules.items():
        # Group productions by their first token
        prefix_groups = {}
        for production in productions:
            tokens = production.split() if isinstance(production, str) else production
            if not tokens:
                continue
            prefix = tokens[0]
            prefix_groups.setdefault(prefix, []).append(production)

        for prefix, group in prefix_groups.items():
            if len(group) > 1:
                print(f"Conflict detected for non-terminal '{non_terminal}' with prefix '{prefix}'")
                new_nt_base = f"New_{non_terminal}_{prefix}"
                new_non_terminals[new_nt_base] = []
                
                # Fill in suffixes for the new non-terminal
                for prod in group:
                    tokens = prod.split()
                    suffix = tokens[1:] if len(tokens) > 1 else []
                    if suffix:
                        new_non_terminals[new_nt_base].append(" ".join(suffix))
                    else:
                        new_non_terminals[new_nt_base].append("ε")




                first_sets = {}
                for nt in rules:
                    find_first(nt, rules, first_sets)

                a = {}
                for symbol in new_non_terminals[new_nt_base]:
                    first_set = compute_first_of_string(symbol, first_sets)
                    for first in first_set:
                        a.setdefault(first, []).append(symbol)
                logging.info(a)

                for first_sym, symbols in a.items():
                    if len(symbols) > 1:
                        sub_nt = f"{new_nt_base}_{first_sym}"
                        new_non_terminals[sub_nt] = []

                        
                        # Raccogli le vere produzioni da tutti i symbol (es. B805, B806)
                        for symb in symbols:
                            real_productions = new_rules.get(symb, [])
                            for prod in real_productions:
                                tokens = prod.split()
                                if tokens and tokens[0] == first_sym:
                                    suffix = tokens[1:]  # rimuove il prefisso
                                    new_non_terminals[sub_nt].append(" ".join(suffix) if suffix else "ε")
                        

                        # Rimpiazza le symbol con il nuovo sub_nt nel base non-terminal
                        new_non_terminals[new_nt_base] = replace_sublist(
                            new_non_terminals[new_nt_base],
                            symbols,
                            f"{first_sym} {sub_nt}"
                        )

                            

                new_rules[non_terminal] = [p for p in new_rules[non_terminal] if p not in group]
                new_rules[non_terminal].append(f"{prefix} {new_nt_base}")

    # Merge all new non-terminals into final rule set
    for nt, prods in new_non_terminals.items():
        new_rules[nt] = prods

    return new_rules

def remove_left_recursion(rules):
    def replace_sublist(lst, sublist, replacement):
        for i in range(len(lst) - len(sublist) + 1):
            if lst[i:i+len(sublist)] == sublist:
                return lst[:i] + [replacement] + lst[i+len(sublist):]
        return lst

    def eliminate_first_conflicts(nt_base, new_rules, new_non_terminals, first_sets, visited=None):
        if visited is None:
            visited = set()
        if nt_base in visited:
            return
        visited.add(nt_base)

        a = {}
        seen_symbols = set()
        

    
        logging.debug(f"Productions for {nt_base}: {new_non_terminals.get(nt_base, [])}")

        for symbol in new_non_terminals.get(nt_base, []):
            if symbol in seen_symbols:
                continue  # evita di analizzare due volte lo stesso simbolo
            seen_symbols.add(symbol)

            first_set = compute_first_of_string(symbol, first_sets)
            for first in first_set:
                a.setdefault(first, []).append(symbol)

            
            ##
            #for key in a:
            #    a[key] = list(dict.fromkeys(a[key]))
            ##
            

        logging.info(f"First set groups for {nt_base}: {a}")

        for first_sym, symbols in a.items():

            if len(symbols) > 1:
                logging.info(f"Conflict detected for non-terminal '{nt_base}' with prefix '{first_sym}'")
                sub_nt = f"{nt_base}_{first_sym}" #e.g nt_base = 'New_A358_BS'
                if sub_nt in visited:
                    continue  # evita ciclo infinito
                new_non_terminals[sub_nt] = []

                for symb in symbols: #e.g symbols = ['B805', 'B806']            
                    # Nota: symb è una stringa di simboli (es. "S * B")
                    real_productions = new_rules.get(symb, []) #regola original dato il simbolo es: B805 restituisce ['OL C465']
                    if not real_productions:
                        new_non_terminals[sub_nt].append(symb)
                        continue

                    for prod in real_productions:
                        tokens = prod.split()
                        if tokens and tokens[0] == first_sym:
                            suffix = tokens[1:]  # rimuove il prefisso E.G 'OL'
                            production_str = " ".join(suffix) if suffix else "ε" #e.g suffix = 'C465'
                            if production_str not in new_non_terminals[sub_nt]:
                                new_non_terminals[sub_nt].append(production_str)

                # Sostituisci le produzioni duplicate con il nuovo sub_nt
                #e.g ['B803', 'B804', 'B805', 'B806', 'B807', 'B808', 'B809', 'B810']
                new_non_terminals[nt_base] = replace_sublist(
                    new_non_terminals[nt_base],
                    symbols,
                    f"{first_sym} {sub_nt}"
                )

                logging.info(f"new_non_terminals[nt_base]: {new_non_terminals[nt_base]}")
                # <-- AGGIUNGI QUESTO SUBITO DOPO
                new_rules[sub_nt] = new_non_terminals[sub_nt]
                new_rules[nt_base] = new_non_terminals[nt_base]

                logging.info(f"Updated productions for {nt_base}: {new_non_terminals[nt_base]}")
                # Richiama ricorsivamente su sub_nt
                eliminate_first_conflicts(sub_nt, new_rules, new_non_terminals, first_sets, visited)


    # --- Inizio corpo principale ---
    new_rules = deepcopy(rules)
    new_non_terminals = {}

    for non_terminal, productions in rules.items():
        prefix_groups = {}
        for production in productions:
            tokens = production.split() if isinstance(production, str) else production
            if not tokens:
                continue
            prefix = tokens[0]
            prefix_groups.setdefault(prefix, []).append(production)
            

        for prefix, group in prefix_groups.items():
            if len(group) > 1:
                print(f"Conflict detected for non-terminal '{non_terminal}' with prefix '{prefix}'")
                new_nt_base = f"New_{non_terminal}_{prefix}"
                new_non_terminals[new_nt_base] = []

                for prod in group:
                    tokens = prod.split()
                    suffix = tokens[1:] if len(tokens) > 1 else []
                    if suffix:
                        new_non_terminals[new_nt_base].append(" ".join(suffix))
                    else:
                        new_non_terminals[new_nt_base].append("ε")

                new_rules[non_terminal] = [p for p in new_rules[non_terminal] if p not in group]
                new_rules[non_terminal].append(f"{prefix} {new_nt_base}")

                first_sets = {}
                for nt in rules:
                    find_first(nt, rules, first_sets)

                eliminate_first_conflicts(new_nt_base, new_rules, new_non_terminals, first_sets)

    # Aggiunge i nuovi non terminali generati
    #for nt, prods in new_non_terminals.items():
    #    new_rules[nt] = prods
    
    logging.info(f"Final rules after left recursion elimination: {new_rules}")
    return new_rules


def parsing_table(final_rules): #messo all'inzio ma in realtà è + in fondo perchè viene chiamto solo quando la final_rules è pronta 
    """Prende in input le produzioni e un tokenizer e restiruisce la tabella di Parsing

    Args:
        productions (dict): Production rules
        tokenizer (obj): model's tokenizer

    Returns:
        dict: LL1 Parsing Table che verrà usata dal PDA
    """
    
    def save_table_parsing_as_txt(parsing_table):
        output_grammar_file = os.path.join('temp', 'table_parsing.json')
        os.makedirs(os.path.dirname(output_grammar_file), exist_ok=True)
        with open(output_grammar_file, "w", encoding="utf-8") as f:
            f.write("{\n")

            items = list(parsing_table.items())
            for i, (nt, rules) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""  
                f.write(f"    {json.dumps(nt)}: {json.dumps(rules)}{comma}\n")

            f.write("}\n")

        logging.info(f"\nTable Parsing as txt saved to {output_grammar_file}")

    def compute_parsing_table(productions, first_sets, follow_sets):
        
        parsing_table = {non_terminal: {} for non_terminal in productions}

        for non_terminal, rules in productions.items():
            for rule in rules:
                first_alpha = compute_first_of_string(rule, first_sets)

                for terminal in first_alpha - {'ε'}:
                    if terminal in parsing_table[non_terminal]:
                        #print(f"Conflict: {non_terminal} → {terminal} {parsing_table[non_terminal]}!\nRegola attuale: {rule}!\nConflitto di ambiguità perchè non possono esserci due regole che abbiano a dx un terminale diverso")
                        raise ValueError(f"Conflict: {non_terminal} → {terminal} {parsing_table[non_terminal][terminal]}!\nRegola attuale: {terminal} {rule}!\nConflitto di ambiguità perchè non possono esserci due regole che abbiano a dx un terminale diverso")
                    parsing_table[non_terminal][terminal] = rule.split()

                if 'ε' in first_alpha:
                    for terminal in follow_sets[non_terminal]:
                        if terminal in parsing_table[non_terminal]:
                            raise ValueError(f"Conflict: {non_terminal} → {terminal} {parsing_table[non_terminal][terminal]}!\nRegola attuale: {terminal} {rule}!\nConflitto di ambiguità perchè non possono esserci due regole che abbiano a dx un terminale diverso")
                        parsing_table[non_terminal][terminal] = []

        for key in parsing_table:
            parsing_table[key].pop('$', None)
        return parsing_table
    ##
    #Conviene modificare qua final_rules in maniera ricorsiva per risolvere tutti i conflitti?
    ##
    
    final_rules = remove_left_recursion(final_rules)

    logging.info("\nProcessed grammar:\n")
    logging.info(final_rules)

    first_sets = {}
    for nt in final_rules:
        find_first(nt, final_rules, first_sets)

    logging.info("\nFirst sets:\n")
    logging.info(first_sets)

    follow_sets = follow(final_rules, first_sets, 'S*')

    logging.info("\nFollow sets:\n")
    logging.info(follow_sets)

    parsing_table = compute_parsing_table(final_rules, first_sets, follow_sets)
    
    save_table_parsing_as_txt(parsing_table)

    return parsing_table