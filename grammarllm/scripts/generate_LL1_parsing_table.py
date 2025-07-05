import os, json, logging
from collections import defaultdict
from copy import deepcopy

def compute_first_of_string(symbols, first_sets):
    """Calcola FIRST per una sequenza di simboli (es. corpo di una produzione)."""
    first_result = set()
    #print('symbols:', symbols) #DEBUG
    for symbol in symbols:
        #print(f"    Calculating of a String ({symbol},{symbols})...") #DEBUG
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

def find_first(symbol, productions, first_sets):
    def calculate_first_of_sequence(symbols, productions, first_sets):
        """Calcola i FIRST di una sequenza di simboli."""
        if not symbols:
            return {"ε"}
        #print(f"Calculating FIRST sequence: {symbols}...") #DEBUG

        first_symbol = symbols[0]
        

        if first_symbol in productions:
            if first_symbol not in first_sets:
                find_first(first_symbol, productions, first_sets)

            result = first_sets[first_symbol].copy() - {"ε"}

            if "ε" in first_sets[first_symbol]:
                rest_first = calculate_first_of_sequence(symbols[1:], productions, first_sets)
                result |= rest_first

            return result
        else:
            return {first_symbol}

    if symbol in first_sets:
        return first_sets[symbol]

    if symbol in productions:
        first_sets[symbol] = set()

        for production in productions[symbol]:
            if production == ['ε']:
                first_sets[symbol].add("ε")
            else:
                first_of_sequence = calculate_first_of_sequence(production, productions, first_sets)
                first_sets[symbol] |= first_of_sequence

        return first_sets[symbol]
    else:
        return {symbol}

def follow(productions, first_sets, start_symbol):
    follow_sets = {nt: set() for nt in productions}
    follow_sets[start_symbol].add("$")

    changed = True
    while changed:
        changed = False
        for lhs, rhs_list in productions.items():
            for rhs in rhs_list:
                for i, symbol in enumerate(rhs):
                    if symbol in productions:  # Non terminale
                        old_size = len(follow_sets[symbol])

                        # Case 1: simbolo seguito da altri simboli
                        if i + 1 < len(rhs):
                            next_symbols = rhs[i+1:]
                            first_of_next = compute_first_of_string(next_symbols, first_sets)
                            follow_sets[symbol] |= first_of_next - {"ε"}
                            if "ε" in first_of_next:
                                follow_sets[symbol] |= follow_sets[lhs]
                        else:
                            # Case 2: simbolo alla fine → eredita follow(lhs)
                            follow_sets[symbol] |= follow_sets[lhs]

                        if len(follow_sets[symbol]) > old_size:
                            changed = True

    return follow_sets

def parsing_table(final_rules):

    # Prepara la grammatica in forma classica
    grammar = defaultdict(list)
    for (nt, _), rules in final_rules.items():
        grammar[nt].extend(rules)

    def save_table_parsing_as_txt(parsing_table):
        output_grammar_file = os.path.join('grammarllm/temp', 'table_parsing.json')
        os.makedirs(os.path.dirname(output_grammar_file), exist_ok=True)
        with open(output_grammar_file, "w", encoding="utf-8") as f:
            f.write("{\n")
            items = list(parsing_table.items())
            for i, (nt, rules) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""  
                f.write(f"    {json.dumps(nt)}: {json.dumps(rules)}{comma}\n")
            f.write("}\n")
        logging.info(f"\nTable Parsing saved to {output_grammar_file}")

    def compute_parsing_table(productions, first_sets, follow_sets):
        """
        Crea la tabella di parsing LL(1) a partire dalle produzioni e dai FIRST e FOLLOW.
        ⚠️ Conflitto LL(1):
        Quando due produzioni di un non terminale condividono lo stesso terminale nel loro FIRST, la tabella LL(1) non può decidere in modo deterministico quale regola usare → errore.
        """
        parsing_table = {non_terminal: {} for non_terminal in productions}
        for non_terminal, rules in productions.items():
            #print(f"Processing non-terminal: {non_terminal}, {rules}\n") #DEBUG
            for rule in rules:
                first_alpha = compute_first_of_string(rule, first_sets)
                for terminal in first_alpha - {'ε'}:
                    if terminal in parsing_table[non_terminal]:
                        raise ValueError(f"Conflict: {non_terminal} → {terminal} {parsing_table[non_terminal][terminal]}!\nRegola attuale: {terminal} {rule}!")
                    parsing_table[non_terminal][terminal] = rule
                if 'ε' in first_alpha:
                    for terminal in follow_sets[non_terminal]:
                        if terminal in parsing_table[non_terminal]:
                            raise ValueError(f"Conflict: {non_terminal} → {terminal} {parsing_table[non_terminal][terminal]}!\nRegola attuale: {terminal} {rule}!")
                        parsing_table[non_terminal][terminal] = []
        for key in parsing_table:
            parsing_table[key].pop('$', None)
        return parsing_table
    
    logging.info("\nProcessed grammar:\n")
    logging.info(final_rules)

    # Calcola FIRST e FOLLOW
    first_sets = {}
    for nt in grammar:
        find_first(nt, grammar, first_sets)

    logging.info("\nFirst sets:\n")
    logging.info(first_sets)
    

    follow_sets = follow(grammar, first_sets, 'S*')
    logging.info("\nFollow sets:\n")
    logging.info(follow_sets)
   

    table = compute_parsing_table(grammar, first_sets, follow_sets)
    save_table_parsing_as_txt(table)

    return table
