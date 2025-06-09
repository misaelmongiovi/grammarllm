from collections import defaultdict
import re
from functools import lru_cache

def resolve_rules_conflicts(rules):
    """
    Resolve conflicts in the rules by introducing new non-terminals when
    multiple productions for the same non-terminal start with the same prefix.
    
    Args:
        rules (dict): A dictionary mapping non-terminals to lists of productions
        
    Returns:
        dict: A modified rules with conflicts resolved
    """
    new_rules = rules.copy()
    new_non_terminals = {}
    
    # Identify non-terminals with conflicting productions
    for non_terminal, productions in rules.items():
        # Group productions by their first symbol
        prefix_groups = {}
        for production in productions:
            # Split production into tokens
            tokens = production.split() if isinstance(production, str) else production
            
            # Skip empty productions
            if not tokens:
                continue
                
            prefix = tokens[0]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(production)
        
        # Find prefixes with multiple productions
        for prefix, group in prefix_groups.items():
            if len(group) > 1:
                # Generate a unique name for the new non-terminal
                print(f"Conflict detected for non-terminal '{non_terminal}' with prefix '{prefix}'")
                # Create a new non-terminal name
                print(f"Creating new non-terminal for prefix '{prefix}'")
                new_nt_name = f"New_{non_terminal}_{prefix}"
                new_non_terminals[new_nt_name] = []
                
                # Create new productions for the new non-terminal
                for prod in group:
                    tokens = prod.split() if isinstance(prod, str) else prod
                    # Add the suffix (rest of production after prefix) as a new production
                    suffix = tokens[1:] if len(tokens) > 1 else []
                    if suffix:
                        new_non_terminals[new_nt_name].append(" ".join(suffix))
                    else:
                        new_non_terminals[new_nt_name].append("ε")  # Empty string
                
                # Remove the conflicting productions from the original non-terminal
                new_rules[non_terminal] = [p for p in new_rules[non_terminal] if p not in group]
                
                # Add the new production with the new non-terminal
                new_rules[non_terminal].append(f"{prefix} {new_nt_name}")
    
    # Add the new non-terminals to the rules
    for nt, prods in new_non_terminals.items():
        new_rules[nt] = prods
    
    return new_rules

def generate_non_terminals(grouped_data, count):
    """
    Generates non-terminals based on grouped data by state and prefix, preserving continuity across levels
    only if the prefix matches. This prevents incorrect assignment of different tokens to the same non-terminal.

    The function constructs:
    - G: a dictionary mapping each non-terminal to its corresponding prefix (string).
    - S: a dictionary mapping each non-terminal to the list of associated numeric positions.

    Logic:
    - For the first state (level 0), non-terminals are assigned sequentially, labeled like A0, A1, etc.
    - For subsequent states (1, 2, ...), the function checks for each token whether there is a token in the
      previous level such that:
        - The current position - 1 exists among the positions of the previous non-terminal.
        - The prefix is the same.
      If both conditions are met, the function reuses the index of the previous non-terminal with a new letter
      (e.g., B0 → C0, etc.), maintaining continuity.
    - If the prefix is different—even if the positions are consecutive—a new non-terminal is created.

    Args:
        grouped_data (dict): Data grouped by state.
                             Example:
                             {
                                 0: {'cell': [('cell', 1), ('cell', 6)],
                                     'enz': [('enz', 3)]},
                                 1: {'Ġbiology': [('Ġbiology', 2)],
                                     'ym': [('ym', 4)],
                                     'Ġpat': [('Ġpat', 7)]},
                                 2: {'ology': [('ology', 5), ('ology', 8)]}
                             }
        count (int): Counter used to generate derived symbols with apostrophes (').

    Returns:
        tuple: A pair of dictionaries (G, S):
            - G (dict): Maps non-terminals to their prefixes.
            - S (dict): Maps non-terminals to lists of numeric positions.

    Example output:
        G = {'A0': 'cell', 'A1': 'enz', 'B0': 'Ġbiology', 'B1': 'ym', 'B2': 'Ġpat', 'C0': 'ology'}
        S = {'A0': [1, 6], 'A1': [3], 'B0': [2], 'B1': [4], 'B2': [7], 'C0': [5, 8]}
    """
    if count is None:
        count = 0
    vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
             'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    global_count = 0

    G = {}
    S = {}

    # Primo stato
    state = 0
    prefix_dict = grouped_data[state]

    for i, (prefix, tuples) in enumerate(prefix_dict.items()):
        nt_key = f"{vocab[global_count]}{i}" + "'" * count
        G[nt_key] = prefix
        S[nt_key] = [pos for _, pos in tuples]

    global_count += 1

    # Stati successivi
    for state in range(1, len(grouped_data)):
        prefix_dict = grouped_data[state]
        temp_G = {}
        temp_S = {}
        used_indices = set()

        for prefix, tuples in prefix_dict.items():
            for _, pos in tuples:
                found = False
                for prev_nt, prev_positions in S.items():
                    if pos - 1 in prev_positions and G[prev_nt] == prefix:
                        index = int(''.join(filter(str.isdigit, prev_nt[1:])))
                        nt_key = f"{vocab[global_count]}{index}" + "'" * count
                        if nt_key in temp_S:
                            temp_S[nt_key].append(pos)
                        else:
                            temp_G[nt_key] = prefix
                            temp_S[nt_key] = [pos]
                        used_indices.add(index)
                        found = True
                        break

                if not found:
                    # Assegniamo un nuovo indice
                    new_index = 0
                    while new_index in used_indices:
                        new_index += 1
                    nt_key = f"{vocab[global_count]}{new_index}" + "'" * count
                    temp_G[nt_key] = prefix
                    temp_S[nt_key] = [pos]
                    used_indices.add(new_index)

        G.update(temp_G)
        S.update(temp_S)
        global_count += 1

    #print("G:", G) #DEBUG
    #print("S:", S) #DEBUG
    return G, S

def generate_grammar(G, S, NT=None, eos_symbol=None, non_terminals_list=None):
    rules = {}

    @lru_cache(maxsize=None)
    def find_number_for_initial_nt(nt):
        def recurse(current_nt, visited_nts=None, depth=0, max_depth=200):
            if visited_nts is None:
                visited_nts = set()
            #print(f"[DEBUG] Recurse: NT={current_nt}, depth={depth}, visited={visited_nts}") #DEBUG
            if depth > max_depth:
                raise RecursionError(f"[ERRORE] Profondità massima superata da {nt}. Percorso: {visited_nts}")
            if current_nt in visited_nts:
                raise ValueError(f"[ERRORE] Loop rilevato: {current_nt} già visitato. Percorso: {visited_nts}")
            visited_nts.add(current_nt)

            if current_nt.startswith("A"):
                number = extract_number(current_nt)
                #print(f"[DEBUG] Terminale iniziale trovato: {current_nt} -> {number}")
                return number

            positions = S.get(current_nt, [])
            for pos in positions:
                prev_pos = pos - 1
                #print(f"[DEBUG] Esaminando posizione {pos}, cercando NT con pos {prev_pos}")
                for candidate_nt, candidate_positions in S.items():
                    if prev_pos in candidate_positions:
                        #print(f"[DEBUG] ↪ Candidato NT: {candidate_nt} (posizioni: {candidate_positions})")
                        if candidate_nt.startswith("A"):
                            return extract_number(candidate_nt)
                        else:
                            try:
                                return recurse(candidate_nt, visited_nts.copy(), depth + 1, max_depth)
                            except (ValueError, RecursionError) as e:
                                print(f"[DEBUG] ✗ Backtracking da {candidate_nt}: {e}")
                                continue
            raise ValueError(f"[ERRORE] Nessun NT iniziale valido trovato da {nt}.")

        return recurse(nt)

    
    def extract_number(nt):
        match = re.match(r'[A-Z](\d+)', nt)
        if match:
            return int(match.group(1))
        raise ValueError(f"Formato non valido per simbolo: {nt}")

    def sort_key(nt):
        letter = nt[0]
        num = int(''.join(filter(str.isdigit, nt[1:])))
        return (num, letter)

    def add_rule(current_NT, rule):
        if rule not in rules.setdefault(current_NT, []):
            rules[current_NT].append(rule)

    def is_consecutive_transition(current_NT, next_NT):
        c_letter, c_num = current_NT[0], int(''.join(filter(str.isdigit, current_NT[1:])))
        n_letter, n_num = next_NT[0], int(''.join(filter(str.isdigit, next_NT[1:])))
        return c_letter == n_letter and n_num == c_num + 1

    @lru_cache(maxsize=None)
    def has_next_symbol(next_NT, current_pos=None):
        if current_pos is not None:
            for fut_NT, fut_pos_list in S.items():
                if current_pos + 1 in fut_pos_list:
                    if fut_NT[0] > next_NT[0]:
                        return True
        else:
            for p in S.get(next_NT, []):
                for fut_NT, fut_pos_list in S.items():
                    if p + 1 in fut_pos_list and fut_NT[0] > next_NT[0]:
                        return True
        return False

    def build_pos_to_index(S):
        pos_to_index = {}
        positions_list = [p for nt, positions in S.items() if nt.startswith('A') for p in positions]
        for idx, pos in enumerate(sorted(positions_list)):
            pos_to_index[pos] = idx
        return pos_to_index

    if NT == "S*":
        add_rule(NT, eos_symbol)

    pos_to_index = build_pos_to_index(S)
    sorted_nts = sorted(S.keys(), key=sort_key)
    non_terminals_list = {i: non_terminals_list[i] for i in range(len(non_terminals_list))}

    position_to_nts = defaultdict(set)
    for nt, positions in S.items():
        for p in positions:
            position_to_nts[p].add(nt)

    for nt, prefix in G.items():
        if re.match(r"^A\d+'*$", nt):
            for pos in S.get(nt, []):
                number = pos_to_index.get(pos)
                if NT not in rules:
                    rules[NT] = []

                if has_next_symbol(nt):
                    add_rule(NT, f"{prefix} {nt}")
                else:
                    nt_from_list = non_terminals_list.get(number)
                    add_rule(NT, f"{prefix}" if nt_from_list is None else f"{prefix} {nt_from_list}")

    if len(G) == 1 and len(S) == 1:
        key_G = next(iter(G))
        key_S = next(iter(S))
        if key_G == key_S:
            return rules

    for current_NT in sorted_nts:
        for pos in S[current_NT]:
            for next_NT in position_to_nts.get(pos + 1, []):
                if is_consecutive_transition(current_NT, next_NT):
                    continue
                if current_NT > next_NT:
                    continue

                next_prefix = G.get(next_NT, "")
                number = find_number_for_initial_nt(current_NT)
                if has_next_symbol(next_NT, pos + 1):
                    add_rule(current_NT, f"{next_prefix} {next_NT}")
                else:
                    nt_from_list = non_terminals_list.get(number)
                    add_rule(current_NT, f"{next_prefix}" if nt_from_list is None else f"{next_prefix} {nt_from_list}")
    
    #rules = resolve_rules_conflicts(rules)
    return rules
