import re
import itertools
import logging


def generate_token_maps(tokenizer, table_parsing, regex_dict=None):
    
    def check_tokens_conflicts(table_parsing, map_terminal_tokens):
        conflicts = []
        for lhs, rhs_list in table_parsing.items():
        
            for a, b in itertools.combinations(rhs_list.keys(), 2):
                
                intersection = set(map_terminal_tokens[a]) & set(map_terminal_tokens[b])
                if intersection:
                    conflicts.append(f"I set di tokens associati ai terminali '{a}' e '{b}' non sono disgiunti. Intersezione: {list(intersection)[:5]}")
            

        # Se ci sono conflitti, li stampiamo o solleviamo un'eccezione con tutti i dettagli
        if conflicts:
            return conflicts
        
    map_terminal_tokens = {}
    vocab = tokenizer.get_vocab()  # token_string -> token_id

    if regex_dict:
        # Usa direttamente i token_id
        map_terminal_tokens = {
            name[6:]: [token_id for token_str, token_id in vocab.items() if regex.match(token_str)]#[:10]
            for name, regex in regex_dict.items()
            if name.startswith('regex_')
        }

        for lhs, rhs_list in table_parsing.items():
            for terminal in rhs_list.keys():
                if terminal not in map_terminal_tokens:
                    #print(f"terminal {terminal} aggiunto al map token terminals!") #debug
                    regex = re.compile(rf"^{re.escape(terminal)}$")
                    map_terminal_tokens[terminal] = [token_id for token_str, token_id in vocab.items() if regex.match(token_str)]

    else:  # Per i casi Classification e VR, senza regex
        for lhs, rhs_list in table_parsing.items():
            for terminal in rhs_list.keys():
                if terminal not in map_terminal_tokens:
                    #print(f"terminal {terminal} aggiunto al map token terminals!") #debug
                    regex = re.compile(rf"^{re.escape(terminal)}$")
                    map_terminal_tokens[terminal] = [token_id for token_str, token_id in vocab.items() if regex.match(token_str)]

    conflicts = check_tokens_conflicts(table_parsing, map_terminal_tokens)
    if conflicts:
        raise ValueError("\n".join(conflicts))

    logging.info('\nmap_terminal_tokens')
    logging.info(map_terminal_tokens)
    return map_terminal_tokens



