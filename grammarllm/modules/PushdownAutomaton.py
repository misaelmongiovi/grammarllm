import logging
class PushdownAutomaton:
    def __init__(self,grammar,startSymbol,map):
        self.stack = [startSymbol]
        self.grammar = grammar
        self.map_terminals_tokens = map
        self.map_tokens_terminals = {}
        
        for non_terminal, value in map.items():
            #print(f"Processing non-terminal: {non_terminal} {value}")  # DEBUG
            if isinstance(value, dict):
                # Caso: valore è un dizionario di {terminal: [tokens]}
                for terminal, tokens in value.items():
                    if isinstance(tokens, list):
                        for token in tokens:
                            if token not in self.map_tokens_terminals:
                                self.map_tokens_terminals[token] = []
                            self.map_tokens_terminals[token].append(terminal)
            elif isinstance(value, list):
                # Caso: valore è direttamente una lista di token
                for token in value:
                    if token not in self.map_tokens_terminals:
                        self.map_tokens_terminals[token] = []
                    self.map_tokens_terminals[token].append(non_terminal)


    def recursive_get_tokens(self, stack, visited=None):
        if visited is None:
            visited = set()

        if not stack:
            return []

        top = stack.pop()

        # If we've already visited this non-terminal, skip it to prevent infinite recursion
        if top in visited:
            return []

        visited.add(top)

        if top not in self.grammar:  # Terminal, return it directly
            return [top]

        tokens = []
        # Iterate over all the possible expansions for the current non-terminal
        for symbol in self.grammar[top]:
            # Before expanding the rule, ensure no infinite recursion occurs
            if symbol not in visited:
                stack.extend(reversed([symbol]))
                tokens += self.recursive_get_tokens(stack, visited)

        return tokens

    def get_tokens(self):
        terminals = self.recursive_get_tokens(self.stack.copy())
        tokens = set()
    
        for terminal in terminals:
            # print(terminal)
            # print(self.map_terminals_tokens[terminal])
            # print(tokens)
            assert set(self.map_terminals_tokens[terminal]).isdisjoint(tokens), "I token associati ai terminali non sono disgiunti"
            tokens.update(self.map_terminals_tokens[terminal])


        self.current_terminals = terminals
        return list(tokens)
    
    def next_state(self, token_gen):
        logging.info(f"current terminals is:{self.current_terminals}")
        check_terminals = set(self.map_tokens_terminals[token_gen]).intersection(set(self.current_terminals))
        logging.info(f"check_terminals is: {check_terminals}")


        assert len(check_terminals) == 1, "Scelto un token ambiguo, in quanto corrispondente a più possibili terminali per questo stato"
        terminal = list(check_terminals)[0]
        self.next_state_terminal(terminal)


    def next_state_terminal(self, terminal):
        token = terminal
        stack = self.stack
        top = stack.pop()
        
        # Se il top dello stack è una regola (non terminale), espanderla prima di confrontare
        if top in self.grammar:
            # Espande la regola e mette i simboli della produzione nello stack
            #print(f"Espando: top={top}, terminal={token}") #DEBUG
            #print(f"Chiavi disponibili per '{top}': {list(self.grammar[top].keys())}") #DEBUG
            
            for symbol in reversed(self.grammar[top][token]):
                stack.append(symbol)

            # Richiama ricorsivamente per gestire il terminale
            self.next_state_terminal(token)
            return
        
        if not top == token:
            # Ora il top dello stack deve essere un terminale
            print("Parser Stack:", stack)
            print("Comparing:", top, "vs", token)
            print(top == token, f"Errore: trovato '{top}', atteso '{token}'")
        assert top == token, f"Errore: trovato '{top}', atteso '{token}'"
        
    def eos(self):
        return True if not self.stack else False
