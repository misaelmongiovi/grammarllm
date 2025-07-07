from grammarllm.scripts.grammar_generation import ProductionRuleProcessor
from grammarllm.scripts.map_terminal_tokens import generate_token_maps
from grammarllm.scripts.generate_LL1_parsing_table import parsing_table

from grammarllm.modules.BaseStreamer import BaseStreamer
from grammarllm.modules.PushdownAutomaton import PushdownAutomaton
from grammarllm.modules.SimpleLogitProcessor import MaskLogitsProcessor

import logging
import os
import re

#from grammarllm.utils.common_regex import regex_dict
#from grammarllm.utils.examples import *
#from grammarllm.utils.gloss_class import classes
from grammarllm.utils.toolbox import create_prompt, CHAT_TEMPLATE 

from transformers import AutoTokenizer, AutoModelForCausalLM



def get_parsing_table_and_map_tt(tokenizer, productions=None, regex_dict=None):

    processor = ProductionRuleProcessor(tokenizer=tokenizer)
    # Process the grammar productions
    if productions is None:
        raise ValueError("Productions cannot be None. Please provide a valid grammar.")
    final_grammar, tag_mapping = processor.process_full_grammar(productions)

    #add eos token to the grammar
    final_grammar[('S*','RULE')].append([tokenizer.eos_token])
    # Generate parsing table
    pars_tab = parsing_table(final_grammar)

    # Generate token maps
    if regex_dict:
        map_terminal_tokens = generate_token_maps(tokenizer, pars_tab, regex_dict)
    else:
        map_terminal_tokens = generate_token_maps(tokenizer, pars_tab)

    # uncomment the following lines to log the parsing table and terminal token mappings
    # logging.info("\nMap Terminal Tokens:\n")
    # for key, values in map_terminal_tokens.items():
    #     logging.info(f"{key} -> {values}")

    return pars_tab, map_terminal_tokens

def generate_grammar_parameters(tokenizer, pars_tab, map_terminal_tokens):
    # Create Pushdown Automaton and initialize processors and streamer
    pda = PushdownAutomaton(grammar=pars_tab, startSymbol='S*', map=map_terminal_tokens)
    return MaskLogitsProcessor(tokenizer, pda), BaseStreamer(tokenizer, pda)

def setup_logging():
    """Setup logging configuration."""
    log_dir = 'grammarllm/temp'
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    
    logging.basicConfig(
        filename=os.path.join(log_dir, 'GRAM-GEN.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w+'  # Overwrites the file every time
    )

def generate_text(model, tokenizer, text, logit_processor, streamer, max_new_tokens=400, do_sample=False, temperature=None, top_p=None, **kwargs):
    """
    Genera testo vincolato dalla grammatica, con configurazione dei parametri di generazione sicura.

    Args:
        model: Il modello pre-addestrato.
        tokenizer: Il tokenizer del modello.
        text: Input text iniziale.
        logit_processor: Processor dei logit basato sulla grammatica.
        streamer: Streamer per l'output live.
        max_new_tokens: Numero massimo di nuovi token da generare.
        do_sample: Se True, abilita la generazione stocastica.
        temperature: Controlla la casualità (usato solo se do_sample=True).
        top_p: Top-p (nucleus sampling), usato solo se do_sample=True.
        **kwargs: Parametri aggiuntivi opzionali per model.generate().
    """

    try:
        # TO USE WHEN CREATE PROMPT IS USED AND PROMPT IS A LIST
        if isinstance(text,list):
            tokenizer.chat_template = CHAT_TEMPLATE
            tokenized_input = tokenizer.apply_chat_template(text, 
                                                        tokenize=True,
                                                        add_generation_prompt=True,
                                                        return_dict=True,
                                                        return_tensors="pt").to(model.device)
            #logging.info(tokenized_input) #DEBUG
        else:
            tokenized_input = tokenizer(text, return_tensors="pt")

        # Safe defaults
        kwargs.setdefault("num_beams", 1)  # beam search disattivato
        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        # Sicurezza num_beams
        if kwargs["num_beams"] != 1:
            logging.warning("⚠️ num_beams > 1 non è compatibile con la generazione vincolata da grammatica. Impostato automaticamente a num_beams=1.")
            kwargs["num_beams"] = 1

        # Sampling parameters
        if do_sample:
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
        else:
            # Rimuovi parametri di sampling se presenti
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)

        # Device compatibility
        device = model.device
        input_ids = tokenized_input["input_ids"].to(device)
        if input_ids.device != model.device:
            logging.warning("Errore: gli 'input_ids' sono sulla device {input_ids.device}, mentre il modello è sulla device {model.device}. Spostando 'input_ids' sulla stessa device del modello.")
        
        attention_mask = tokenized_input["attention_mask"].to(device)
        if attention_mask.device != model.device:
            logging.warning(f"Errore: l'attention_mask è sulla device {attention_mask.device}, mentre il modello è sulla device {model.device}. Spostando 'attention_mask' sulla stessa device del modello.")
        

        start = input_ids.shape[1]

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            logits_processor=[logit_processor],
            **kwargs
        )

        answer = tokenizer.decode(output[0][start:], skip_special_tokens=True)

        return answer

    except Exception as e:
        raise RuntimeError(f"Errore nella generazione del testo: {e}")
    
def main():
    setup_logging()
    
    ######## HIERARCHICAL CLASSIFICATION EXAMPLE ##########
    # productions = { 'S*': ["<<positive >> A", "<<negative >> B", "<<neutral >> C"],
    #                 'A': ["<<happy>>", "<<peaceful>>", "<<joyful>>"],
    #                 'B': ['<<sad>>', '<<angry>>', '<<frustrated>>'],
    #                 'C': ['<<calm>>', '<<indifferent>>', '<<unemotional>>']
    #               }
    
    # system_prompt = """You are a hierarchical classification assistant. Your task is to classify the user input 
    #                     into one of the following hierarchical categories as shown in the followig examples\n\n"""

    # examples = [
    #     {"role": "user", "content": "I just got a promotion!"},
    #     {"role": "assistant", "content": "positive joyful"},

    #     {"role": "user", "content": "Nothing ever goes my way."},
    #     {"role": "assistant", "content": "negative frustrated"},

    #     {"role": "user", "content": "The lake was still and quiet."},
    #     {"role": "assistant", "content": "neutral calm"},

    #     {"role": "user", "content": "I miss my family so much."},
    #     {"role": "assistant", "content": "negative sad"}
    # ]

    # prompt=create_prompt(
    #     prompt_input="It's raining and I feel a bit down.",
    #     system_prompt=system_prompt,
    #     examples=examples
    # )
    #############################################################

    ######## VOCABOULARY RESTRICTION EXAMPLE ##########
    # productions = {
    # 'S*': [
    #     "<< Yes>> S*",
    #     "<< I'm>> S*",
    #     "<< very>> S*",
    #     "<< happy>> S*",
    #     "<< !>> S*",
    #     "<< so>> S*",
    #     "<< really>> S*",
    #     "<< excited>> S*",
    #     "<< today>> S*",
    #     "<< thanks>> S*",
    #     "<< you>> S*",
    #     "<< much>> S*",
    #     "<< great>> S*",
    #     "<< good>> S*",
    #     "<< fine>> S*",
    #     "<< amazing>> S*",
    #     ]
    # }

    # system_prompt = """You are a text generation assistant. When generating responses, you must use only the words that
    #                     appear in the provided examples below. You should not introduce any new words outside of those examples."""


    # examples = [
    #     {"role": "user", "content": "How are you?"},
    #     {"role": "assistant", "content": "I'm very happy"},

    #     {"role": "user", "content": "Is everything okay?"},
    #     {"role": "assistant", "content": "Yes, I'm so excited!"},

    #     {"role": "user", "content": "How was your day?"},
    #     {"role": "assistant", "content": "I'm really happy today!"},

    #     {"role": "user", "content": "Do you feel good?"},
    #     {"role": "assistant", "content": "Yes, I feel great, thanks!"},

    #     {"role": "user", "content": "What's up?"},
    #     {"role": "assistant", "content": "I'm fine, thank you so much!"},

    #     {"role": "user", "content": "Anything special today?"},
    #     {"role": "assistant", "content": "I'm very excited and happy today!"}
    #     ]
    
    # prompt=create_prompt(
    #     prompt_input="Say something of positive:",
    #     system_prompt=system_prompt,
    #     examples=examples
    #     )
    #############################################################

    ######## STRUCTURED GENERATION EXAMPLE ##########
    productions = {
        'S*': ["SUBJ PRED OBJ . S*"],
        'SUBJ': ["IRI", "BLANKNODE"],
        'PRED': ["IRI"],
        'OBJ': ["IRI", "BLANKNODE", "LITERAL"],
        'IRI': ["< URI >"],
        'BLANKNODE': ["<<_:>> NAME"],
        'LITERAL': ["\" STRING \" DESCRIPTION_LANG"],
        'DESCRIPTION_LANG': ["^^ IRI", "@ LANGTAG", "ε"],

        'URI': [
            # People
            "<<http://example.org/people/MarioRossi>>",
            "<<http://example.org/people/LuisaVerdi>>",
            "<<http://example.org/people/GiovanniBianchi>>",

            # Properties
            "<<http://example.org/properties/hasAge>>",
            "<<http://example.org/properties/hasProfession>>",
            "<<http://example.org/properties/hasSalary>>",

            # Other types or datatypes
            "<<http://www.w3.org/2001/XMLSchema#decimal>>",
            "<<http://www.w3.org/2001/XMLSchema#integer>>",
            "<<http://www.w3.org/2001/XMLSchema#string>>"
        ],

        'STRING':["alfanum STRING", "ε"],
        'NAME': ["ids NAME_C"],
        'NAME_C': ["idc NAME_C", "ε"],

        'LANGTAG': ['<<it >>', '<<en >>', '<<fr >>', '<<sp >>']
    }

    regex_alfanum = re.compile(r"[a-zA-Z0-9]+")  # es. "abc123"
    regex_right_round_bracket = re.compile(r"\)$")  # match only ')'
    regex_left_round_bracket = re.compile(r"\($")  # match only '('
    regex_less_than = re.compile(r"^<$") # match only '<'
    regex_greater_than = re.compile(r"^>$") # match only '>'
    regex_double_quote = re.compile(r'^\"$') # match only '"'
    regex_datatype = re.compile(r"^\^\^$")   # match only '^^'
    regex_langtag = re.compile(r"^@$")       # match only '@'
    regex_dot = re.compile(r"^\.$")  # match only '.'

    # Starting identifier: must start with a letter or an underscore
    regex_ids = re.compile(r'[A-Za-z_][A-Za-z0-9_-]*')
    # Continuation identifier: cannot start with a letter or an underscore
    regex_idc = re.compile(r'(?![A-Za-z_])[0-9_-][A-Za-z0-9_-]*')


    regex_dict = {
        'regex_alfanum': regex_alfanum,
        'regex_)': regex_right_round_bracket,
        'regex_(': regex_left_round_bracket,
        'regex_<': regex_less_than,
        'regex_>': regex_greater_than,
        'regex_"': regex_double_quote,
        'regex_^^': regex_datatype,
        'regex_@': regex_langtag,
        'regex_.': regex_dot,

        'regex_ids':regex_ids,
        'regex_idc':regex_idc

        }


    system_prompt = """You are an assistant that converts natural language sentences into RDF triples syntax.

    Follow these rules:

    1. Use URIs (`<...>`) for:
    - Identifiable entities such as people, properties, or concepts.
    - Example:
        <http://example.org/people/MarioRossi> <http://example.org/properties/hasFriend> <http://example.org/people/LuisaVerdi> .

    2. Use literals (`"..."`) for:
    - Plain values such as professions, cities, names, numbers, dates, or booleans.
    - Add datatypes (`^^<...>`) or language tags (`@lang`) if needed.
    - Examples:
        "engineer"@en  
        "40"^^<http://www.w3.org/2001/XMLSchema#integer>

    3. Use blank nodes (`_:`) only if:
    - The object is anonymous and has internal structure (i.e., it has its own properties).
    - Example:
        <http://example.org/people/MarioRossi> <http://example.org/properties/hasAddress> _:b1 .
        _:b1 <http://example.org/properties/street> "Via Roma" .
        _:b1 <http://example.org/properties/city> "Milano" .

    Never use a blank node (`_:`) for simple values like "engineer" or "teacher". Use a literal (`"..."`) instead.

    Now use the following examples to generate clean and correct RDF triples from user input."""

    examples = [
        {"role": "user", "content": "Mario Rossi is 40 years old."},
        {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasAge> \"40\" ^^<http://www.w3.org/2001/XMLSchema#integer> ."},

        {"role": "user", "content": "Luisa Verdi is an engineer."},
        {"role": "assistant", "content": "<http://example.org/people/LuisaVerdi> <http://example.org/properties/hasProfession> \"engineer\" @en ."},

        {"role": "user", "content": "Giovanni Bianchi earns 55000."},
        {"role": "assistant", "content": "<http://example.org/people/GiovanniBianchi> <http://example.org/properties/hasSalary> \"55000\" ^^<http://www.w3.org/2001/XMLSchema#decimal> ."},

        {"role": "user", "content": "Mario Rossi has an anonymous node as a contact."},
        {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasContact> _:ids ."},

        {"role": "user", "content": "Mario Rossi has the profession of teacher."},
        {"role": "assistant", "content": "<http://example.org/people/MarioRossi> <http://example.org/properties/hasProfession> \"teacher\" @en ."}
    ]

    prompt=create_prompt(
        prompt_input="Giovanni Bianchi was born 30 years ago.",
        system_prompt=system_prompt,
        examples=examples
    )

    # Initialize tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Generate grammar parameters
    pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(
        tokenizer, 
        productions=productions, 
        regex_dict=regex_dict,
    )

    LogitProcessor, Streamer = generate_grammar_parameters(tokenizer, pars_table, map_terminal_tokens)
    output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer)
    print(output) # Example output: "<http://example.org/people/GiovanniBianchi><http://example.org/properties/hasAge>"30"^^<http://www.w3.org/2001/XMLSchema#integer>."
  


if __name__ == "__main__":
    main()
