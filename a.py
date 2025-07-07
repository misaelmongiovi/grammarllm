from transformers import AutoTokenizer, AutoModelForCausalLM
from grammarllm import generate_grammar_parameters, generate_text, get_parsing_table_and_map_tt, setup_logging, create_prompt, chat_template

setup_logging()

productions = { 'S*': ["<<positive >> A", "<<negative >> B", "<<neutral >> C"],
                'A': ["<<happy>>", "<<peaceful>>", "<<joyful>>"],
                'B': ['<<sad>>', '<<angry>>', '<<frustrated>>'],
                'C': ['<<calm>>', '<<indifferent>>', '<<unemotional>>']
                }

system_prompt = """You are a hierarchical classification assistant. Your task is to classify the user input 
                    into one of the following hierarchical categories as shown in the followig examples\n\n"""

examples = [
    {"role": "user", "content": "I just got a promotion!"},
    {"role": "assistant", "content": "positive joyful"},

    {"role": "user", "content": "Nothing ever goes my way."},
    {"role": "assistant", "content": "negative frustrated"},

    {"role": "user", "content": "The lake was still and quiet."},
    {"role": "assistant", "content": "neutral calm"},

    {"role": "user", "content": "I miss my family so much."},
    {"role": "assistant", "content": "negative sad"}
]

prompt=create_prompt(
    prompt_input="It's raining and I feel a bit down.",
    system_prompt=system_prompt,
    examples=examples
)


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

pars_table, map_terminal_tokens = get_parsing_table_and_map_tt(tokenizer, productions)

LogitProcessor, Streamer = generate_grammar_parameters(
    tokenizer, pars_table, map_terminal_tokens
)

output = generate_text(model, tokenizer, prompt, LogitProcessor, Streamer, chat_template)
print(output)  # → "negative sad"