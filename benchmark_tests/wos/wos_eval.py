import importlib.util
import os
import pandas as pd
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
    create_prompt,
    chat_template,
)
from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions

from wos_schema import WosClassification

# Path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# Maps n_shot -> fixed few-shot example file shipped alongside this script
FEW_SHOT_FILES = {
    0: None,
    1: os.path.join(script_dir, "few_shot1.py"),
    10: os.path.join(script_dir, "few_shot10.py"),
}


def load_few_shot_examples(n_shot):
    """Load the `few_shot` list from few_shot{n_shot}.py, or [] for 0-shot."""
    if n_shot not in FEW_SHOT_FILES:
        raise ValueError(f"Unsupported n_shot={n_shot}. Choose one of {sorted(FEW_SHOT_FILES)}.")

    file_path = FEW_SHOT_FILES[n_shot]
    if file_path is None:
        return []

    spec = importlib.util.spec_from_file_location(f"few_shot_{n_shot}", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.few_shot

def main():
    setup_logging(log_dir=os.path.join(script_dir, "output", "temp"))
    load_dotenv()
    
    # Path to the specific WoS config directory
    #enhanced_config_path = os.path.join(script_dir, "config_enhanced.yaml")
    base_config_path = os.path.join(script_dir, "config.yaml")
    
    # if os.path.exists(enhanced_config_path):
    #     print(f"Using ENHANCED config from {enhanced_config_path}")
    #     config_path = enhanced_config_path
    # else:
    print(f"Using BASE config from {base_config_path}")
    config_path = base_config_path
        
    cfg = OmegaConf.load(config_path)
    
    print(f"Initializing Model: {cfg.model.name}...")
    # Initialize generator (loads model and tokenizer via transformers)
    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # grammarllm's exported `chat_template` renders <|system|>/<|user|>/<|assistant|>.
    # Those are NOT Llama-3 special tokens: they shatter into '<','|','system','|','>'
    # and the model sees a chat format it was never trained on. Small models stop
    # classifying and start echoing the system prompt back. Default to the
    # tokenizer's own template; set model.chat_template: grammarllm to A/B it.
    which_template = getattr(cfg.model, "chat_template", "native")
    if which_template == "native":
        if tokenizer.chat_template is None:
            raise ValueError(f"{model_name} has no native chat template; "
                             "set model.chat_template: grammarllm in config.yaml")
        active_chat_template = tokenizer.chat_template
    elif which_template == "grammarllm":
        active_chat_template = chat_template
    else:
        raise ValueError(f"Unsupported model.chat_template={which_template!r}. "
                         "Choose 'native' or 'grammarllm'.")
    print(f"Chat template: {which_template}")

    configured_device = getattr(cfg.model, "device", "mps")
    if configured_device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"
    else:
        device = configured_device

    model.to(device)
    model.eval()

    # Prepare grammar parameters
    print("Building Parsing Table and Mapping Tokens...")
    grammar_mode = getattr(cfg.grammar, "mode", "tags")
    if grammar_mode == "json":
        print("Using JSON-notation grammar (wos_schema.WosClassification)...")
        productions, regex_dict = pydantic_to_productions(WosClassification)
        pars_table, map_tt = get_parsing_table_and_map_tt(
            tokenizer, productions=productions, regex_dict=regex_dict
        )
    elif grammar_mode == "tags":
        productions = OmegaConf.to_container(cfg.grammar.productions, resolve=True)
        pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions=productions)
    else:
        raise ValueError(f"Unsupported grammar.mode={grammar_mode!r}. Choose 'tags' or 'json'.")
    
    # Load test data - relative to project root
    test_data_path = os.path.join(project_root, "data/WebOfScience/test_data.csv")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Error: {test_data_path} not found.")
        
    test_df = pd.read_csv(test_data_path)
    print(f"Loaded {len(test_df)} rows from {test_data_path}")
    
    # Configuration for batching
    batch_size = int(getattr(cfg.model, "batch_size", 1))
    num_beams = int(getattr(cfg.model, "num_beams", 3))
    do_sample = bool(getattr(cfg.model, "do_sample", True))
    token_lookahead = bool(getattr(cfg.model, "token_lookahead", True))
    test_df['pred'] = None
    
    # Output directory
    folder_path = os.path.join(script_dir, "output")
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, "test_predictions_wos.csv")
    
    # Prepare system prompt and examples from config
    system_prompt = cfg.prompt.system_prompt
    if grammar_mode == "json":
        system_prompt += (
            "\n\nRespond only with a JSON object: "
            '{"parent": <top-level category>, "child": <subcategory>}.'
        )
    n_shot = int(getattr(cfg.prompt, "n_shot", 0))
    examples = load_few_shot_examples(n_shot)
    print(f"Using {n_shot}-shot prompting ({len(examples)} example messages).")

    print(f"Starting batch generation (batch_size={batch_size})...")
    
    for i in range(0, len(test_df), batch_size):
        batch_slice = test_df.iloc[i : i + batch_size]
        abstracts = batch_slice['Abstract'].tolist()
        
        # Create prompts for the batch
        batch_prompts = []
        for abstract in abstracts:
            p = create_prompt(
                prompt_input=abstract,
                system_prompt=system_prompt,
                examples=examples,
            )
            batch_prompts.append(p)
        
        # Initialize PDA/Streamer for the batch
        # Note: num_return_sequences=1 for classification usually
        pdas, streamer = generate_grammar_parameters(
            tokenizer,
            pars_table,
            map_tt,
            token_lookahead=token_lookahead,
        )
        
        print(f"Processing batch {i//batch_size + 1}/{(len(test_df)+batch_size-1)//batch_size}...")
        
        # Run constrained generation in batch
        output = generate_text(
            model,
            tokenizer,
            batch_prompts,
            pdas,
            streamer,
            active_chat_template,
            do_sample=do_sample,
            num_beams=num_beams,
            # num_return_sequences=1,
            max_new_tokens=400, # Max tokens for the classification path
            output_scores=False
        )


        print(f"\noutput: {output}")
        # generate_text returns a bare dict when batch_prompts == 1, and a
        # list of dicts otherwise — normalize to a list before indexing.
        out_list = output if isinstance(output, list) else [output]
        for idx, out in enumerate(out_list):
            test_df.at[i + idx, 'pred'] = out
        
        # Save progress periodically
        test_df.to_csv(output_path, index=False)
            
    print(f"\nFinished! Results saved to {output_path}")

if __name__ == "__main__":
    main()