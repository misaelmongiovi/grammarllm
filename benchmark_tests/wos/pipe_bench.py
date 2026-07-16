"""
pipe_bench.py — WoS con separatore PIPE fra i due livelli.

Grammatica:  S* -> <<parent>> SEP <<child>>   con SEP = '|'
Prompt:      lista piatta di tutti i percorsi "parent|child" (wording GramDec)
Esempi:      le risposte di few_shot{1,10}.py convertite in forma pipe
             ("medical medicare" -> "medical|medicare").  Con 0-shot nessun esempio.

Il separatore '|' non compare dentro nessuna etichetta, quindi il confine fra i due
livelli e' non ambiguo — a differenza dello spazio, che collide con gli spazi interni
ai figli ("low testosterone").

usage:
  pipe_bench.py --model PATH --nshot {0,1,10} --out FILE
                [--beams 3] [--sample] [--no-lookahead] [--rows N] [--batch 5]
"""
import argparse, importlib.util, logging, os, sys, time

import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from grammarllm import (get_parsing_table_and_map_tt, generate_grammar_parameters,
                        generate_text, create_prompt)

SEP = '|'
NT = {'biochemistry': 'A', 'civil': 'B', 'cs': 'C', 'ece': 'D',
      'mae': 'E', 'medical': 'F', 'psychology': 'G'}


def build_grammar(cfg):
    """S* -> <<parent>> SEP <<child>>, con i figli ristretti al proprio parent."""
    base = OmegaConf.to_container(cfg.grammar.productions, resolve=True)
    prods = {'S*': [f'<<{p}>> SEP {nt}' for p, nt in NT.items()],
             'SEP': [f'<<{SEP}>>']}
    for nt in NT.values():
        prods[nt] = base[nt]
    labels = [f'{p}{SEP}{c[2:-2]}' for p, nt in NT.items() for c in base[nt]]
    return prods, labels


def build_system_prompt(labels):
    return (
        "You are a hierarchical classification assistant. Your task is to classify user inputs "
        "into one of the following hierarchical categories:\n\n"
        f"{labels}\n\n"
        "Respond only with the most specific label from the hierarchy based on the user input. "
        "If uncertain, choose the most appropriate parent category."
    )


def load_examples(nshot):
    """Esempi few-shot con la risposta convertita in forma pipe. 0-shot -> nessuno."""
    if nshot == 0:
        return []
    path = os.path.join(HERE, f'few_shot{nshot}.py')
    if not os.path.exists(path):
        raise SystemExit(f"nshot={nshot}: manca {path}")
    spec = importlib.util.spec_from_file_location(f'few_shot_{nshot}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = []
    for msg in mod.few_shot:
        msg = dict(msg)
        if msg['role'] == 'assistant':
            parent, _, child = msg['content'].strip().partition(' ')
            msg['content'] = f'{parent}{SEP}{child}'
        out.append(msg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--nshot', type=int, required=True, choices=[0, 1, 10])
    ap.add_argument('--out', required=True)
    ap.add_argument('--beams', type=int, default=3)
    ap.add_argument('--sample', action='store_true', default=True)
    ap.add_argument('--no-sample', dest='sample', action='store_false')
    ap.add_argument('--no-lookahead', dest='lookahead', action='store_false', default=True)
    ap.add_argument('--rows', type=int, default=0, help='0 = tutte')
    ap.add_argument('--batch', type=int, default=5)
    args = ap.parse_args()

    logging.disable(logging.INFO)
    name = os.path.basename(args.model)
    print(f">>> {name} | {args.nshot}-shot | beams={args.beams} sample={args.sample} "
          f"lookahead={args.lookahead} | sep='{SEP}'", flush=True)

    cfg = OmegaConf.load(os.path.join(HERE, 'config.yaml'))
    prods, labels = build_grammar(cfg)
    system_prompt = build_system_prompt(labels)
    examples = load_examples(args.nshot)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.chat_template is None:
        raise SystemExit(f"{name} non ha un chat template nativo (modello base?)")
    native_template = tok.chat_template      # MAI grammarllm.chat_template: non e' llama-3

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = model.to('cuda').eval()
    pars_table, map_tt = get_parsing_table_and_map_tt(tok, productions=prods)

    df = pd.read_csv(os.path.join(ROOT, 'data/WebOfScience/test_data.csv'))
    if args.rows:
        df = df.head(args.rows)
    df['pred'] = None
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    t0 = time.time()
    for i in range(0, len(df), args.batch):
        chunk = df.iloc[i:i + args.batch]
        prompts = [create_prompt(prompt_input=a, system_prompt=system_prompt,
                                 examples=examples) for a in chunk.Abstract]
        pdas, streamer = generate_grammar_parameters(
            tok, pars_table, map_tt, token_lookahead=args.lookahead)
        out = generate_text(model, tok, prompts, pdas, streamer, native_template,
                            max_new_tokens=400, do_sample=args.sample,
                            num_beams=args.beams)
        # generate_text restituisce un dict quando c'e' un solo prompt, una lista altrimenti
        for j, o in enumerate(out if isinstance(out, list) else [out]):
            df.at[i + j, 'pred'] = o

        if (i // args.batch) % 40 == 0:
            el = time.time() - t0
            done = i + len(chunk)
            eta = el / max(done, 1) * (len(df) - done)
            print(f"  {done}/{len(df)}  ({el/60:.1f}m trascorsi, ~{eta/60:.0f}m rimanenti)",
                  flush=True)
            df.to_csv(args.out, index=False)

    df.to_csv(args.out, index=False)
    print(f"DONE {args.out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
