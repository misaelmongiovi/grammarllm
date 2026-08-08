"""
pipe_bench.py
=============
SMILES molecule generation through grammarllm constrained decoding, scored
by QED (Bickerton et al., 2012) — mirrors this suite's other tasks (wos,
gloss_translation, conll_ner): grammar-constrained decoding, greedy vs beam3,
token-lookahead engine on by default.

Setup, adapted from the reference paper spec (few-shot molecule generation,
GDB-17, QED, base LM Llama; see README.md "Deviations" for the exact list):
  * dataset: jablonkagroup/pubchem-smiles-molecular-formula (not GDB-17)
  * base LM: Llama-3.2-1B-Instruct (not Llama 3.1 8B)
  * 20 fixed few-shot examples (not resampled per generation)
  * 2000-row fixed test set, split from the same source

Deterministic decoding (do_sample=False, matching this repo's convention)
needs a per-row input to differ, or every row's output would be identical
under greedy AND beam3. The dataset's molecular_formula column supplies
that: each test row conditions generation on ITS OWN formula, few-shot
examples are (formula -> SMILES) pairs from the fixed 20-example pool. This
is a deviation from the paper's unconditional generation — documented, not
silent — chosen specifically so beam3 vs greedy is a meaningful comparison
under this repo's decoding convention rather than 2000 copies of one greedy
argmax.

Usage
-----
    python pipe_bench.py --name greedy
    python pipe_bench.py --name beam3 --num-beams 3
    python pipe_bench.py --name smoke --limit 10
"""
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
from smiles import build_smiles_grammar  # noqa: E402

SCRIPT_DIR = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a medicinal chemist. Given a molecular formula, respond with "
    "exactly one valid, drug-like SMILES string that matches that formula "
    "and nothing else."
)


def build_messages(formula: str, few_shot: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shot:
        messages.append({"role": "user", "content": ex["molecular_formula"]})
        messages.append({"role": "assistant", "content": ex["smiles"]})
    messages.append({"role": "user", "content": formula})
    return messages


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--name", default=None, help="Run label (output subfolder); "
                        "defaults to beam{num_beams}")
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--num-beams", type=int, default=None, help="Override num_beams")
    parser.add_argument("--dataset", default=None,
                        help="Override cfg.dataset: 'pubchem' or 'gdb17'")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override cfg.model.batch_size")
    parser.add_argument("--no-lookahead", action="store_true",
                        help="Disable the token-lookahead engine (A/B baseline)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--start", type=int, default=0, help="Row index to resume from")
    parser.add_argument("--resume-csv", default=None,
                        help="Partial predictions csv whose first --start rows are merged")
    args = parser.parse_args()

    cfg = OmegaConf.load(SCRIPT_DIR / args.config)
    model_path = args.model or cfg.model.name
    num_beams = args.num_beams if args.num_beams is not None else cfg.model.num_beams
    token_lookahead = (not args.no_lookahead) and bool(cfg.model.token_lookahead)
    run_name = args.name or f"beam{num_beams}"
    dataset = args.dataset or cfg.dataset

    out_dir = SCRIPT_DIR / cfg.paths.output_dir / dataset / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=str(out_dir / "temp"))
    logging.info(f"Run '{run_name}': dataset={dataset} model={model_path} "
                 f"beams={num_beams} lookahead={token_lookahead}")

    device = cfg.model.device if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    if tokenizer.chat_template is None:
        raise ValueError(f"{model_path} has no native chat template.")

    print("Building the OpenSMILES parsing table (single compile)...")
    t0 = time.time()
    productions, regex_dict = build_smiles_grammar()
    pars_tab, map_tt = get_parsing_table_and_map_tt(tokenizer, productions, regex_dict=regex_dict)
    print(f"Grammar ready in {time.time() - t0:.1f}s")

    data_dir = SCRIPT_DIR / "data" / dataset
    few_shot = load_jsonl(data_dir / cfg.paths.few_shot)
    test_rows = load_jsonl(data_dir / cfg.paths.test_data)
    if args.start > 0:
        test_rows = test_rows[args.start:]
    if args.limit is not None:
        test_rows = test_rows[: args.limit]
    batch_size = int(args.batch if args.batch is not None else cfg.model.batch_size)
    print(f"Processing {len(test_rows)} {dataset} rows (batch={batch_size}).")

    predictions = []
    checkpoint_path = out_dir / "checkpoint.csv"

    for i in tqdm(range(0, len(test_rows), batch_size), desc="Inference"):
        batch = test_rows[i:i + batch_size]
        messages_batch = [build_messages(row["molecular_formula"], few_shot) for row in batch]

        pdas, streamer = generate_grammar_parameters(
            tokenizer, pars_tab, map_tt, token_lookahead=token_lookahead,
        )

        result = generate_text(
            model, tokenizer, messages_batch, pdas, streamer,
            max_new_tokens=cfg.model.max_new_tokens,
            do_sample=bool(cfg.model.do_sample),
            num_beams=num_beams,
            return_pda_stack=True,
        )
        # batch_prompts > 1 -> list[dict], one dict per prompt (each dict has
        # "text" and "pda_stack" since n_ret=1, return_pda_stack=True).
        results = result if isinstance(result, list) else [result]

        for row, item in zip(batch, results):
            pred = item["text"].strip()
            stack = item.get("pda_stack")
            predictions.append({
                "molecular_formula": row["molecular_formula"],
                "gold_smiles": row["smiles"],
                "pred_smiles": pred,
                "grammar_satisfied": (stack == [] or stack is None),
            })
            logging.info(f"FORMULA: {row['molecular_formula']}  PRED: {pred}")

        if len(predictions) % int(cfg.run.checkpoint_every) < batch_size:
            pd.DataFrame(predictions).to_csv(checkpoint_path, index=False)

    df = pd.DataFrame(predictions)
    if args.resume_csv:
        prior = pd.read_csv(args.resume_csv)
        df = pd.concat([prior, df], ignore_index=True)
    out_csv = out_dir / "predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} predictions to {out_csv}")


if __name__ == "__main__":
    main()
