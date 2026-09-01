"""
unconstrained_baseline.py
=========================
Same model, same prompts, same deterministic decoding — but WITHOUT the
grammar mask. This is the A/B that says whether constrained decoding helps
or hurts on this task, which is the whole point of the benchmark suite.

Usage:
    python unconstrained_baseline.py --name nogram_beam1
    python unconstrained_baseline.py --name nogram_beam3 --num-beams 3
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from pipe_bench import build_messages, load_jsonl, SCRIPT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(SCRIPT_DIR / args.config)
    out_dir = SCRIPT_DIR / cfg.paths.output_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=torch.bfloat16)
    model.to(cfg.model.device).eval()

    few_shot = load_jsonl(SCRIPT_DIR / cfg.paths.few_shot)
    rows = load_jsonl(SCRIPT_DIR / cfg.paths.test_data)
    if args.limit:
        rows = rows[: args.limit]

    bs = int(cfg.model.batch_size)
    preds = []
    for i in tqdm(range(0, len(rows), bs), desc="Inference (no grammar)"):
        batch = rows[i:i + bs]
        msgs = [build_messages(r["molecular_formula"], few_shot) for r in batch]
        enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, padding=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=cfg.model.max_new_tokens,
                                 do_sample=False, num_beams=args.num_beams,
                                 pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for r, g in zip(batch, gen):
            preds.append({
                "molecular_formula": r["molecular_formula"],
                "gold_smiles": r["smiles"],
                "pred_smiles": tok.decode(g, skip_special_tokens=True).strip(),
                # no grammar involved: the column exists so score_bench.py can
                # read this file with the same schema
                "grammar_satisfied": False,
            })

    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)
    print(f"Wrote {len(preds)} predictions to {out_dir/'predictions.csv'}")


if __name__ == "__main__":
    main()
