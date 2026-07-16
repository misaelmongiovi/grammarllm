"""
conll_eval.py
=============
CoNLL-2003 NER-to-JSON extraction with grammarllm constrained decoding.

Third benchmark of the suite (after wos and gloss_translation): structured
JSON generation with per-field closed enums. The grammar is compiled ONCE
over the FULL taxonomy (every entity span observed in train+validation+test,
~20k enum values — same single-compile approach as the 16k-gloss grammar),
via the pydantic_to_grammar pipeline:

    dynamic Pydantic model (Literal enums per field)
        └─► pydantic_to_productions
              └─► get_parsing_table_and_map_tt
                    └─► generate_text (greedy+lookahead / beam3+lookahead)

Few-shot: dynamic top-k retrieval over the embedded train pool
(mxbai-embed-large-v1), most similar first — same scheme as gloss_translation.

Usage
-----
    python conll_eval.py --name 1b_la_greedy
    python conll_eval.py --name 1b_la_beam3 --num-beams 3
    python conll_eval.py --name smoke --limit 5

Resume an interrupted run:
    python conll_eval.py --name 1b_la_greedy --start N --resume-csv output/1b_la_greedy/checkpoint.csv
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pydantic import create_model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
)
from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions

SCRIPT_DIR = Path(__file__).parent

FIELDS = ("person", "organization", "location", "misc")


# ---------------------------------------------------------------------------
# Grammar from the full closed taxonomy (compiled once)
# ---------------------------------------------------------------------------

def build_taxonomy_grammar(tokenizer, taxonomy: dict[str, list[str]]):
    """Dynamic Pydantic model with one Literal enum per field ('' = absent),
    translated to productions and compiled into the parsing table."""
    field_defs = {
        f: (Literal[tuple([""] + taxonomy[f])], ...)  # type: ignore[valid-type]
        for f in FIELDS
    }
    ConllNER = create_model("ConllNER", **field_defs)

    t0 = time.time()
    productions, regex_dict = pydantic_to_productions(ConllNER)
    pars_tab, map_tt = get_parsing_table_and_map_tt(tokenizer, productions, regex_dict)
    n_values = sum(len(v) for v in taxonomy.values())
    print(f"Grammar ready in {time.time() - t0:.1f}s ({n_values} enum values)")
    return pars_tab, map_tt


# ---------------------------------------------------------------------------
# Retrieval (same logic as gloss_translation)
# ---------------------------------------------------------------------------

class Retriever:
    """Cosine-similarity retrieval over a pre-embedded pool DataFrame."""

    def __init__(self, df: pd.DataFrame, text_cols: list[str], device: str):
        self.columns = {c: df[c].tolist() for c in text_cols}
        emb = torch.tensor(np.vstack(df["embedding"].values), dtype=torch.float32,
                           device=device)
        self.embeddings = torch.nn.functional.normalize(emb, p=2, dim=1)
        self.device = device

    def top_n(self, query_emb: np.ndarray, n: int) -> dict[str, list]:
        q = torch.tensor(query_emb.reshape(1, -1), dtype=torch.float32,
                         device=self.device)
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        similarities = (self.embeddings @ q.T).flatten()
        idx = torch.topk(similarities, n).indices.cpu().numpy()
        return {c: [vals[i] for i in idx] for c, vals in self.columns.items()}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def gold_json(example: dict[str, str]) -> str:
    # json.dumps default separators (', ', ': ') match the grammar skeleton
    # chunks exactly; ensure_ascii=False keeps raw chars as in the enum values.
    return json.dumps({f: example[f] for f in FIELDS}, ensure_ascii=False)


def build_messages(input_sentence: str, system_prompt: str,
                   examples: dict[str, list]) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for i in range(len(examples["sentence"])):
        messages.append({"role": "user", "content": examples["sentence"][i].strip()})
        messages.append({"role": "assistant",
                         "content": gold_json({f: examples[f][i] for f in FIELDS})})
    messages.append({"role": "user",
                     "content": "Extract the entities from the following sentence "
                                "as shown in the previous examples"})
    messages.append({"role": "user", "content": input_sentence.strip()})
    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--name", required=True, help="Run label (output subfolder)")
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--num-beams", type=int, default=None, help="Override num_beams")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--start", type=int, default=0, help="Row index to resume from")
    parser.add_argument("--resume-csv", default=None,
                        help="Partial predictions csv whose first --start rows are merged")
    args = parser.parse_args()

    cfg = OmegaConf.load(SCRIPT_DIR / args.config)
    model_path = args.model or cfg.model.name
    num_beams = args.num_beams if args.num_beams is not None else cfg.model.num_beams
    token_lookahead = bool(cfg.model.token_lookahead)

    out_dir = SCRIPT_DIR / cfg.paths.output_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=str(out_dir / "temp"))
    logging.info(f"Run '{args.name}': model={model_path} beams={num_beams} "
                 f"lookahead={token_lookahead}")

    device = cfg.model.device if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    if tokenizer.chat_template is None:
        raise ValueError(f"{model_path} has no native chat template.")

    print("Loading taxonomy and building the grammar (full taxonomy, takes a while)...")
    with open(SCRIPT_DIR / cfg.paths.taxonomy, encoding="utf-8") as fh:
        taxonomy = json.load(fh)
    pars_tab, map_tt = build_taxonomy_grammar(tokenizer, taxonomy)

    print("Loading retrieval pool...")
    pool_df = pd.read_parquet(SCRIPT_DIR / cfg.paths.retrieval_pool)
    pool = Retriever(pool_df, ["sentence", *FIELDS], device=cfg.embedder.device)

    print(f"Loading encoder: {cfg.embedder.name}")
    encoder = SentenceTransformer(cfg.embedder.name, device=cfg.embedder.device)

    system_prompt = (SCRIPT_DIR / "prompt.txt").read_text()

    test_df = pd.read_csv(SCRIPT_DIR / cfg.paths.test_data, keep_default_na=False)
    done_df = pd.read_csv(args.resume_csv, keep_default_na=False) if args.resume_csv else None
    if args.start > 0:
        test_df = test_df.iloc[args.start:]
    if args.limit is not None:
        test_df = test_df.head(args.limit)
    test_df = test_df.reset_index(drop=True)
    print(f"Processing {len(test_df)} test sentences.")

    predictions = []
    checkpoint_path = out_dir / "checkpoint.csv"

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        sentence = row["sentence"].strip()

        query_emb = encoder.encode(sentence, normalize_embeddings=True)
        examples = pool.top_n(query_emb, cfg.retrieval.n_examples)
        messages = build_messages(sentence, system_prompt, examples)

        pdas, streamer = generate_grammar_parameters(
            tokenizer, pars_tab, map_tt, token_lookahead=token_lookahead)

        # Beam extra: passati SOLO se presenti in config, altrimenti default HF
        # (condizioni identiche a greedy: nessun parametro aggiuntivo).
        gen_kwargs = {}
        if num_beams > 1:
            lp = OmegaConf.select(cfg, "model.length_penalty", default=None)
            if lp is not None:
                gen_kwargs["length_penalty"] = float(lp)
            es = OmegaConf.select(cfg, "model.early_stopping", default=None)
            if es is not None:
                gen_kwargs["early_stopping"] = bool(es)

        result = generate_text(
            model, tokenizer, messages, pdas, streamer,
            max_new_tokens=cfg.model.max_new_tokens,
            do_sample=bool(cfg.model.do_sample),
            num_beams=num_beams,
            return_pda_stack=False,
            output_scores=False,
            **gen_kwargs,
        )
        prediction = (result if isinstance(result, str) else result["text"]).strip()
        predictions.append(prediction)
        logging.info(f"INPUT: {sentence}\nPRED: {prediction}")
        tqdm.write(f"PRED: {prediction}")

        if len(predictions) % int(cfg.run.checkpoint_every) == 0:
            ckpt = test_df.iloc[:len(predictions)].copy()
            ckpt["prediction"] = predictions
            ckpt.to_csv(checkpoint_path, index=False)

    test_df["prediction"] = predictions
    if done_df is not None:
        test_df = pd.concat([done_df.iloc[:args.start], test_df], ignore_index=True)

    out_path = out_dir / f"predictions_{args.name}.csv"
    test_df.to_csv(out_path, index=False)
    print(f"Done. Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
