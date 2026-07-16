"""
gloss_eval.py
=============
ASLG-PC12 text-to-gloss translation with grammarllm constrained decoding.

Replicates the paper setup (GramDec/main_t2_cons.py) — same test set, same
mxbai encoder, dynamic top-30 few-shot retrieval, top-50 gloss hints in the
system prompt — while decoding through the grammarllm library so that the
new token-lookahead engine and beam search can be evaluated.

Intentional deviations from the paper run (documented in README.md):
  * no '@@' separator: glosses are separated by plain spaces (the lookahead
    engine handles BPE tokens that span gloss boundaries);
  * native chat template (the custom template harmed small models, see
    benchmark_tests/wos).

Gloss form (config grammar.gloss_form):
  * 'suffix' (default, paper setting): semantic_switch rewrite, DESC-X → X-DESC,
    X-Y → Y-X. Not just a BPE workaround: with greedy constrained decoding the
    model picks the content stem BEFORE the tag. In prefix form it must commit
    to 'DESC-' first, and when its intended continuation (e.g. DESC-QUESTION)
    is not in the vocabulary, the mask forces a wrong gloss (DESC-QUESTIONABLE)
    and the output degenerates.
  * 'prefix': original ASLG form, kept as an ablation of the above.

Usage
-----
    python gloss_eval.py --name 1b_la_greedy
    python gloss_eval.py --name 1b_la_beam3 --num-beams 3
    python gloss_eval.py --name smoke --limit 5

Resume an interrupted run:
    python gloss_eval.py --name 1b_la_greedy --start N --resume-csv output/1b_la_greedy/checkpoint.csv
"""

import argparse
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
)
SCRIPT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# semantic_switch (paper transform, GramDec/main_t2_cons.py)
# ---------------------------------------------------------------------------

def semantic_switch_gloss(gloss: str) -> str:
    """DESC-X → X-DESC, X-Y → Y-X, plus the paper's corpus typo fixes."""
    gloss = re.sub(r"\bDESC-N\b", "DESC-THAN", gloss)
    gloss = re.sub(r"\bDESC-RE\b", "DESC-THERE", gloss)
    gloss = re.sub(r"\bDESC-REFORE\b", "DESC-THEREFORE", gloss)
    if "DESC-" in gloss:
        return gloss.replace("DESC-", "") + "-DESC"
    if "X-" in gloss:
        return gloss.replace("X-", "") + "-X"
    return gloss


def semantic_switch_sequence(sequence: str) -> str:
    return " ".join(semantic_switch_gloss(g) for g in sequence.split(" ") if g)


# ---------------------------------------------------------------------------
# Retrieval (same logic as GramDec find_most_similar_*_gpu)
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
# Prompt construction (paper format, minus the '@@' rule)
# ---------------------------------------------------------------------------

def build_messages(input_sentence: str, system_prompt: str,
                   sentences: list[str], decompositions: list[str]) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for sentence, decomposition in zip(sentences, decompositions):
        if not sentence or not decomposition:
            logging.warning("Missing sentence or decomposition in few-shot pool.")
            continue
        messages.append({"role": "user", "content": sentence.replace("\n", "").strip()})
        messages.append({"role": "assistant", "content": decomposition.replace("\n", "").strip()})
    messages.append({"role": "user",
                     "content": "Decompose the following sentence as shown in the previous examples"})
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

    gloss_form = str(getattr(cfg.grammar, "gloss_form", "suffix"))
    if gloss_form not in ("suffix", "prefix"):
        raise ValueError(f"Unsupported grammar.gloss_form={gloss_form!r}")
    print(f"Gloss form: {gloss_form}")

    print("Loading retrieval pool and gloss vocabulary...")
    pool_df = pd.read_parquet(SCRIPT_DIR / cfg.paths.retrieval_pool)
    vocab_df = pd.read_parquet(SCRIPT_DIR / cfg.paths.gloss_vocab)
    vocab_df["word"] = vocab_df["word"].astype(str).str.strip()
    vocab_df = vocab_df[vocab_df["word"] != ""]
    if gloss_form == "suffix":
        pool_df["decomposition"] = pool_df["decomposition"].map(semantic_switch_sequence)
        vocab_df["word"] = vocab_df["word"].map(semantic_switch_gloss)

    pool = Retriever(pool_df, ["sentence", "decomposition"], device=cfg.embedder.device)
    vocab = Retriever(vocab_df, ["word"], device=cfg.embedder.device)

    print("Building parsing table and token maps (16k glosses, takes a while)...")
    t0 = time.time()
    terminals = sorted(set(vocab_df["word"]))
    # Space is its own separator nonterminal (wos-style): the lookahead engine
    # works out which stacked terminals a merged token consumes, so the
    # separator does not need to live inside the gloss tag.
    # The separator sits BETWEEN glosses only: after a gloss, GLOSSEND offers
    # either 'space + more glosses' or eot directly. With a trailing separator
    # ('<<g>> WSEP S*') the model could emit eot only AFTER a lone trailing
    # space token — p(' ' | "... .") is tiny for Llama, so no beam hypothesis
    # ever finished and greedy looped on ' .' to max_new_tokens (17/50 rows).
    # NB: NT names must be uppercase and must not equal any tokenizer token
    # string — 'WS' collides with the 'WS' token inside LAWSUIT
    # ['LA','WS','UIT'] and produces a bogus LL(1) conflict (Ġ vs epsilon);
    # 'SEP', 'SPACE' and 'CONT' are tokens too.
    sep_nt = "GLOSSEND"
    assert sep_nt not in tokenizer.get_vocab(), \
        f"Separator NT name '{sep_nt}' collides with a tokenizer token"
    productions = {
        "S*": [f"<<{w}>> {sep_nt}" for w in terminals],
        sep_nt: ["<< >> S*", f"<<{tokenizer.eos_token}>>"],
    }
    pars_tab, map_tt = get_parsing_table_and_map_tt(tokenizer, productions)
    print(f"Grammar ready in {time.time() - t0:.1f}s ({len(terminals)} glosses)")

    print(f"Loading encoder: {cfg.embedder.name}")
    encoder = SentenceTransformer(cfg.embedder.name, device=cfg.embedder.device)

    system_prompt_template = (SCRIPT_DIR / "prompt.txt").read_text()

    test_df = pd.read_csv(SCRIPT_DIR / cfg.paths.test_data)
    done_df = pd.read_csv(args.resume_csv) if args.resume_csv else None
    if args.start > 0:
        test_df = test_df.iloc[args.start:]
    if args.limit is not None:
        test_df = test_df.head(args.limit)
    test_df = test_df.reset_index(drop=True)
    print(f"Processing {len(test_df)} test sentences.")

    predictions = []
    checkpoint_path = out_dir / "checkpoint.csv"

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        sentence = row["text"].replace("\n", "").strip()

        query_emb = encoder.encode(sentence, normalize_embeddings=True)
        examples = pool.top_n(query_emb, cfg.retrieval.n_examples)
        hints = vocab.top_n(query_emb, cfg.retrieval.n_gloss_hints)["word"]

        system_prompt = system_prompt_template.format(
            similar_glosses='", "'.join(str(g) for g in hints))
        messages = build_messages(sentence, system_prompt,
                                  examples["sentence"], examples["decomposition"])

        pdas, streamer = generate_grammar_parameters(
            tokenizer, pars_tab, map_tt, token_lookahead=token_lookahead)

        # Beam extra: passati SOLO se presenti in config, altrimenti default HF
        # (condizioni identiche a greedy/paper: nessun parametro aggiuntivo).
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
    if gloss_form == "suffix":
        test_df["gloss_adj"] = test_df["gloss"].map(
            lambda g: semantic_switch_sequence(str(g).replace("\n", " ")))
    if done_df is not None:
        test_df = pd.concat([done_df.iloc[:args.start], test_df], ignore_index=True)

    out_path = out_dir / f"predictions_{args.name}.csv"
    test_df.to_csv(out_path, index=False)
    print(f"Done. Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
