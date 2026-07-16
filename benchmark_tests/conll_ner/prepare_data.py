"""
prepare_data.py
===============
One-time preparation of the CoNLL-2003 NER-to-JSON benchmark data.

Task: given a sentence, produce a JSON object with the FIRST occurrence of
each entity type:
    {"person": "...", "organization": "...", "location": "...", "misc": ""}
Missing types use the empty string.

The value vocabulary is closed over the FULL taxonomy: every entity span
observed in train+validation+test becomes an enum alternative in the grammar
(same single-compile approach as the gloss benchmark, which handles a 16k
vocabulary). Spans containing a quote, backslash or control character cannot
be expressed by the v1 JSON grammar (no escape sequences); samples containing
them are dropped and counted.

Outputs (benchmark_tests/conll_ner/data/):
    taxonomy.json           {field: [span, ...]} spans sorted by descending frequency
    test.csv                sentence, person, organization, location, misc
    retrieval_pool.parquet  sentence, person, organization, location, misc, embedding
                            (train split, deduplicated, mxbai-embed-large-v1)

Usage
-----
    python prepare_data.py                 # full pipeline (needs GPU for embeddings)
    python prepare_data.py --no-embed      # skip the retrieval pool embedding step
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent

FIELDS = ("person", "organization", "location", "misc")

# Parquet mirrors first: modern `datasets` refuses legacy dataset scripts.
CONLL_DATASET_CANDIDATES = ("eriktks/conll2003", "conll2003", "lhoestq/conll2003")

_INT_TO_STR = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

_TAG_TO_TYPE = {
    "B-PER": "person",       "I-PER": "person",
    "B-ORG": "organization", "I-ORG": "organization",
    "B-LOC": "location",     "I-LOC": "location",
    "B-MISC": "misc",        "I-MISC": "misc",
}

# v1 grammar emits no JSON escape sequences: these characters cannot appear
# inside an enum value (see pydantic_to_grammar._UNSAFE_LITERAL).
_UNSAFE_SPAN = re.compile(r'["\\\x00-\x1f]')


def extract_entities(tokens: list[str], tags: list[str]) -> dict[str, str]:
    """First span per entity type from BIO tags; missing types -> ''."""
    entities: dict[str, list[str]] = {f: [] for f in FIELDS}
    found = {f: False for f in FIELDS}
    current_type: str | None = None
    current_span: list[str] = []

    def flush():
        nonlocal current_type, current_span
        if current_type and not found[current_type]:
            entities[current_type] = current_span[:]
            found[current_type] = True
        current_type = None
        current_span = []

    for token, tag in zip(tokens, tags):
        etype = _TAG_TO_TYPE.get(tag)
        if tag.startswith("B-"):
            flush()
            current_type = etype
            current_span = [token]
        elif tag.startswith("I-") and current_type == etype:
            current_span.append(token)
        else:
            flush()
    flush()
    return {k: " ".join(v) for k, v in entities.items()}


def load_split(split: str) -> pd.DataFrame:
    from datasets import load_dataset

    raw = None
    errors = []
    for name in CONLL_DATASET_CANDIDATES:
        try:
            raw = load_dataset(name, split=split)
            print(f"Loaded {name} [{split}]: {len(raw)} rows")
            break
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{name}: {exc}")
    if raw is None:
        raise RuntimeError(f"Could not load CoNLL-2003 [{split}]. " + " | ".join(errors))

    rows = []
    for row in raw:
        tokens = row["tokens"]
        tags = [_INT_TO_STR[t] for t in row["ner_tags"]]
        gold = extract_entities(tokens, tags)
        if not any(gold.values()):          # entity-free sentences are uninformative
            continue
        rows.append({"sentence": " ".join(tokens), **gold})
    return pd.DataFrame(rows)


def drop_unsafe(df: pd.DataFrame, label: str) -> pd.DataFrame:
    unsafe = df[list(FIELDS)].apply(lambda col: col.str.contains(_UNSAFE_SPAN)).any(axis=1)
    if unsafe.any():
        print(f"{label}: dropped {int(unsafe.sum())} samples with unsafe spans "
              f'(quote/backslash/control chars, inexpressible in the v1 grammar)')
    return df[~unsafe].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip the retrieval-pool embedding step")
    parser.add_argument("--embedder", default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = SCRIPT_DIR / "data"
    out_dir.mkdir(exist_ok=True)

    splits = {s: drop_unsafe(load_split(s), s) for s in ("train", "validation", "test")}

    # ── Full closed taxonomy over ALL splits (single grammar compile) ────────
    counts: dict[str, Counter] = {f: Counter() for f in FIELDS}
    for df in splits.values():
        for f in FIELDS:
            counts[f].update(v for v in df[f] if v)
    taxonomy = {f: [v for v, _ in counts[f].most_common()] for f in FIELDS}
    with open(out_dir / "taxonomy.json", "w", encoding="utf-8") as fh:
        json.dump(taxonomy, fh, ensure_ascii=False, indent=1)
    sizes = {f: len(v) for f, v in taxonomy.items()}
    print(f"Taxonomy: {sizes} (total {sum(sizes.values())} enum values)")

    # ── Test set ─────────────────────────────────────────────────────────────
    splits["test"].to_csv(out_dir / "test.csv", index=False)
    print(f"Wrote test.csv ({len(splits['test'])} rows)")

    # ── Retrieval pool (train, deduplicated) ─────────────────────────────────
    pool = splits["train"].drop_duplicates(
        subset=["sentence", *FIELDS]).reset_index(drop=True)
    print(f"Retrieval pool: {len(pool)} unique train samples")

    if args.no_embed:
        print("Skipping embeddings (--no-embed); retrieval_pool.parquet not written.")
        return

    from sentence_transformers import SentenceTransformer
    print(f"Embedding pool with {args.embedder} ...")
    encoder = SentenceTransformer(args.embedder, device=args.device)
    emb = encoder.encode(pool["sentence"].tolist(), batch_size=64,
                         show_progress_bar=True)
    pool["embedding"] = [np.asarray(v, dtype=np.float32) for v in emb]
    pool.to_parquet(out_dir / "retrieval_pool.parquet", index=False)
    print(f"Wrote retrieval_pool.parquet ({len(pool)} rows, dim={emb.shape[1]})")


if __name__ == "__main__":
    main()
