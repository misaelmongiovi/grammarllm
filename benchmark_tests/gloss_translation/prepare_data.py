"""
prepare_data.py
===============
One-time conversion of the GramDec gloss-translation data into the parquet
files used by gloss_eval.py.

Sources (paper runs, GramDec/main_t2_cons.py):
    GramDec/data/Gloss/embeddings.json        sentence, decomposition, embedding_sentence
    GramDec/data/Gloss/gloss_embeddings.json  word, embedding_gloss
    GramDec/data/Gloss/test.csv               gloss, text

Outputs (benchmark_tests/gloss_translation/data/):
    retrieval_pool.parquet  sentence, decomposition, embedding
    gloss_vocab.parquet     word, embedding
    test.csv                copied as-is

Embeddings are kept exactly as stored (mxbai-embed-large-v1, unnormalized);
normalization happens at query time, as in the paper pipeline.
Glosses keep the original prefix form (DESC-, X-): the semantic_switch
suffix rewrite of the paper was a workaround for the old FSA and is no
longer needed with grammarllm's shared-prefix handling.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
DEFAULT_SRC = Path("/home/gtuccio/GramDec/data/Gloss")


def convert_pool(src: Path, dst: Path) -> None:
    print(f"Loading {src} (may take a few minutes)...")
    with open(src) as f:
        df = pd.DataFrame(json.load(f))
    df = df.rename(columns={"embedding_sentence": "embedding"})
    df["sentence"] = df["sentence"].str.replace("\n", "", regex=False)
    df["decomposition"] = df["decomposition"].str.replace("\n", "", regex=False)
    df["embedding"] = df["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32))
    dim = len(df["embedding"].iloc[0])
    assert dim == 1024, f"Expected mxbai-embed-large-v1 dim 1024, got {dim}"
    df[["sentence", "decomposition", "embedding"]].to_parquet(dst, index=False)
    print(f"Wrote {dst} ({len(df)} rows, dim={dim})")


def convert_vocab(src: Path, dst: Path) -> None:
    print(f"Loading {src}...")
    with open(src) as f:
        df = pd.DataFrame(json.load(f))
    df = df.rename(columns={"embedding_gloss": "embedding"})
    df["embedding"] = df["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32))
    dim = len(df["embedding"].iloc[0])
    assert dim == 1024, f"Expected mxbai-embed-large-v1 dim 1024, got {dim}"
    df[["word", "embedding"]].to_parquet(dst, index=False)
    print(f"Wrote {dst} ({len(df)} rows, dim={dim})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help="GramDec data/Gloss directory")
    args = parser.parse_args()

    out_dir = SCRIPT_DIR / "data"
    out_dir.mkdir(exist_ok=True)

    shutil.copy(args.src / "test.csv", out_dir / "test.csv")
    print(f"Copied test.csv ({out_dir / 'test.csv'})")

    convert_vocab(args.src / "gloss_embeddings.json", out_dir / "gloss_vocab.parquet")
    convert_pool(args.src / "embeddings.json", out_dir / "retrieval_pool.parquet")


if __name__ == "__main__":
    main()
