"""
prepare_data.py
================
Builds the fixed test set (2000 rows) and the fixed 20-shot few-shot pool
for smiles_qed, deterministically, from one shard of

    jablonkagroup/pubchem-smiles-molecular-formula  (train, 82.3M rows, 11 parquet shards)

Deviation from the paper setup (documented, not hidden): the paper samples
from GDB-17 (Ruddigkeit et al., 2012); here we use the PubChem SMILES
dataset specified for this run instead. We only need 2020 rows total, so
rather than shuffling the full 82M-row streaming dataset (too slow — killed
after 120s at buffer_size=200000), we download ONE parquet shard
(train-00003-of-00011, ~7.5M rows) and sample locally. The shard choice is
fixed and the sample is a seeded shuffle, so this is reproducible; it is not
a uniform sample over all 11 shards, which is the deviation.

Output: data/test_2000.jsonl (test set, not used as generation targets —
only to fix N=2000 and hold these SMILES out of the few-shot pool) and
data/few_shot_20.jsonl (the fixed few-shot gold set, disjoint from test).
"""
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SEED = 42
N_TEST = 2000
N_FEWSHOT = 20
SHARD = "data/train-00003-of-00011.parquet"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    path = hf_hub_download(
        "jablonkagroup/pubchem-smiles-molecular-formula", SHARD, repo_type="dataset"
    )
    table = pq.read_table(path, columns=["smiles", "molecular_formula"])
    rows = table.to_pylist()

    # Dedupe on smiles (keeping the paired molecular_formula), then a seeded
    # shuffle over the deduped list — deterministic, independent of input
    # row order.
    by_smiles = {r["smiles"]: r["molecular_formula"] for r in rows}
    uniq = sorted(by_smiles)
    import random
    rng = random.Random(SEED)
    rng.shuffle(uniq)

    need = N_FEWSHOT + N_TEST
    assert len(uniq) >= need, f"shard has only {len(uniq)} unique SMILES, need {need}"

    few_shot = uniq[:N_FEWSHOT]
    test = uniq[N_FEWSHOT:need]
    assert set(few_shot).isdisjoint(test)

    with open(os.path.join(OUT_DIR, "few_shot_20.jsonl"), "w") as f:
        for s in few_shot:
            f.write(json.dumps({"smiles": s, "molecular_formula": by_smiles[s]}) + "\n")

    with open(os.path.join(OUT_DIR, "test_2000.jsonl"), "w") as f:
        for s in test:
            f.write(json.dumps({"smiles": s, "molecular_formula": by_smiles[s]}) + "\n")

    print(f"few-shot gold : {len(few_shot)} rows -> data/few_shot_20.jsonl")
    print(f"test set      : {len(test)} rows -> data/test_2000.jsonl")


if __name__ == "__main__":
    main()
