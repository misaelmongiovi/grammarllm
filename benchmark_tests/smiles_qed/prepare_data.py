"""
prepare_data.py
================
Builds the fixed test set (2000 rows) and the fixed 20-shot few-shot pool
for smiles_qed, deterministically, from either source dataset.

    --dataset pubchem   jablonkagroup/pubchem-smiles-molecular-formula
                        82.3M rows over 11 parquet shards, with a
                        molecular_formula column.

    --dataset gdb17     Pixelatory/GDB-17 — the paper's dataset
                        (Ruddigkeit et al., 2012), 50M canonical SMILES
                        re-published from the original GDB-17-Set at
                        gdb.unibe.ch. SMILES only, so the molecular formula
                        is computed with RDKit.

Both write data/<dataset>/{few_shot_20,test_2000}.jsonl with the same schema
({"smiles", "molecular_formula"}), so pipe_bench.py treats them alike.

Sampling is a seeded shuffle over the deduplicated SMILES, hence
reproducible. For pubchem only ONE shard is read (train-00003-of-00011,
~7.5M rows) rather than shuffling the full 82M-row streaming dataset, which
was too slow to be worth it for 2020 rows — a documented deviation: the
sample is uniform within that shard, not across all 11.

Usage
-----
    python prepare_data.py --dataset pubchem
    python prepare_data.py --dataset gdb17 --gdb17-csv /tmp/gdb17/GDB17.csv
"""
import argparse
import json
import os
import random

SEED = 42
N_TEST = 2000
N_FEWSHOT = 20
PUBCHEM_SHARD = "data/train-00003-of-00011.parquet"
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def load_pubchem():
    """{smiles: molecular_formula} from one PubChem parquet shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "jablonkagroup/pubchem-smiles-molecular-formula", PUBCHEM_SHARD,
        repo_type="dataset",
    )
    table = pq.read_table(path, columns=["smiles", "molecular_formula"])
    return {r["smiles"]: r["molecular_formula"] for r in table.to_pylist()}


def load_gdb17(csv_path, n_scan=2_000_000):
    """
    SMILES from the GDB-17 csv, with formulas computed by RDKit.

    The file is 50M rows / 1.5 GB; we only need 2020. Reading the first
    n_scan rows and shuffling those is enough and keeps this fast — but note
    GDB-17 is ordered, so this is a sample of that prefix, not of the whole
    set. Same class of deviation as the pubchem single-shard read, recorded
    here rather than buried.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")

    smiles = []
    with open(csv_path) as f:
        header = next(f)
        assert header.strip() == "smiles", f"unexpected header: {header!r}"
        for i, line in enumerate(f):
            if i >= n_scan:
                break
            smiles.append(line.strip())

    rng = random.Random(SEED)
    rng.shuffle(smiles)

    out = {}
    for s in smiles:
        if len(out) >= (N_TEST + N_FEWSHOT) * 2:   # headroom for dedupe
            break
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[s] = CalcMolFormula(mol)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pubchem", "gdb17"], default="pubchem")
    ap.add_argument("--gdb17-csv", default="/tmp/gdb17/GDB17.csv",
                    help="Path to the extracted GDB17.csv")
    args = ap.parse_args()

    out_dir = os.path.join(DATA_ROOT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    by_smiles = (load_pubchem() if args.dataset == "pubchem"
                 else load_gdb17(args.gdb17_csv))

    uniq = sorted(by_smiles)
    rng = random.Random(SEED)
    rng.shuffle(uniq)

    need = N_FEWSHOT + N_TEST
    assert len(uniq) >= need, f"only {len(uniq)} unique SMILES, need {need}"

    few_shot = uniq[:N_FEWSHOT]
    test = uniq[N_FEWSHOT:need]
    assert set(few_shot).isdisjoint(test)

    for name, rows in (("few_shot_20", few_shot), ("test_2000", test)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for s in rows:
                f.write(json.dumps({"smiles": s,
                                    "molecular_formula": by_smiles[s]}) + "\n")

    print(f"[{args.dataset}] few-shot : {len(few_shot)} -> {out_dir}/few_shot_20.jsonl")
    print(f"[{args.dataset}] test     : {len(test)} -> {out_dir}/test_2000.jsonl")


if __name__ == "__main__":
    main()
