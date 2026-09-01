"""
score_bench.py
==============
Scores a smiles_qed run: QED (Bickerton et al., 2012) plus validity and
novelty diagnostics.

Metrics
-------
Grammar validity   share of rows whose PDA stack emptied (the grammar was
                   satisfied — no truncation at max_new_tokens)
RDKit validity     share of rows RDKit can parse into a molecule. Distinct
                   from grammar validity: the OpenSMILES grammar is
                   context-free, so it cannot enforce ring-closure digit
                   pairing or valence — a string can be grammatical and
                   still not be a molecule.
Mean QED           over RDKit-valid rows only (QED is undefined otherwise).
                   The gold test set's own mean QED is printed alongside as
                   the reference point.
Uniqueness         distinct canonical SMILES / valid rows — catches a model
                   that satisfies the grammar by emitting the same molecule
                   over and over.
Novelty            share of valid predictions whose canonical form is not
                   the gold molecule for that row.
Yield              share of ALL rows that produced a molecule that is both
                   RDKit-valid and drug-like (QED >= 0.5). Mean QED alone is
                   misleading across runs, because it averages over each
                   run's own valid subset: a run that emits three compact
                   molecules and fails everywhere else scores a high mean
                   QED on a tiny, self-selected sample. Yield is over the
                   full denominator, so the two are comparable.

Usage
-----
    python score_bench.py output/beam1/predictions.csv
    python score_bench.py output/beam1/predictions.csv output/beam3/predictions.csv
"""
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED

RDLogger.DisableLog("rdApp.*")   # RDKit prints a parse error per bad SMILES


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def score(path):
    df = pd.read_csv(path)
    n = len(df)

    mols = [Chem.MolFromSmiles(str(s)) for s in df["pred_smiles"]]
    valid = [m for m in mols if m is not None]
    qeds = [QED.qed(m) for m in valid]

    canon = [Chem.MolToSmiles(m) for m in valid]
    gold_canon = [canonical(str(s)) for s in df["gold_smiles"]]
    gold_qeds = [QED.qed(Chem.MolFromSmiles(str(s))) for s in df["gold_smiles"]
                 if Chem.MolFromSmiles(str(s)) is not None]

    # novelty: valid predictions differing from that row's gold molecule
    novel = sum(
        1 for m, g in zip(mols, gold_canon)
        if m is not None and g is not None and Chem.MolToSmiles(m) != g
    )

    return {
        "run": Path(path).parent.name,
        "rows": n,
        "grammar_valid": df["grammar_satisfied"].mean() if "grammar_satisfied" in df else float("nan"),
        "rdkit_valid": len(valid) / n if n else 0.0,
        "mean_qed": sum(qeds) / len(qeds) if qeds else float("nan"),
        "gold_mean_qed": sum(gold_qeds) / len(gold_qeds) if gold_qeds else float("nan"),
        "uniqueness": len(set(canon)) / len(canon) if canon else float("nan"),
        "novelty": novel / len(valid) if valid else float("nan"),
        "yield": sum(1 for q in qeds if q >= 0.5) / n if n else 0.0,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    rows = [score(p) for p in sys.argv[1:]]

    hdr = (f"{'run':<14} {'rows':>5} {'gram':>7} {'rdkit':>7} {'QED':>7} "
           f"{'gold':>7} {'uniq':>7} {'novel':>7} {'yield':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<14} {r['rows']:>5} {r['grammar_valid']:>7.3f} "
              f"{r['rdkit_valid']:>7.3f} {r['mean_qed']:>7.3f} {r['gold_mean_qed']:>7.3f} "
              f"{r['uniqueness']:>7.3f} {r['novelty']:>7.3f} {r['yield']:>7.3f}")


if __name__ == "__main__":
    main()
