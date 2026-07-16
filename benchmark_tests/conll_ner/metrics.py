"""
metrics.py
==========
Metrics for the CoNLL NER-to-JSON benchmark.

  * JSON validity     — prediction parses as JSON with exactly the 4 keys
  * Taxonomy validity — every non-empty predicted value is in the taxonomy
  * Per-field accuracy — exact match, empty string included
  * Entity micro/macro P/R/F1 — over non-empty gold/pred values:
        tp: gold != ''  and pred == gold
        fp: pred != ''  and pred != gold
        fn: gold != ''  and pred != gold

Usage
-----
    python metrics.py output/1b_la_greedy/predictions_1b_la_greedy.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
FIELDS = ("person", "organization", "location", "misc")


def parse_prediction(text: str) -> dict[str, str] | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(FIELDS):
        return None
    if not all(isinstance(v, str) for v in obj.values()):
        return None
    return obj


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_csv",
                        help="Predictions csv (columns: sentence, <fields>, prediction)")
    parser.add_argument("--taxonomy", default=str(SCRIPT_DIR / "data/taxonomy.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv, keep_default_na=False)
    with open(args.taxonomy, encoding="utf-8") as fh:
        taxonomy = {f: set(v) for f, v in json.load(fh).items()}

    n = len(df)
    json_valid = 0
    tax_valid = 0
    field_correct = {f: 0 for f in FIELDS}
    counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}

    detail_path = Path(args.pred_csv).parent / "scores_per_line.txt"
    with open(detail_path, "w", encoding="utf-8") as out:
        for idx, row in df.iterrows():
            gold = {f: str(row[f]) for f in FIELDS}
            pred = parse_prediction(str(row["prediction"]))

            if pred is None:
                out.write(f"Linea {idx}: JSON invalido\nPred: {row['prediction']}\n\n")
                for f in FIELDS:
                    if gold[f]:
                        counts[f]["fn"] += 1
                continue

            json_valid += 1
            in_tax = all(not v or v in taxonomy[f] for f, v in pred.items())
            tax_valid += in_tax

            line_ok = True
            for f in FIELDS:
                g, p = gold[f], pred[f]
                if p == g:
                    field_correct[f] += 1
                else:
                    line_ok = False
                if g and p == g:
                    counts[f]["tp"] += 1
                else:
                    if p:
                        counts[f]["fp"] += 1
                    if g:
                        counts[f]["fn"] += 1

            if not line_ok:
                out.write(f"Linea {idx}:\nSent: {row['sentence']}\n"
                          f"Gold: {json.dumps(gold, ensure_ascii=False)}\n"
                          f"Pred: {json.dumps(pred, ensure_ascii=False)}\n"
                          f"InTaxonomy: {in_tax}\n\n")

        tp = sum(c["tp"] for c in counts.values())
        fp = sum(c["fp"] for c in counts.values())
        fn = sum(c["fn"] for c in counts.values())
        micro_p, micro_r, micro_f = prf(tp, fp, fn)
        per_field_f = {f: prf(**counts[f])[2] for f in FIELDS}
        macro_f = sum(per_field_f.values()) / len(FIELDS)

        lines = [f"N={n}",
                 f"JSON validity: {100 * json_valid / n:.1f}%",
                 f"Taxonomy validity: {100 * tax_valid / n:.1f}%"]
        for f in FIELDS:
            lines.append(f"Accuracy[{f}]: {100 * field_correct[f] / n:.1f}%   "
                         f"F1[{f}]: {per_field_f[f]:.4f}")
        lines += [f"Micro P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f:.4f}",
                  f"Macro F1: {macro_f:.4f}"]

        out.write("Risultati Corpus:\n" + "\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"Per-line details: {detail_path}")


if __name__ == "__main__":
    main()
