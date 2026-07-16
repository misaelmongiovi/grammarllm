"""
metrics.py
==========
Paper metrics for the gloss-translation benchmark: BLEU, chrF, set-based F1,
Validity. Same formulas as GramDec/scripts/task2/metrics_temp.py, computed on
the raw prefix gloss form (gold column: 'gloss').

Usage
-----
    python metrics.py output/1b_la_greedy/predictions_1b_la_greedy.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sacrebleu import metrics

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from gloss_eval import semantic_switch_gloss  # noqa: E402


def set_f1(gold_tokens: list[str], pred_tokens: list[str]) -> float:
    gold, pred = set(gold_tokens), set(pred_tokens)
    if not pred or not gold:
        return 0.0
    p = len(pred & gold) / len(pred)
    r = len(pred & gold) / len(gold)
    if p == 0 or r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_csv", help="Predictions csv (columns: gloss, prediction)")
    parser.add_argument("--vocab", default=str(SCRIPT_DIR / "data/gloss_vocab.parquet"),
                        help="Gloss vocabulary parquet for the Validity metric")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    # gloss_adj (suffix/paper form) is written by gloss_eval when
    # grammar.gloss_form == suffix; fall back to the raw prefix gold.
    gold_col = "gloss_adj" if "gloss_adj" in df.columns else "gloss"
    df[gold_col] = df[gold_col].fillna("").astype(str)
    df["prediction"] = df["prediction"].fillna("").astype(str)
    print(f"Gold column: {gold_col}")

    golds = [g.replace("\n", " ").strip() for g in df[gold_col]]
    preds = [p.replace("\n", " ").strip() for p in df["prediction"]]

    bleu = metrics.BLEU(effective_order=True)
    chrf = metrics.CHRF()

    vocab_words = set(pd.read_parquet(args.vocab, columns=["word"])["word"].astype(str).str.strip())
    if gold_col == "gloss_adj":
        vocab_words = {semantic_switch_gloss(w) for w in vocab_words}

    sent_bleu, sent_chrf, sent_f1, valid = [], [], [], []
    detail_path = Path(args.pred_csv).parent / "scores_per_line.txt"
    with open(detail_path, "w", encoding="utf-8") as out:
        for idx, (gold, pred) in enumerate(zip(golds, preds)):
            if not gold or not pred:
                out.write(f"Linea {idx}: riga vuota\n\n")
                sent_bleu.append(0.0)
                sent_chrf.append(0.0)
                sent_f1.append(0.0)
                valid.append(False)
                continue
            b = bleu.sentence_score(pred, [gold]).score
            c = chrf.sentence_score(pred, [gold]).score
            f = set_f1(gold.split(), pred.split())
            v = all(tok in vocab_words for tok in pred.split())
            sent_bleu.append(b)
            sent_chrf.append(c)
            sent_f1.append(f)
            valid.append(v)
            out.write(f"Linea {idx}:\nGold: {gold}\nPred: {pred}\n"
                      f"BLEU: {b:.2f}\nchrF: {c:.2f}\nF1: {f:.4f}\nValid: {v}\n\n")

        corpus_bleu = bleu.corpus_score(preds, [golds]).score
        corpus_chrf = chrf.corpus_score(preds, [golds]).score
        out.write("Risultati Corpus:\n")
        out.write(f"Corpus BLEU: {corpus_bleu:.2f}\n")
        out.write(f"Corpus chrF: {corpus_chrf:.2f}\n")
        out.write(f"Mean sentence BLEU: {sum(sent_bleu)/len(sent_bleu):.2f}\n")
        out.write(f"Mean sentence chrF: {sum(sent_chrf)/len(sent_chrf):.2f}\n")
        out.write(f"Mean F1: {sum(sent_f1)/len(sent_f1):.4f}\n")
        out.write(f"Validity: {100*sum(valid)/len(valid):.1f}%\n")

    print(f"N={len(golds)}")
    print(f"Corpus BLEU: {corpus_bleu:.2f}")
    print(f"Corpus chrF: {corpus_chrf:.2f}")
    print(f"Mean sentence BLEU: {sum(sent_bleu)/len(sent_bleu):.2f}")
    print(f"Mean sentence chrF: {sum(sent_chrf)/len(sent_chrf):.2f}")
    print(f"Mean F1: {sum(sent_f1)/len(sent_f1):.4f}")
    print(f"Validity: {100*sum(valid)/len(valid):.1f}%")
    print(f"Per-line details: {detail_path}")


if __name__ == "__main__":
    main()
