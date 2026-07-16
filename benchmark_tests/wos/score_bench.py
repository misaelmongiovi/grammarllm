"""
score_bench.py — micro/macro-F1 su L1 (parent) e L2 (child) per i CSV di pipe_bench.py.

La predizione e' un dict con 'text' = "parent|child".  Riporta anche i parse invalidi
(pda_stack non vuoto: la grammatica non e' stata soddisfatta) — devono essere 0.

usage: score_bench.py out_pipe_bench/*.csv
"""
import ast
import sys

import pandas as pd
from sklearn.metrics import f1_score

f1 = lambda y, yh, avg: f1_score(y, yh, average=avg, zero_division=0)


def score(path):
    df = pd.read_csv(path)
    df = df[df.pred.notna()]
    preds = [ast.literal_eval(x) for x in df.pred]
    texts = [p['text'].strip() for p in preds]
    p1 = [t.split('|')[0] for t in texts]
    p2 = [t.partition('|')[2] for t in texts]

    invalid = sum(1 for p in preds if p['pda_stack'] != [])
    ok = sum(a == b for a, b in zip(df.Domain, p1))
    both = sum(a == b and c == d for a, b, c, d in zip(df.Domain, p1, df.area, p2))
    return {
        'n': len(df),
        'L1': f1(df.Domain, p1, 'micro'), 'L1_macro': f1(df.Domain, p1, 'macro'),
        'L2': f1(df.area, p2, 'micro'),   'L2_macro': f1(df.area, p2, 'macro'),
        'invalid': invalid,
        'child_given_parent': both / ok if ok else 0.0,
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    print(f"{'run':<44}{'n':>6}{'L1':>8}{'L2':>8}{'L1mac':>8}{'L2mac':>8}"
          f"{'inval':>7}{'child|par':>10}")
    print('-' * 99)
    for p in sorted(paths):
        try:
            r = score(p)
        except Exception as e:
            print(f"{p.split('/')[-1]:<44}  errore: {e}")
            continue
        print(f"{p.split('/')[-1].replace('.csv',''):<44}{r['n']:>6}{r['L1']:>8.4f}"
              f"{r['L2']:>8.4f}{r['L1_macro']:>8.4f}{r['L2_macro']:>8.4f}"
              f"{r['invalid']:>7}{r['child_given_parent']:>9.1%}")


if __name__ == '__main__':
    main()
