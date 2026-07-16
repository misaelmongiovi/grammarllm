# Gloss Translation Benchmark (ASLG-PC12, text → gloss)

Re-runs the paper's text-to-gloss experiment (GramDec/main_t2_cons.py, table
in `grammarllm/temp/gloss.png`) through the grammarllm library, to measure
the impact of the token-lookahead engine and beam search.

## Setup kept identical to the paper

- Test set: `GramDec/data/Gloss/test.csv` (full, copied to `data/test.csv`)
- Encoder: `mixedbread-ai/mxbai-embed-large-v1`, same stored embeddings
  (converted json → parquet, values untouched)
- Dynamic few-shot: top-30 most similar train sentences, most similar first
- System prompt: paper's `NOrules.txt` with top-50 similar glosses injected
- Greedy decoding, `max_new_tokens=350`

## Intentional deviations

| Change | Reason |
|---|---|
| No `@@` separator (plain spaces, standalone `WSEP` space nonterminal) | `@@` was a BPE workaround; the lookahead engine handles tokens spanning gloss boundaries |
| Native chat template | the custom template shatters into non-special tokens on Llama-3 and harms small models (see wos benchmark) |

`semantic_switch` (suffix form, `QUESTION-DESC`) is KEPT as the default
(`grammar.gloss_form: suffix`): it is not just a BPE workaround. With greedy
constrained decoding the prefix form forces the model to commit to `DESC-`
before the content stem; when its intended gloss (e.g. `DESC-QUESTION`) is
missing from the 16k vocabulary, the mask forces a wrong gloss
(`DESC-QUESTIONABLE`) and the output degenerates into loops (verified on
Llama-3.2-1B, test rows 1-2). Suffix form picks the stem first and degrades
gracefully. `gloss_form: prefix` is kept as an ablation.

The greedy run is the replication anchor: it should land near the paper
numbers (G-LlaMa-1B 0.47, G-LlaMa-8B 0.81 BLEU) before lookahead/beam gains
are claimed.

Note: the separator nonterminal name must be uppercase and must not equal any
tokenizer token string — `WS` collides with the `WS` token inside `LAWSUIT`
(`['LA','WS','UIT']`) and yields a spurious LL(1) conflict; `SEP` and `SPACE`
are Llama tokens too. `gloss_eval.py` asserts this at startup.

## Usage

```bash
# one-time data conversion (json → parquet)
python prepare_data.py

# greedy + lookahead (replication anchor)
python gloss_eval.py --name 1b_la_greedy

# beam search
python gloss_eval.py --name 1b_la_beam3 --num-beams 3

# boundary-strict A/B baseline (no lookahead)
python gloss_eval.py --name 1b_nola_greedy --no-lookahead

# 8B
python gloss_eval.py --name 8b_la_greedy --model /home/stlab/models/Meta-Llama-3-8B-Instruct

# smoke test
python gloss_eval.py --name smoke --limit 5

# metrics (BLEU, chrF, F1, Validity)
python metrics.py output/1b_la_greedy/predictions_1b_la_greedy.csv
```

Runs checkpoint every 100 rows to `output/<name>/checkpoint.csv`; resume with
`--start N --resume-csv output/<name>/checkpoint.csv`.

## Results (2000 test rows, `do_sample=False`, lookahead on)

Corpus BLEU / chrF, mean set-F1, Validity (share of rows whose every predicted
gloss is in the 16121-gloss vocabulary). Gold column is `gloss_adj` (suffix
form), written by `gloss_eval.py` when `grammar.gloss_form: suffix`.

| model | decoding | BLEU | chrF | F1 | Validity |
|---|---|---|---|---|---|
| 1B | greedy | 57.45 | 83.31 | 0.855 | 100.0% |
| 1B | beam3  | **65.15** | 88.01 | 0.907 | 99.8% |
| 3B | greedy | 72.71 | 89.67 | 0.911 | 99.8% |
| 3B | beam3  | **74.45** | 91.86 | 0.938 | 99.9% |
| 8B | greedy | 81.67 | 93.19 | 0.942 | 100.0% |
| 8B | beam3  | **84.39** | 94.61 | 0.956 | 100.0% |

Beam3 adds **+7.7 / +1.7 / +2.7 BLEU** on 1B / 3B / 8B — largest by far on the
smallest model, but not monotonic in scale (the 8B gain exceeds the 3B one).
Validity never drops below 99.8%: the mask holds, and the few misses are rows
where generation hit `max_new_tokens` mid-gloss.

On the replication anchor: the 8B greedy run lands on the paper (81.67 vs
0.81 BLEU), but **1B greedy is ~10 BLEU above it** (57.45 vs 0.47). The gain
is unexplained by the intentional deviations alone and has not been chased
down — treat the 1B paper comparison as unreplicated rather than beaten.
