# WoS Benchmark (Web of Science, abstract → `parent|child`)

Hierarchical classification as constrained decoding. Given a paper abstract,
emit a two-level label path:

```
medical|medicare
```

First benchmark of the suite (see also [`gloss_translation`](../gloss_translation/)
and [`conll_ner`](../conll_ner/)). The grammar enforces the hierarchy itself:
the child alternatives offered at step 2 are only those of the parent chosen at
step 1, so an impossible path like `cs|cancer` is unreachable by construction.

## Design

- **Grammar (`tags` mode)**: productions come straight from `config.yaml` —
  `S*` picks one of 7 parents, each parent non-terminal (`A`..`G`) lists its
  own children. 7 parents, 9–56 children each, **145 distinct paths**.
- **Separator**: a single `|` between parent and child. It is its own
  terminal, so the lookahead engine resolves tokens that span the boundary.
- **Prompt**: `pipe_bench.py` builds its own system prompt listing all 145
  `parent|child` labels, plus 0 / 1 / 10 static few-shot examples from
  `few_shot{1,10}.py` (their answers are rewritten to pipe form at load time).
- **Decoding**: fp16, `max_new_tokens=400`, batch 5, lookahead on. Both
  decodings are deterministic (`do_sample=False`), so `num_beams` is the only
  variable between the greedy and beam3 columns.
- **Test set**: `data/WebOfScience/test_data.csv` at the repo root (2000 rows;
  gold parent in `Domain`, gold child in `area`). Not in the repo — `/data` is
  gitignored.

Note on the label space: two child names (`depression`, `schizophrenia`) exist
under both `medical` and `psychology`. All 145 paths are distinct and appear
separately in the prompt and the grammar, but `score_bench.py` scores L2 on the
bare child name, so those two score as correct under either parent.

## Usage

```bash
# greedy + lookahead, all models × {0,1,10} shots, 2000 rows
BEAMS=1 SAMPLE=off OUT="$PWD/out_pipe_bench_greedy" ./run_bench.sh

# beam3 + lookahead
BEAMS=3 SAMPLE=off OUT="$PWD/out_pipe_bench_beam3_nosample" ./run_bench.sh

# subset / smoke
ROWS=200 MODELS=Llama-3.2-1B-Instruct SHOTS=1 ./run_bench.sh

# single run
python pipe_bench.py --model /home/stlab/models/Llama-3.2-1B-Instruct \
    --nshot 1 --out preds.csv --beams 1 --no-sample

# scoring (micro/macro F1 on L1 and L2, invalid parses, child|parent)
python score_bench.py out_pipe_bench_greedy/*.csv
```

`run_bench.sh` keeps one job per GPU and advances the queue as GPUs free up.
Env knobs: `BEAMS`, `SAMPLE=on|off`, `LOOKAHEAD=on|off`, `ROWS`, `MODELS`,
`SHOTS`, `GPUS`, `OUT`, `FORCE=1` (re-run instead of skipping existing output).

`SAMPLE=off` matters: the script otherwise passes `--sample`, so `BEAMS=1`
alone would give pure multinomial sampling rather than greedy.

## Results (2000 rows, `sample=False`, lookahead on)

**0 invalid parses in all 18 runs** — the PDA stack is empty on every row, so
every output is a well-formed path in the hierarchy.

| model | shots | L1 greedy | L1 beam3 | Δ L1 | L2 greedy | L2 beam3 | Δ L2 |
|---|---|---|---|---|---|---|---|
| 1B | 0  | 0.267 | 0.298 | +0.032 | 0.109 | 0.128 | +0.019 |
| 1B | 1  | 0.516 | 0.554 | +0.039 | 0.216 | 0.275 | +0.060 |
| 1B | 10 | 0.497 | 0.507 | +0.010 | 0.170 | 0.234 | +0.065 |
| 3B | 0  | 0.188 | 0.194 | +0.006 | 0.081 | 0.084 | +0.004 |
| 3B | 1  | 0.574 | 0.606 | +0.033 | 0.277 | 0.317 | +0.040 |
| 3B | 10 | 0.527 | 0.543 | +0.016 | 0.238 | 0.269 | +0.031 |
| 8B | 0  | 0.574 | 0.567 | **−0.007** | 0.277 | 0.298 | +0.021 |
| 8B | 1  | 0.653 | 0.659 | +0.006 | 0.303 | 0.330 | +0.027 |
| 8B | 10 | 0.681 | **0.713** | +0.032 | 0.310 | **0.361** | +0.051 |

Macro F1 and the `child|parent` conditional (share of correct-parent rows whose
child is also correct):

| model | shots | L1 macro g/b3 | L2 macro g/b3 | child\|parent g/b3 |
|---|---|---|---|---|
| 1B | 0  | 0.157 / 0.209 | 0.073 / 0.111 | 40.7% / 43.0% |
| 1B | 1  | 0.357 / 0.442 | 0.194 / 0.265 | 41.3% / 49.0% |
| 1B | 10 | 0.376 / 0.396 | 0.159 / 0.235 | 33.5% / 45.4% |
| 3B | 0  | 0.137 / 0.149 | 0.077 / 0.093 | 42.1% / 42.6% |
| 3B | 1  | 0.463 / 0.522 | 0.258 / 0.319 | 47.5% / 51.7% |
| 3B | 10 | 0.445 / 0.481 | 0.237 / 0.279 | 43.7% / 48.0% |
| 8B | 0  | 0.518 / 0.537 | 0.260 / 0.303 | 47.6% / 51.9% |
| 8B | 1  | 0.602 / 0.623 | 0.285 / 0.324 | 45.9% / 50.0% |
| 8B | 10 | 0.642 / 0.683 | 0.294 / 0.347 | 44.9% / 50.0% |

**Beam3 improves L2 in 9/9 configurations and `child|parent` in 9/9** (up to
+11.9 points on 1B 10-shot), while **L1 is flat-to-negative on the 8B**
(−0.007 at 0-shot, +0.006 at 1-shot). The parent is the first token, where
both decoders see the same distribution and there is little to recover; the
child is chosen *inside* the branch greedy already committed to, which is
where keeping alternatives alive pays. Beam can even trade a correct parent
for a better cumulative log-prob — that is what the 8B 0-shot regression
looks like.

On few-shot count: **1-shot is the best setting on 1B and 3B** under both
decodings, and only the 8B profits from 10 examples. 0-shot collapses hardest
on the 3B (L1 0.188 greedy), below the 1B 0-shot run.

## Files

| file | role |
|---|---|
| `pipe_bench.py` | the benchmark runner — builds the grammar from `config.yaml`, its own system prompt, and writes `pred` as a dict with `text` and `pda_stack` |
| `score_bench.py` | scoring; `pda_stack != []` counts as an invalid parse |
| `run_bench.sh` | GPU queue over models × shots |
| `few_shot{1,10}.py` | static few-shot examples |
| `config.yaml` | grammar productions + the `wos_eval.py` prompt |
| `wos_eval.py` | alternative single-run entry point; supports `grammar.mode: json` (a flat-child enum via `wos_schema.py`) besides `tags`. **Not** used for the results above |
| `old/` | superseded experiments (fp32, lookahead-off, beam-sample, JSON-mode ablations), untracked |
