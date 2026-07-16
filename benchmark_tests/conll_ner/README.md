# CoNLL NER-to-JSON Benchmark (CoNLL-2003, sentence → JSON)

Third benchmark of the suite (after `wos` and `gloss_translation`): structured
JSON generation. Given a sentence, extract the FIRST occurrence of each
CoNLL-2003 entity type:

```json
{"person": "...", "organization": "...", "location": "...", "misc": ""}
```

Empty string = entity type absent. Two runs per model: **greedy+lookahead**
and **beam3+lookahead**, through the `pydantic_to_grammar` JSON pipeline.

## Design

- **Closed taxonomy, single grammar compile** (gloss-style): every entity span
  observed in train+validation+test becomes a `Literal` enum alternative of
  its field (~20k values total, comparable to the 16k-gloss grammar). The
  grammar is built once per run via a dynamic Pydantic model →
  `pydantic_to_productions` → `get_parsing_table_and_map_tt`.
- **Artificial closure, declared**: unlike glosses, the CoNLL value vocabulary
  is not naturally closed — the taxonomy is constructed from all splits. This
  is the standard setup in constrained-decoding NER evaluations; it measures
  the decoder under a large-enum grammar, not open-world NER.
- **Unsafe spans dropped**: the v1 JSON grammar emits no escape sequences, so
  spans containing `"`, `\` or control characters are inexpressible; samples
  containing them are removed at prep time (counted in the prep log).
- **Dynamic few-shot** (gloss-style): top-k most similar train sentences
  (mxbai-embed-large-v1, most similar first), assistant turns rendered as the
  gold JSON with `json.dumps` default separators — byte-identical to the
  grammar skeleton.
- **Native chat template**, entity-free sentences filtered out (uninformative).

Why this task for greedy vs beam3: real branch points at each value slot —
entity present vs `""`, span start choice, span extension vs closing quote —
over enum tries with heavily shared prefixes, exactly what the lookahead
engine (b89a591) has to disambiguate.

## Usage

```bash
# one-time data prep (downloads CoNLL-2003, builds taxonomy, embeds train pool)
python prepare_data.py

# greedy + lookahead
python conll_eval.py --name 1b_la_greedy

# beam3 + lookahead
python conll_eval.py --name 1b_la_beam3 --num-beams 3

# 8B
python conll_eval.py --name 8b_la_greedy --model /home/stlab/models/Meta-Llama-3-8B-Instruct

# smoke test
python conll_eval.py --name smoke --limit 5

# metrics (JSON validity, taxonomy validity, per-field accuracy, micro/macro F1)
python metrics.py output/1b_la_greedy/predictions_1b_la_greedy.csv

# full grid (greedy + beam3 on 2 GPUs)
./run_full_benchmark.sh 1b
```

Runs checkpoint every 100 rows to `output/<name>/checkpoint.csv`; resume with
`--start N --resume-csv output/<name>/checkpoint.csv`.

## Results (2756 test rows, 2026-07-16)

JSON validity and taxonomy validity: 100% on all six runs.

| model        | decoding  | Micro P | Micro R | Micro F1 | Macro F1 |
|--------------|-----------|---------|---------|----------|----------|
| Llama-3.2-1B | greedy+LA | 0.478   | 0.683   | 0.562    | 0.514    |
| Llama-3.2-1B | beam3+LA  | 0.849   | 0.749   | **0.796**| 0.717    |
| Llama-3.2-3B | greedy+LA | 0.730   | 0.727   | 0.728    | 0.704    |
| Llama-3.2-3B | beam3+LA  | 0.895   | 0.756   | **0.820**| 0.798    |
| Llama-3-8B   | greedy+LA | 0.717   | 0.824   | 0.767    | 0.736    |
| Llama-3-8B   | beam3+LA  | 0.899   | 0.866   | **0.882**| 0.858    |

Per-field F1:

| run       | person | organization | location | misc  |
|-----------|--------|--------------|----------|-------|
| 1b greedy | 0.620  | 0.589        | 0.587    | 0.261 |
| 1b beam3  | 0.921  | 0.852        | 0.780    | 0.317 |
| 3b greedy | 0.857  | 0.737        | 0.740    | 0.481 |
| 3b beam3  | 0.935  | 0.811        | 0.801    | 0.645 |
| 8b greedy | 0.935  | 0.819        | 0.798    | 0.393 |
| 8b beam3  | 0.965  | 0.906        | 0.864    | 0.697 |

Takeaways: beam3 beats greedy on every model and every field, with the gain
concentrated in precision (greedy commits to a wrong enum prefix and the mask
drags it to a wrong span; beam keeps alternatives alive). 1B+beam3 (0.796)
outperforms 8B+greedy (0.767). `misc` is the weak field everywhere.
