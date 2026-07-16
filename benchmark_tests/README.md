# grammarllm — Benchmark Suite

Three tasks, three output structures, one decoding engine. Every run decodes
through the grammarllm constrained-decoding pipeline (parsing table + token
maps + token-lookahead engine). `gloss_translation` and `conll_ner` compare
**greedy vs beam3** under the grammar mask across model scales; `wos` holds
beam3 fixed and varies the number of few-shot examples instead.

| task | output structure | grammar | vocabulary | test rows |
|---|---|---|---|---|
| [`wos`](wos/) | 2-level classification (`parent\|child`) | hierarchy productions | 7 parents, 145 child productions (143 unique labels) | 2000 |
| [`gloss_translation`](gloss_translation/) | ASL gloss sequence | separator-nonterminal terminals | 16121 glosses (naturally closed) | 2000 |
| [`conll_ner`](conll_ner/) | 4-field JSON object | pydantic → enum grammar | 8804 entity spans (closed over all splits) | 2756 |

Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Meta-Llama-3-8B-Instruct.
All runs use the native chat template and the token-lookahead engine (LA).
Setup details, deviations and per-run commands in each task's README.

## Prompt & decoding configuration

All tasks share the native chat template and the lookahead engine; prompt
content, grammar size and sampling differ.

| | wos | gloss_translation | conll_ner |
|---|---|---|---|
| few-shot examples | 0 / 1 / 10 (static, from `few_shot{1,10}.py`) | 30 (dynamic retrieval, `retrieval.n_examples`) | 10 (dynamic retrieval, `retrieval.n_examples`) |
| extra prompt hints | all 145 `parent\|child` labels listed in the system prompt | top-50 similar glosses injected in the system prompt (`retrieval.n_gloss_hints`) | task + entity-type descriptions only (no candidate list) |
| retrieval pool | — (static examples) | 79125 train sentences | 10154 train sentences (deduplicated) |
| grammar size | 7 parents, 9–56 children each (tags mode, parent→child enforced) | 16121 gloss terminals | 8804 enum spans (person 3880 / org 2516 / loc 1397 / misc 1011) |
| `max_new_tokens` | 400 | 350 | 128 |
| decoding reported | beam3, `do_sample=True`, batch 5 | greedy & beam3, `do_sample=False` | greedy & beam3, `do_sample=False` |
| output separator | `\|` (pipe) between parent and child | single space between glosses | JSON skeleton (`json.dumps` default separators) |

Retrieval encoder (gloss, conll): `mixedbread-ai/mxbai-embed-large-v1`, cosine
similarity, most-similar-first ordering. Beam runs pass no extra HF parameters
(no `length_penalty` / `early_stopping` override) unless set in `config.yaml`,
so they sit at the same conditions as greedy. Full config in each task's
`config.yaml`.

## Results

### wos — Web of Science hierarchical classification

2000 test rows, beam3 + `do_sample=True` + LA, static few-shot grid, run with
`pipe_bench.py` (outputs in `out_pipe_bench/`, scored with `score_bench.py`).
Micro/macro F1 on parent (L1) and child (L2) labels; **0 invalid parses in
every run** (empty PDA stack on all 2000 rows × 9 runs).

| model | shots | L1 micro | L2 micro | L1 macro | L2 macro | child\|parent |
|---|---|---|---|---|---|---|
| 1B | 0  | 0.298 | 0.131 | 0.208 | 0.114 | 44.0% |
| 1B | 1  | **0.556** | 0.277 | 0.444 | 0.268 | 49.2% |
| 1B | 10 | 0.507 | 0.235 | 0.394 | 0.236 | 45.4% |
| 3B | 0  | 0.194 | 0.084 | 0.149 | 0.093 | 42.6% |
| 3B | 1  | **0.606** | 0.317 | 0.522 | 0.319 | 51.7% |
| 3B | 10 | 0.543 | 0.269 | 0.481 | 0.279 | 48.0% |
| 8B | 0  | 0.567 | 0.298 | 0.537 | 0.303 | 51.9% |
| 8B | 1  | 0.659 | 0.330 | 0.623 | 0.324 | 50.0% |
| 8B | 10 | **0.713** | 0.361 | 0.683 | 0.347 | 50.0% |

`child|parent` = share of correct-parent rows whose child is also correct.
It sits near 50% in every configuration, barely moving with scale or shots:
picking the right parent is what improves, while resolving the child inside
the correct branch stays a coin flip. 1-shot is the best setting on 1B and
3B — only 8B profits from 10 examples.

### gloss_translation — ASLG-PC12 text → gloss

2000 test rows (full test set), dynamic top-30 few-shot, 16121-gloss grammar
(single compile). Corpus BLEU / chrF, mean set-F1, Validity (share of rows
whose every predicted gloss is in the vocabulary).

| model | decoding | BLEU | chrF | F1 | Validity |
|---|---|---|---|---|---|
| 1B | greedy+LA | 57.45 | 83.31 | 0.855 | 100.0% |
| 1B | beam3+LA  | **65.15** | 88.01 | 0.907 | 99.8% |
| 3B | greedy+LA | 72.71 | 89.67 | 0.911 | 99.8% |
| 3B | beam3+LA  | **74.45** | 91.86 | 0.938 | 99.9% |
| 8B | greedy+LA | 81.67 | 93.19 | 0.942 | 100.0% |
| 8B | beam3+LA  | **84.39** | 94.61 | 0.956 | 100.0% |

Greedy runs are the paper-replication anchors (see task README); beam3 adds
+7.7 / +1.7 / +2.7 BLEU on 1B / 3B / 8B.

### conll_ner — CoNLL-2003 NER → JSON

2756 test rows, first-span extraction into
`{"person", "organization", "location", "misc"}`, 8804-value enum grammar via
`pydantic_to_grammar` (single compile), top-10 few-shot. **JSON validity and
taxonomy validity: 100% on all six runs.**

| model | decoding | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---|---|---|---|---|
| 1B | greedy+LA | 0.478 | 0.683 | 0.562 | 0.514 |
| 1B | beam3+LA  | 0.849 | 0.749 | **0.796** | 0.717 |
| 3B | greedy+LA | 0.730 | 0.727 | 0.728 | 0.704 |
| 3B | beam3+LA  | 0.895 | 0.756 | **0.820** | 0.798 |
| 8B | greedy+LA | 0.717 | 0.824 | 0.767 | 0.736 |
| 8B | beam3+LA  | 0.899 | 0.866 | **0.882** | 0.858 |

Per-field F1:

| run | person | organization | location | misc |
|---|---|---|---|---|
| 1B greedy | 0.620 | 0.589 | 0.587 | 0.261 |
| 1B beam3  | 0.921 | 0.852 | 0.780 | 0.317 |
| 3B greedy | 0.857 | 0.737 | 0.740 | 0.481 |
| 3B beam3  | 0.935 | 0.811 | 0.801 | 0.645 |
| 8B greedy | 0.935 | 0.819 | 0.798 | 0.393 |
| 8B beam3  | 0.965 | 0.906 | 0.864 | 0.697 |

The beam3 gain is concentrated in precision: greedy commits to a wrong enum
prefix and the mask drags it to a wrong span; beam keeps alternatives alive.
**1B+beam3 (0.796) outperforms 8B+greedy (0.767).** `misc` is the weak field
for every model and both decodings (best 0.697), as expected from a catch-all
class the enum cannot disambiguate.

## Cross-task takeaways

1. **Constraint satisfaction is a solved problem at every scale**: 0 invalid
   parses (wos), ≥99.8% gloss validity, 100% JSON+taxonomy validity (conll),
   including the 1B model.
2. **Beam3 under the mask beats greedy on every model, on both tasks that
   compare them** (gloss, conll). The gain is by far the largest on the 1B
   model (conll micro F1 +0.233, gloss BLEU +7.7) and much smaller from 3B up
   (conll +0.091 on 3B, +0.115 on 8B; gloss BLEU +1.7 and +2.7) — but it does
   not decay monotonically with scale: on both tasks the 8B gain exceeds the
   3B one.
3. **Search partly substitutes for scale on structured tasks**: on conll,
   1B+beam3 (0.796) beats 8B+greedy (0.767); on gloss, 1B+beam3 (65.2 BLEU)
   lands between 1B greedy (57.5) and 3B greedy (72.7). Where the grammar
   prunes hard (large enums, shared prefixes), keeping alternatives alive
   buys more than the next model size — but only conll shows a full
   scale-jump overtake.
