# grammarllm — Benchmark Suite

Three tasks, three output structures, same decoding engine. Every run decodes
through the grammarllm constrained-decoding pipeline (parsing table + token
maps + token-lookahead engine); the suite compares **greedy** vs **beam
search** under the grammar mask across model scales.

| task | output structure | grammar | vocabulary |
|---|---|---|---|
| [`wos`](wos/) | 2-level classification (`parent\|child`) | hierarchy productions | 7 domains × 134 areas |
| [`gloss_translation`](gloss_translation/) | ASL gloss sequence | WS-separated terminals | 16k glosses (naturally closed) |
| [`conll_ner`](conll_ner/) | 4-field JSON object | pydantic → enum grammar | 8804 entity spans (closed over all splits) |

Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Meta-Llama-3-8B-Instruct.
All runs use the native chat template and the token-lookahead engine (LA).
Setup details, deviations and per-run commands in each task's README.

## Results

### wos — Web of Science hierarchical classification

2000 test rows, beam3 + sampling + LA, dynamic few-shot grid. Micro F1 on
parent (L1) and child (L2) labels; **0 invalid parses in every run**.

| model | shots | L1 micro | L2 micro | L1 macro | L2 macro |
|---|---|---|---|---|---|
| 1B | 0  | 0.298 | 0.131 | 0.208 | 0.114 |
| 1B | 1  | 0.556 | 0.277 | 0.444 | 0.268 |
| 1B | 10 | 0.507 | 0.235 | 0.394 | 0.236 |
| 3B | 0  | 0.194 | 0.084 | 0.149 | 0.093 |
| 3B | 1  | 0.606 | 0.317 | 0.522 | 0.319 |
| 3B | 10 | 0.543 | 0.269 | 0.481 | 0.279 |
| 8B | 0  | 0.567 | 0.298 | 0.537 | 0.303 |
| 8B | 1  | 0.659 | 0.330 | 0.623 | 0.324 |
| 8B | 10 | 0.713 | 0.361 | 0.683 | 0.347 |

1-shot beats 10-shot on the small models; only 8B profits from more examples.

### gloss_translation — ASLG-PC12 text → gloss

2756-row full test set, dynamic top-30 few-shot, 16k-gloss grammar
(single compile). Corpus BLEU / chrF, set-F1, Validity (all predicted
glosses in vocabulary).

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

The beam3 gain is concentrated in precision: greedy commits to a wrong enum
prefix and the mask drags it to a wrong span; beam keeps alternatives alive.
**1B+beam3 (0.796) outperforms 8B+greedy (0.767).**

## Cross-task takeaways

1. **Constraint satisfaction is a solved problem at every scale**: 0 invalid
   parses (wos), ≥99.8% gloss validity, 100% JSON+taxonomy validity (conll),
   including the 1B model.
2. **Beam3 under the mask beats greedy everywhere**, on every task and scale.
   The gap shrinks as the model grows (conll micro F1: +0.23 on 1B, +0.09 on
   3B, +0.12 on 8B; gloss BLEU: +7.7 → +2.7): bigger models pick the right
   branch first more often.
3. **Search can substitute for scale on structured tasks**: on conll,
   1B+beam3 beats 8B+greedy; on gloss, 1B+beam3 (65.2 BLEU) lands between 1B
   and 3B greedy. Where the grammar prunes hard (large enums, shared
   prefixes), keeping alternatives alive matters more than parameters.
