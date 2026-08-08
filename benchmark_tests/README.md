# grammarllm — Benchmark Suite

Three tasks, three output structures, one decoding engine. Every run decodes
through the grammarllm constrained-decoding pipeline (parsing table + token
maps + token-lookahead engine) and compares **greedy vs beam3** under the
grammar mask across model scales. All decoding is deterministic
(`do_sample=False`), so `num_beams` is the only variable between the two
columns; `wos` additionally sweeps the few-shot count.

| task | output structure | grammar | vocabulary | test rows |
|---|---|---|---|---|
| [`wos`](wos/README.md) | 2-level classification (`parent\|child`) | hierarchy productions | 7 parents × 145 distinct `parent\|child` paths | 2000 |
| [`gloss_translation`](gloss_translation/README.md) | ASL gloss sequence | separator-nonterminal terminals | 16121 glosses (naturally closed) | 2000 |
| [`conll_ner`](conll_ner/README.md) | 4-field JSON object | pydantic → enum grammar | 8804 entity spans (closed over all splits) | 2756 |
| [`smiles_qed`](smiles_qed/README.md) | SMILES molecule | full OpenSMILES §2.2 grammar | open (recursive, not an enum) | 2000 |

`smiles_qed` differs from the other three in kind: its output space is open
and recursive rather than a closed label set, and the property that decides
success (ring-closure pairing, valence) is context-sensitive, so no CFG can
enforce it. It is the one task here where the grammar is not what separates a
good run from a bad one — search is. Run on Llama-3.2-1B-Instruct only.

Models: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Meta-Llama-3-8B-Instruct.
All runs use the native chat template and the token-lookahead engine (LA).
Setup details, intentional deviations and per-run commands live in each task's
README: [wos](wos/README.md) · [gloss_translation](gloss_translation/README.md)
· [conll_ner](conll_ner/README.md).

## Prompt & decoding configuration

All tasks share the native chat template, the lookahead engine and
deterministic decoding; prompt content and grammar size differ.

| | wos | gloss_translation | conll_ner |
|---|---|---|---|
| few-shot examples | 0 / 1 / 10 (static, from `few_shot{1,10}.py`) | 30 (dynamic retrieval, `retrieval.n_examples`) | 10 (dynamic retrieval, `retrieval.n_examples`) |
| extra prompt hints | all 145 `parent\|child` labels listed in the system prompt | top-50 similar glosses injected in the system prompt (`retrieval.n_gloss_hints`) | task + entity-type descriptions only (no candidate list) |
| retrieval pool | — (static examples) | 79125 train sentences | 10154 train sentences (deduplicated) |
| grammar size | 7 parents, 9–56 children each → 145 paths (tags mode, parent→child enforced) | 16121 gloss terminals | 8804 enum spans (person 3880 / org 2516 / loc 1397 / misc 1011) |
| `max_new_tokens` | 400 | 350 | 128 |
| decoding reported | greedy & beam3, `do_sample=False`, batch 5 | greedy & beam3, `do_sample=False` | greedy & beam3, `do_sample=False` |
| output separator | `\|` (pipe) between parent and child | single space between glosses | JSON skeleton (`json.dumps` default separators) |

Retrieval encoder (gloss, conll): `mixedbread-ai/mxbai-embed-large-v1`, cosine
similarity, most-similar-first ordering. Beam runs pass no extra HF parameters
(no `length_penalty` / `early_stopping` override) unless set in `config.yaml`,
so they sit at the same conditions as greedy. Full config in each task's
`config.yaml`.

## Results

### wos — Web of Science hierarchical classification

→ setup and usage: [`wos/README.md`](wos/README.md)

2000 test rows × 9 configurations × 2 decodings, run with `pipe_bench.py`
(`out_pipe_bench_greedy/` and `out_pipe_bench_beam3_nosample/`, scored with
`score_bench.py`). Micro F1 on parent (L1) and child (L2) labels.
**0 invalid parses in all 18 runs** (empty PDA stack on every row).

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

**Beam3 improves L2 in all 9 configurations** (+0.004 … +0.065) and the
`child|parent` conditional in all 9 (up to +11.9 points on 1B 10-shot).
**L1 is not uniform**: on 8B 0-shot beam3 is slightly worse (−0.007) and on
8B 1-shot the gain is within noise (+0.006). The split matches where each
decision sits: the parent is the first token, where greedy and beam see the
same distribution and there is little to recover — while the child is chosen
*inside* the branch greedy already committed to, which is exactly where
keeping alternatives alive pays. Beam can even trade a correct parent for a
better cumulative log-prob, which is what the 8B 0-shot regression looks like.

On few-shot count: 1-shot is the best setting on 1B and 3B under both
decodings, and only 8B profits from 10 examples. 0-shot collapses hardest on
3B (L1 0.188 greedy), below the 1B 0-shot run.

Note on the label space: two child names (`depression`, `schizophrenia`)
belong to both `medical` and `psychology`, so the 145 paths cover 143 distinct
child names. All 145 appear in the prompt and the grammar as distinct paths
(`medical|depression` and `psychology|depression` are separate alternatives),
but `score_bench.py` computes L2 on the bare child name, so those two pairs
score as correct under either parent.

### gloss_translation — ASLG-PC12 text → gloss

→ setup and usage: [`gloss_translation/README.md`](gloss_translation/README.md)

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

Beam3 adds +7.7 / +1.7 / +2.7 BLEU on 1B / 3B / 8B.

### smiles_qed — drug-like molecule generation (SMILES → QED)

→ setup and usage: [`smiles_qed/README.md`](smiles_qed/README.md)

2000 rows, Llama-3.2-1B-Instruct, 20 fixed few-shot examples, full OpenSMILES
grammar. `yield` = share of all rows giving a molecule that is both
RDKit-valid and drug-like (QED >= 0.5); it is the metric to compare on,
because mean QED averages over each run's own valid subset.

| run | grammar | RDKit valid | QED | uniq | **yield** |
|---|---|---|---|---|---|
| greedy + grammar | **0.990** | 0.192 | 0.800 | **0.162** | 0.180 |
| beam3 + grammar | 0.738 | **0.683** | 0.573 | 0.103 | **0.451** |
| greedy, no grammar | — | 0.186 | 0.804 | 0.155 | 0.176 |
| beam3, no grammar | — | 0.657 | 0.577 | 0.099 | 0.436 |

Gold test-set mean QED: 0.653. **Beam3 gives 2.5× the yield of greedy**
(0.451 vs 0.180) off 3.6× the chemical validity. Greedy's higher mean QED is
an artefact of averaging over the 19% of rows it got right.

Unlike the other three tasks, the grammar is *not* what decides the outcome
here: the unconstrained baseline reaches nearly the same validity (0.186 /
0.657), because the failure that dominates — unpaired ring-closure digits,
~89% of invalid outputs — is context-sensitive and outside what any CFG can
state. What the grammar guarantees is the output *shape*: both unconstrained
runs contain rows answered in prose ("I can't provide a SMILES string…"),
both constrained runs contain zero.

### conll_ner — CoNLL-2003 NER → JSON

→ setup, data prep and usage: [`conll_ner/README.md`](conll_ner/README.md)

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
   parses across all 18 wos runs, ≥99.8% gloss validity, 100% JSON+taxonomy
   validity on conll — the 1B model included. The grammar always holds; what
   varies is whether the content inside it is right.
2. **Beam3 under the mask beats greedy almost everywhere**, on all three
   tasks, with one exception: wos L1 on 8B 0-shot (−0.007) and a
   within-noise gain on 8B 1-shot (+0.006). Everywhere else the sign is
   positive across 3 tasks × 3 model sizes.
3. **The gain concentrates on the deep decision, not the first one.** On wos
   beam3 improves L2 in 9/9 configurations while L1 is flat-to-negative on
   the 8B: the parent is the first token, where both decoders see the same
   distribution, whereas the child is picked inside the branch greedy already
   committed to. conll shows the same shape as a precision effect (greedy
   locks onto a wrong enum prefix, and the mask then drags it to a wrong
   span).
4. **Search partly substitutes for scale on structured tasks**: on conll,
   1B+beam3 (0.796) beats 8B+greedy (0.767); on gloss, 1B+beam3 (65.2 BLEU)
   lands between 1B greedy (57.5) and 3B greedy (72.7); on wos, 1B+beam3
   (L1 0.554) roughly matches 3B+greedy (0.574). Where the grammar prunes
   hard (large enums, shared prefixes), keeping alternatives alive buys much
   of a model-size jump — though only conll shows a full overtake.
5. **The gain is largest on the smallest model but not monotonic in scale**
   (conll micro F1 +0.233 / +0.091 / +0.115 on 1B / 3B / 8B; gloss BLEU
   +7.7 / +1.7 / +2.7): on both tasks the 8B gain exceeds the 3B one.
