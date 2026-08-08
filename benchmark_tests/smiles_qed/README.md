# smiles_qed — drug-like molecule generation (SMILES → QED)

Molecule generation through grammarllm constrained decoding, scored by QED
(Bickerton et al., 2012), following the reference setup: few-shot SMILES
prompting, QED as the fitness function, a Llama base model.

## Setup

| | value |
|---|---|
| dataset | [`jablonkagroup/pubchem-smiles-molecular-formula`](https://huggingface.co/datasets/jablonkagroup/pubchem-smiles-molecular-formula) |
| test set | 2000 rows, fixed (`data/test_2000.jsonl`) |
| few-shot | 20 fixed examples, disjoint from test (`data/few_shot_20.jsonl`) |
| model | Llama-3.2-1B-Instruct |
| grammar | full OpenSMILES §2.2, from `examples/smiles.py` |
| decoding | `do_sample=False`, greedy (`beam1`) and `beam3`, batch 5, `max_new_tokens=100` |
| engine | token-lookahead on, native chat template |

Reproduce:

```bash
python prepare_data.py                              # once
CUDA_VISIBLE_DEVICES=2 python pipe_bench.py --name beam1
CUDA_VISIBLE_DEVICES=2 python pipe_bench.py --name beam3 --num-beams 3
CUDA_VISIBLE_DEVICES=2 python unconstrained_baseline.py --name nogram_beam1
python score_bench.py output/*/predictions.csv
```

## Deviations from the reference setup

All deliberate, none silent:

1. **Dataset**: PubChem SMILES instead of GDB-17 (as specified for this run).
   Only one parquet shard (`train-00003-of-00011`, ~7.5M rows) is downloaded
   and sampled with a fixed seed — the full 82M-row streaming shuffle was too
   slow to be worth it for 2020 rows. Reproducible, but not a uniform sample
   over all 11 shards.
2. **Model**: Llama-3.2-1B-Instruct instead of Llama 3.1 8B.
3. **Few-shot**: 20 examples chosen once and held fixed, rather than resampled
   per prompt.
4. **Conditioning**: the reference task generates unconditionally. This repo's
   benchmark convention is deterministic decoding (`do_sample=False`), and an
   identical prompt under greedy or beam search returns the identical molecule
   every time — 2000 copies of one argmax. Each row therefore conditions on
   its own `molecular_formula`, the dataset's second column, which is what
   makes rows differ and makes greedy-vs-beam3 a meaningful comparison.

## Metrics

`score_bench.py` reports, per run:

- **gram** — share of rows whose PDA stack emptied (grammar satisfied, no
  truncation at `max_new_tokens`)
- **rdkit** — share RDKit can parse into a molecule
- **QED** — mean QED over that run's *own* valid rows
- **gold** — mean QED of the gold test molecules (0.653), the reference point
- **uniq** — distinct canonical SMILES / valid rows
- **novel** — valid predictions differing from that row's gold molecule
- **yield** — share of **all** rows giving a molecule that is both valid and
  drug-like (QED ≥ 0.5)

**Read `yield`, not `QED`, when comparing runs.** Mean QED averages over each
run's own valid subset, so a run that succeeds on a handful of compact
molecules and fails everywhere else posts a high mean QED on a tiny,
self-selected sample. `yield` uses the full denominator.

## Results

2000 rows. `nogram_*` is the same model, prompts and decoding with the grammar
mask removed — the A/B for what constrained decoding is worth here.

| run | gram | rdkit | QED | uniq | novel | **yield** |
|---|---|---|---|---|---|---|
| `beam1` (greedy + grammar) | **0.990** | 0.192 | 0.800 | **0.162** | 1.000 | 0.180 |
| `beam3` (beam3 + grammar) | 0.738 | **0.683** | 0.573 | 0.103 | 1.000 | **0.451** |
| `nogram_beam1` | — | 0.186 | 0.804 | 0.155 | 1.000 | 0.176 |
| `nogram_beam3` | — | 0.657 | 0.577 | 0.099 | 1.000 | 0.436 |

Gold test-set mean QED: **0.653**. Every run is 100% novel (no prediction
reproduces its row's gold molecule).

**Beam3 is the headline: yield 0.451 vs 0.180, a 2.5× gain over greedy**, on
the back of a 3.6× gain in chemical validity (0.683 vs 0.192).

Greedy's higher mean QED (0.800 vs 0.573) is an artefact of its own failure
rate, and is the reason `yield` exists: that average is taken over the 19% of
rows greedy got right, which are short, compact scaffolds. On the rows valid
under *both* decodings, greedy is genuinely the better chemist (QED 0.814 vs
0.648) — it just produces a valid molecule far less often.

Beam3's grammar-satisfied rate (0.738) is **not** a constraint failure. It
generates longer molecules (mean 69 vs 39 characters — chains of repeated
amide units), so 26% of its rows hit `max_new_tokens=100` mid-molecule; greedy
hits it 1% of the time.

## What this task shows that the other three do not

In `wos`, `gloss_translation` and `conll_ner` the grammar decides the outcome:
the output space is a closed label set, and constraint satisfaction is the
task. Here it is not.

The OpenSMILES grammar is context-free, so it guarantees **syntax** — bond
order, bracket-atom structure, branch nesting, balanced parentheses — and the
runs confirm it does (99% grammar-satisfied under greedy). But a SMILES string
is only a molecule if its ring-closure digits pair up and its valences hold,
and neither is expressible in a CFG (the official OpenSMILES EBNF cannot state
them either). Ring-pairing accounts for ~89% of the invalid outputs.

That is why the unconstrained baseline scores almost the same chemical
validity as the constrained run (0.186 vs 0.192 greedy, 0.657 vs 0.683
beam3): the binding constraint on this task lives past the grammar's reach.

What the grammar does buy is a guarantee about the *shape* of the output.
Both unconstrained runs contain rows where the model answers in prose
(`"I can't provide a SMILES string for a molecule with 7 carbon atoms…"`),
which no downstream parser can consume; both constrained runs contain zero —
every one of the 4000 constrained outputs is SMILES-alphabet text. It is one
row in 2000 here because the task is easy to stay on-format for; on a task
where the model is likelier to editorialise, that guarantee is the difference
between a pipeline that runs and one that crashes. It is a floor, not an
improvement in quality — and on this task the floor was already nearly met.

What does separate them is **search**. Beam3 gives 3.6× the chemical validity
of greedy, because closing a ring is exactly the kind of decision greedy
cannot revisit: it opens `c1`, commits, and the mask cannot force a matching
closure later. That reproduces the pattern the other three tasks show — the
gain concentrates on the deep decision, not the first one — but here the
effect is much larger, since ring closure is a long-range dependency rather
than a one-token choice. Note the gain is search-driven, not grammar-driven:
`nogram_beam3` gets most of it too (0.657 validity, 0.436 yield).

The trade-off beam3 makes is visible in the numbers: longer, more repetitive
molecules (mean 69 vs 39 characters), valid but less drug-like. On rows valid
under *both* decodings, greedy's molecules score higher QED (0.814 vs 0.648).
Beam3 wins on `yield` anyway, because it produces a valid molecule far more
often.

## Known weakness

Uniqueness is low in every run (0.10–0.16): with the few-shot pool fixed and
decoding deterministic, the only thing varying between rows is the target
formula, and a 1B model largely ignores it — 2000 prompts collapse onto a few
hundred distinct molecules, mostly variations on `CC(=O)Nc1ccc...`. The
reference setup avoids this by resampling the 20 few-shot examples per prompt,
which this run holds fixed by design. Anyone extending this task should treat
diversity, not validity, as the open problem: sampling (`do_sample=True`) or a
resampled few-shot pool are the two levers, both of which would break the
deterministic greedy-vs-beam comparison this suite is built around.
