# smiles_qed — drug-like molecule generation (SMILES → QED)

Molecule generation through grammarllm constrained decoding, scored by QED.
Replicates the reference setup: few-shot SMILES prompting on GDB-17
(Ruddigkeit et al., 2012), QED (Bickerton et al., 2012) as the fitness
function.

## How the task was built

- **Dataset**: [`Pixelatory/GDB-17`](https://huggingface.co/datasets/Pixelatory/GDB-17),
  a re-publication of the paper's GDB-17-Set (50M canonical SMILES).
  `prepare_data.py` reads the first 2M rows of the CSV, computes each
  molecular formula with RDKit, deduplicates, and takes a seeded shuffle:
  20 rows for the fixed few-shot pool, 2000 for the fixed test set.
- **Model**: Llama-3.1-8B-Instruct (the paper's) and Llama-3.2-1B-Instruct,
  for a size comparison.
- **Grammar**: the full OpenSMILES §2.2 grammar from `examples/smiles.py`.
- **Prompt**: each test row is conditioned on its own `molecular_formula`
  (the 20 few-shot examples are formula → SMILES pairs). This is a deviation
  from the paper, which generates unconditionally — but with
  `do_sample=False` and a fixed few-shot pool, an unconditional prompt is
  identical for every row and would return the identical molecule 2000
  times. Conditioning on the formula is what makes rows differ while
  keeping decoding deterministic.
- **Decoding**: `do_sample=False`, greedy and beam3, `max_new_tokens=100`,
  native chat template, token-lookahead engine on.

```bash
python prepare_data.py --dataset gdb17
CUDA_VISIBLE_DEVICES=2 python pipe_bench.py --dataset gdb17 \
    --model /path/to/Llama-3.1-8B-Instruct --name 8b_beam1 --batch 2
CUDA_VISIBLE_DEVICES=2 python pipe_bench.py --dataset gdb17 \
    --model /path/to/Llama-3.1-8B-Instruct --name 8b_beam3 --num-beams 3 --batch 2
python score_bench.py output/gdb17/*/predictions.csv
```

## Metrics

- **QED** (Quantitative Estimate of Drug-likeness) — a score in [0, 1] for
  how closely a molecule's physicochemical properties (molecular weight,
  lipophilicity, polar surface area, aromatic rings, etc.) resemble those of
  approved oral drugs. Undefined for a string that isn't a valid molecule.
- **RDKit valid** — share of predictions RDKit can parse into a molecule.
  Stricter than grammar validity: the OpenSMILES grammar is context-free, so
  it cannot enforce that ring-closure digits pair up or that valences hold —
  a string can be grammatically perfect and still not be a molecule.
- **grammar** — share of rows whose PDA stack emptied (no truncation at
  `max_new_tokens`). Syntax only, independent of RDKit validity.
- **QED over all** — mean QED across all 2000 rows, counting an unparseable
  prediction as 0. This is what the paper reports, and the number to compare
  runs on: mean QED *over valid rows only* rewards a run that fails often
  and succeeds only on easy molecules.

## Results — GDB-17, 2000 rows (gold QED 0.564)

| run | grammar | RDKit valid | QED (valid rows) | **QED over all** |
|---|---|---|---|---|
| 1B greedy | 0.964 | 0.173 | 0.433 | 0.075 |
| 1B beam3 | 0.894 | 0.387 | 0.477 | 0.185 |
| 8B greedy | 0.870 | 0.689 | 0.602 | 0.415 |
| 8B beam3 | 0.882 | 0.784 | 0.621 | 0.487 |

Reference paper (QED over all): base LM 0.132 (0.12–0.15), locally
constrained decoding 0.189 (0.17–0.21). Not directly comparable to the table
above — this setup conditions on the molecular formula, which the paper's
does not, making the task easier.
