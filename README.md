# PULSE

This folder contains the code to reproduce the experiments and
figures in the PULSE paper (Prediction-powered Unbiased Low-rank
Sequential Evaluation).

The headline experiment is **test-with-train-side**: identifying the
best-mean model from a 1000-arm test pool using UCB-E with a low-rank
matrix-factorization proxy trained on a separate train matrix
of historical model evaluations.

Three methods are compared:

| Method            | Variant flag        | Color in plots          |
| ----------------- | ------------------- | ----------------------- |
| UCB-E (baseline)  | `std_wor`           | green (`#55A868`)       |
| **PULSE**         | `cv_wor`            | blue  (`#4C72B0`)       |
| Naive-Pooling     | `imp_wor_noretrain` | red   (`#C44E52`)       |

## Layout

```
pulse-llm-eval/
├── README.md                                  (this file)
├── requirements.txt
├── test_with_train_side.py                    entry point: warm-cache / populate-prewarm / variant / merge
├── bandit.py                                  bandit runners + NuclearNormMF
├── bandit_si.py                               SI orchestration (V cache, U prewarm, retrains)
├── data_loaders.py                            CSV → binary observation matrix
├── build_test_subsets.py                      build test_delta_*.csv from test.csv
├── plot_test_cvopt_zoom_inset.py              4 main paper figures (--with-retrain)
├── plot_legend.py                             standalone legend PDF
├── plot_budget_to_95_bars.py                  bar chart (savings @ 95% accuracy)
├── make_budget_to_95_csvs.py                  feeds plot_budget_to_95_bars.py
├── configs/                                   chosen hyperparameters (CV-tuned)
│   ├── cvopt_bbh.json                         BBH suite
│   └── cvopt_MMLU.json                        MMLU suite
└── slurm/                                     SLURM templates (edit account/partition)
    ├── run_warm_cache.sh
    ├── run_populate_prewarm.sh
    ├── run_test_with_train_side.sh
    ├── run_cpu.sh
    └── run_cpu_fat.sh
```

## Environment

```bash
pip install -r requirements.txt
```

A NVIDIA GPU is recommended for the stage-1 V cache and per-seed
prewarm cache (~20 minutes per dataset on an A40). The bandit loop
itself is CPU-only and parallelizable across seeds.

## Data

Place the following CSVs at the indicated paths (relative to this
folder). All files share the schema:

- index column = model id
- optional metadata columns `created_date`, `sha` (dropped on load)
- remaining columns = per-example correctness, ∈ {0, 1, NaN} (NaN → 0,
  > 0 → 1)

Required files (dataset shapes vary per benchmark, but the schema is
identical):

```
pulse-llm-eval/
├── bbh/                          BBH+GPQA+IFEval+MATH+MuSR composite
│   ├── train.csv                 historical models × examples (side info)
│   ├── test.csv                  candidate models × examples (test pool)
│   ├── test_delta_0.02.csv       1000-row subset of test, top-arm gap = 0.02
│   └── test_delta_0.03.csv       1000-row subset of test, top-arm gap = 0.03
└── MMLU/                         MMLU-Pro
    ├── train.csv
    ├── test.csv
    ├── test_delta_0.02.csv
    └── test_delta_0.03.csv
```

`train.csv` and `test.csv` for both benchmarks come from the
[skbwu/efficiently-evaluating-llms](https://github.com/skbwu/efficiently-evaluating-llms/tree/main/data/processed)
repository (commit
[`857ee18607bd9c84e90431ce0b8f36fc3f72ae68`](https://github.com/skbwu/efficiently-evaluating-llms/tree/857ee18607bd9c84e90431ce0b8f36fc3f72ae68/data/processed)).
Pull the CSVs from `data/processed/` there and rename them to the
layout above.

The `test_delta_0.0X.csv` files are 1000-row subsets of the
corresponding `test.csv` chosen so the gap between the top arm and the
second-best arm is exactly `0.0X`. Build them with:

```bash
python build_test_subsets.py --src bbh/test.csv
python build_test_subsets.py --src MMLU/test.csv
```

Each invocation writes `test_delta_{0.001, 0.003, 0.005, 0.01, 0.02,
0.03, 0.05, 0.10}.csv` next to the source CSV.

## Pipeline

For each `(family, dataset)` combination — i.e. `(bbh,
test_delta_0.02)`, `(bbh, test_delta_0.03)`, `(MMLU,
test_delta_0.02)`, `(MMLU, test_delta_0.03)` — run the four stages
below. Substitute the right `--train-path`, `--test-path`, and
hyperparameters from the corresponding `configs/` JSON.

### Hyperparameters (from CV-tuned configs)

| Family | rank | stage1 lr | stage1 lam | prewarm epochs | prewarm lr | prewarm lam |
| ------ | ---- | --------- | ---------- | -------------- | ---------- | ----------- |
| `bbh`  | 100  | 0.01      | **0.01**   | 50             | 0.01       | 0.01        |
| `MMLU` | 100  | 0.01      | **0.001**  | 50             | 0.01       | 0.01        |

### Stage A — stage-1 V cache (one-shot per dataset, GPU, ~minutes)

```bash
python test_with_train_side.py --warm-cache --device cuda \
    --mf-rank 100 --mf-init-epochs 5 \
    --stage1-lr 0.01 --stage1-lam 0.01 \
    --prewarm-epochs 50 --prewarm-lr 0.01 --prewarm-lam 0.01 \
    --train-path bbh/train.csv \
    --test-path bbh/test.csv
```

Repeat per `--test-path` (test.csv plus each `test_delta_0.0X.csv`).
For `MMLU` add `--stage1-lam 0.001 --prewarm-lam 0.001`.

### Stage B — per-seed prewarm cache (one-shot per dataset, GPU, ~1h for 500 seeds)

```bash
python test_with_train_side.py --populate-prewarm-cache --device cuda \
    --n-seeds 500 --batch-size 64 \
    --mf-rank 100 --stage1-lr 0.01 --stage1-lam 0.01 \
    --prewarm-epochs 50 --prewarm-lr 0.01 --prewarm-lam 0.01 \
    --train-path bbh/train.csv \
    --test-path bbh/test_delta_0.02.csv
```

Caches land in `output_si/mf_cache/prewarm/`. Bit-identical across
runs of `cv_wor` / `imp_wor` on the same `(seed, dataset)`.

### Stage C — bandit runs (CPU multiprocess, ~hours per variant × dataset)

Run each of `std_wor`, `cv_wor`, `imp_wor` (and optionally
`imp_wor_noretrain`) per dataset:

```bash
SI_WARMUP_STEPS=0 SI_TRAIN_EVERY=1000 \
python test_with_train_side.py --variant cv_wor \
    --n-seeds 500 --processes 50 --batch-size 64 --budget 400000 \
    --mf-rank 100 --stage1-lr 0.01 --stage1-lam 0.01 \
    --prewarm-epochs 50 --prewarm-lr 0.01 --prewarm-lam 0.01 \
    --train-path bbh/train.csv \
    --test-path bbh/test_delta_0.02.csv \
    --device cpu --save-raw \
    --out-suffix _delta0.02_bs64_b400K_initretrain_cvopt_warmup0
```

The exact `--out-suffix` matters — the plot scripts read npz files by
filename. Use these conventions:

| Variant             | npz suffix tail                          |
| ------------------- | ---------------------------------------- |
| `std_wor`           | `_bs64_b400K`                            |
| `cv_wor`            | `_bs64_b400K_initretrain_cvopt_warmup0`  |
| `imp_wor`           | `_bs64_b400K_initretrain_cvopt_warmup0`  |
| `imp_wor_noretrain` | `_bs64_b400K_cvopt_noretrain_warmup0`    |

Prepend the dataset piece (`_delta0.02`, `_delta0.03`) and, for the
`MMLU` family, an `_MMLU` token between the variant and the suffix
tail. Final filenames look like:

```
output_si/test_with_train_side__std_wor_MMLU_delta0.02_bs64_b400K.npz
output_si/test_with_train_side__cv_wor_MMLU_delta0.02_bs64_b400K_initretrain_cvopt_warmup0.npz
output_si/test_with_train_side__imp_wor_noretrain_MMLU_delta0.02_bs64_b400K_cvopt_noretrain_warmup0.npz
```

### Stage D — figure generation

After all variants × datasets have completed:

```bash
# 4 zoom-inset figures (the paper's main result)
for combo in "bbh 0.02" "bbh 0.03" "MMLU 0.02" "MMLU 0.03"; do
    python plot_test_cvopt_zoom_inset.py $combo --with-retrain
done

# Bar chart of LLM calls to reach 95% accuracy
python make_budget_to_95_csvs.py
python plot_budget_to_95_bars.py

# Standalone legend
python plot_legend.py
```

PDFs land in `plots_bbh/` and `plots_MMLU/`. The paper uses:

- `plots_<family>/budget_to_95_bars_d02_d03_horizontal.pdf` — savings bar chart
- `plots_<family>/legend.pdf` — shared legend
- `plots_<family>/<family>_cvopt_delta0.0X_top1_zoom_inset_with_retrain.pdf` — main 4 figures

## SLURM

Templates in `slurm/` give the canonical submission shape.

- `run_warm_cache.sh` — Stage A
- `run_populate_prewarm.sh` — Stage B
- `run_test_with_train_side.sh` — Stage C, multi-GPU chunked
- `run_cpu.sh` — Stage C, generic CPU multiprocess
- `run_cpu_fat.sh` — Stage C, same as above but tuned for ~200+ CPU
  nodes (pin via `--nodelist=<node>` at submit time)

You will need to edit each script to:

1. Add `--account=<your account>` and `--partition=<your partition>`
   to the `sbatch` invocation (the templates omit them so you can
   wire in your own).
2. Activate your Python environment in the marked `# TODO:` block
   near the top of each script.

Submission pattern:

```bash
sbatch --account=<acct> --partition=<part> --time=04:00:00 \
    --export="ALL,VARIANT=cv_wor,N_SEEDS=500,SEED_START=0,PROCESSES=50,
BATCH_SIZE=64,BUDGET=400000,
SI_MF_RANK=100,STAGE1_LR=0.01,STAGE1_LAM=0.01,
SI_PREWARM_EPOCHS=50,SI_PREWARM_LR=0.01,SI_PREWARM_LAM=0.01,
SI_TRAIN_EVERY=1000,SI_WARMUP_STEPS=0,
TRAIN_PATH=bbh/train.csv,
TEST_PATH=bbh/test_delta_0.02.csv,
FINAL_SUFFIX=_delta0.02_bs64_b400K_initretrain_cvopt_warmup0,
CHUNK_TAG=cpu_d02" \
    --job-name=ts_cv_d02 slurm/run_cpu.sh
```

## Caveat: env vars matter

- `SI_WARMUP_STEPS=0` is mandatory for the cvopt headline runs.
  Default is 10% of the budget steps (legacy behavior); the paper's
  results assume no in-bandit UCB-E warmup.
- `SI_TRAIN_EVERY=1000` controls retrain cadence (default 1000).
- `SI_INIT_COUNT_BUDGET=0` (default) means `--budget` is post-init
  bandit cells. Setting `=1` makes init-mask cells count toward the
  budget.
