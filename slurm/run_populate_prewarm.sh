#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=ts_pp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ts_pp_%x_%j.log

# GPU prewarm-cache populator for test_with_train_side.py.
#
# Loops N_SEEDS sequentially in one Python process on a single GPU,
# running the deterministic per-seed prewarm and saving each result to
# output_si/mf_cache/prewarm/. Subsequent CPU bandit jobs (cv_wor,
# imp_wor, ...) cache-hit and skip prewarm entirely.
#
# Submit one per dataset (full test + delta_{0.01,0.02,0.03}). Each
# takes ~7s/seed on a single GPU at the cvopt config (50 prewarm
# epochs); 500 seeds ≈ 1 hour wall.
#
# Required env: TEST_PATH (e.g. bbh/test.csv).
# Optional: TRAIN_PATH, N_SEEDS, SEED_START, BATCH_SIZE,
# SI_MF_RANK, STAGE1_LR, STAGE1_LAM,
# SI_PREWARM_EPOCHS, SI_PREWARM_LR, SI_PREWARM_LAM,
# SI_INIT_PULLS_PER_ARM (default 1).

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

# TODO: activate your Python environment (conda/venv/etc.) here.

mkdir -p logs output_si/mf_cache/prewarm

nvidia-smi || true

N_SEEDS="${N_SEEDS:-500}"
SEED_START="${SEED_START:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_PATH="${TEST_PATH:-bbh/test.csv}"
TRAIN_PATH="${TRAIN_PATH:-bbh/train.csv}"
MF_RANK="${SI_MF_RANK:-100}"
MF_INIT_EPOCHS="${SI_MF_INIT_EPOCHS:-5}"
STAGE1_LR="${STAGE1_LR:-0.01}"
STAGE1_LAM="${STAGE1_LAM:-0.01}"
PREWARM_EPOCHS="${SI_PREWARM_EPOCHS:-50}"
PREWARM_LR="${SI_PREWARM_LR:-0.01}"
PREWARM_LAM="${SI_PREWARM_LAM:-0.01}"
INIT_PULLS_PER_ARM="${SI_INIT_PULLS_PER_ARM:-1}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export SI_INIT_PULLS_PER_ARM="$INIT_PULLS_PER_ARM"

echo "Populating prewarm cache: test=$TEST_PATH n_seeds=$N_SEEDS"
echo "  cvopt config: rank=$MF_RANK stage1_lr=$STAGE1_LR stage1_lam=$STAGE1_LAM"
echo "  prewarm: ep=$PREWARM_EPOCHS lr=$PREWARM_LR lam=$PREWARM_LAM bs=$BATCH_SIZE"
echo "  init_pulls_per_arm=$INIT_PULLS_PER_ARM"

python -u test_with_train_side.py --populate-prewarm-cache \
    --device cuda \
    --n-seeds "$N_SEEDS" --seed-start "$SEED_START" \
    --batch-size "$BATCH_SIZE" \
    --mf-rank "$MF_RANK" --mf-init-epochs "$MF_INIT_EPOCHS" \
    --test-path "$TEST_PATH" --train-path "$TRAIN_PATH" \
    --stage1-lr "$STAGE1_LR" --stage1-lam "$STAGE1_LAM" \
    --prewarm-epochs "$PREWARM_EPOCHS" \
    --prewarm-lr "$PREWARM_LR" --prewarm-lam "$PREWARM_LAM"

echo "Done."
