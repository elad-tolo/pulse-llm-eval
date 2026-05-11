#!/bin/bash
#SBATCH --cpus-per-task=100
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ts_cpu_%x_%j.log

# Generic CPU multiprocess runner for any test_with_train_side variant.
# Submit with VARIANT/BATCH_SIZE/N_SEEDS/PROCESSES env overrides:
#
#   sbatch --export=ALL,VARIANT=cv_wor,BATCH_SIZE=64,N_SEEDS=100,FINAL_SUFFIX=_bs64,CHUNK_TAG=cpu_bs64 \
#       --job-name=ts_cv_wor_cpu_bs64 slurm/run_cpu.sh

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

# TODO: activate your Python environment (conda/venv/etc.) here.

mkdir -p logs output_si

VARIANT="${VARIANT:-cv_wor}"
N_SEEDS="${N_SEEDS:-100}"
SEED_START="${SEED_START:-0}"
PROCESSES="${PROCESSES:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUDGET="${BUDGET:-1000000}"
MF_RANK="${SI_MF_RANK:-10}"
TEST_PATH="${TEST_PATH:-bbh/test.csv}"
TRAIN_PATH="${TRAIN_PATH:-bbh/train.csv}"
STAGE1_LR="${STAGE1_LR:-0.1}"
STAGE1_LAM="${STAGE1_LAM:-0.01}"
PREWARM_EPOCHS="${SI_PREWARM_EPOCHS:-10}"
PREWARM_LR="${SI_PREWARM_LR:-0.1}"
PREWARM_LAM="${SI_PREWARM_LAM:-0.01}"
INIT_PULLS_PER_ARM="${SI_INIT_PULLS_PER_ARM:-1}"
CHUNK_TAG="${CHUNK_TAG:-cpu}"
FINAL_SUFFIX="${FINAL_SUFFIX:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SI_INIT_PULLS_PER_ARM="$INIT_PULLS_PER_ARM"

python -u test_with_train_side.py \
    --variant "$VARIANT" \
    --n-seeds "$N_SEEDS" --seed-start "$SEED_START" \
    --processes "$PROCESSES" \
    --batch-size "$BATCH_SIZE" --budget "$BUDGET" \
    --mf-rank "$MF_RANK" \
    --test-path "$TEST_PATH" --train-path "$TRAIN_PATH" \
    --stage1-lr "$STAGE1_LR" --stage1-lam "$STAGE1_LAM" \
    --prewarm-epochs "$PREWARM_EPOCHS" \
    --prewarm-lr "$PREWARM_LR" --prewarm-lam "$PREWARM_LAM" \
    --device cpu \
    --save-raw \
    --out-suffix "_gpu_chunk_${CHUNK_TAG}_0"

if [ "${MERGE_AT_END:-1}" = "1" ]; then
    echo "Merging into ${FINAL_SUFFIX:-canonical} npz..."
    python -u test_with_train_side.py --merge --variant "$VARIANT" \
        --chunk-tag-filter "${CHUNK_TAG}" \
        --final-suffix "${FINAL_SUFFIX}"
fi

echo "Done."
