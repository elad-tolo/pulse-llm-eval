#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ts_%x_%j.log

# Per-variant N-GPU runner for test_with_train_side.py. Submit one per
# variant; pass --gres=gpu:N and --nodelist=<node> at submit time.
# Splits N_SEEDS evenly across the N GPUs (ceil partition).

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."

# TODO: activate your Python environment (conda/venv/etc.) here.

mkdir -p logs output_si

VARIANT="${VARIANT:-cv_wor}"
N_SEEDS="${N_SEEDS:-10}"
SEED_START="${SEED_START:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# Chunks share the variant prefix; CHUNK_TAG makes them unique across
# concurrent SLURM jobs so their `rm`s don't race.
CHUNK_TAG="${CHUNK_TAG:-${SLURM_JOB_ID:-local}}"
# FINAL_SUFFIX is appended to the merged npz filename so e.g. bs=64
# runs don't overwrite bs=1 results.
FINAL_SUFFIX="${FINAL_SUFFIX:-}"
TEST_PATH="${TEST_PATH:-bbh/test.csv}"
TRAIN_PATH="${TRAIN_PATH:-bbh/train.csv}"
STAGE1_LR="${STAGE1_LR:-0.1}"
STAGE1_LAM="${STAGE1_LAM:-0.01}"
PREWARM_EPOCHS="${SI_PREWARM_EPOCHS:-10}"
PREWARM_LR="${SI_PREWARM_LR:-0.1}"
PREWARM_LAM="${SI_PREWARM_LAM:-0.01}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

nvidia-smi || true

# Clear stale chunks belonging to *this* job/tag only (no race with peers).
rm -f output_si/test_with_train_side__${VARIANT}_gpu_chunk_${CHUNK_TAG}_*.npz

# Detect available GPUs from the SLURM allocation.
N_GPUS="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ -z "$N_GPUS" ] || [ "$N_GPUS" -lt 1 ]; then
    N_GPUS=1
fi
echo "Partitioning $N_SEEDS seeds across $N_GPUS GPUs"

# Even ceil-partition: each chunk gets ceil(remaining / chunks_left) seeds.
COUNTS=()
STARTS=()
S=$SEED_START
LEFT=$N_SEEDS
for ((i=0; i<N_GPUS; i++)); do
    GPUS_LEFT=$((N_GPUS - i))
    COUNT=$(( (LEFT + GPUS_LEFT - 1) / GPUS_LEFT ))
    COUNTS+=($COUNT)
    STARTS+=($S)
    S=$((S + COUNT))
    LEFT=$((LEFT - COUNT))
done

PIDS=()
for ((i=0; i<N_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python -u test_with_train_side.py \
        --variant "$VARIANT" \
        --n-seeds "${COUNTS[$i]}" --seed-start "${STARTS[$i]}" \
        --batch-size "${BATCH_SIZE}" \
        --test-path "$TEST_PATH" --train-path "$TRAIN_PATH" \
        --stage1-lr "$STAGE1_LR" --stage1-lam "$STAGE1_LAM" \
        --prewarm-epochs "$PREWARM_EPOCHS" \
        --prewarm-lr "$PREWARM_LR" --prewarm-lam "$PREWARM_LAM" \
        --save-raw --out-suffix "_gpu_chunk_${CHUNK_TAG}_${i}" \
        > "logs/ts_${VARIANT}_chunk_${CHUNK_TAG}_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${N_GPUS} chunks (pids=${PIDS[*]}, counts=${COUNTS[*]}, starts=${STARTS[*]}, tag=${CHUNK_TAG}), waiting..."
wait "${PIDS[@]}"

echo "All chunks done."
# Skip auto-merge when MERGE_AT_END=0 (multiple peer jobs feed a single merge).
if [ "${MERGE_AT_END:-1}" = "1" ]; then
    echo "Merging (filter=${CHUNK_TAG}, final_suffix=${FINAL_SUFFIX})..."
    python -u test_with_train_side.py --merge --variant "$VARIANT" \
        --chunk-tag-filter "${CHUNK_TAG}" \
        --final-suffix "${FINAL_SUFFIX}"
fi
echo "Done."
