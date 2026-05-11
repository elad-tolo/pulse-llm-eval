#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=ts_warm
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ts_warm_%j.log

# One-shot stage-1 V cache warm-up for test_with_train_side.py.
# Trains the side-only V on the train matrix and writes to
# output_si/mf_cache/mf_side_only_<sha1>.pt. All variant jobs that
# follow load this cache instantly.

set -euo pipefail

# Resolve repo root from the script location (slurm/<this>.sh -> ..).
cd "$(dirname "$(readlink -f "$0")")/.."

# TODO: activate your Python environment (conda/venv/etc.) here.
# Example: `source <conda_root>/etc/profile.d/conda.sh && conda activate <env>`

mkdir -p logs output_si/mf_cache

nvidia-smi || true

python -u test_with_train_side.py --warm-cache --device cuda \
    --mf-rank "${MF_RANK:-10}" \
    --test-path "${TEST_PATH:-bbh/test.csv}" \
    --train-path "${TRAIN_PATH:-bbh/train.csv}" \
    --stage1-lr "${STAGE1_LR:-0.1}" \
    --stage1-lam "${STAGE1_LAM:-0.01}" \
    --prewarm-epochs "${SI_PREWARM_EPOCHS:-10}" \
    --prewarm-lr "${SI_PREWARM_LR:-0.1}" \
    --prewarm-lam "${SI_PREWARM_LAM:-0.01}"
