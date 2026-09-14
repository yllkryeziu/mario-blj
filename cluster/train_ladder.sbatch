#!/usr/bin/env bash
#SBATCH --job-name=mario-blj-ladder
#SBATCH --partition=standard
#SBATCH --exclude=hai002
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=none
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ladder_%A_%a.out
#SBATCH --error=logs/ladder_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/fast/project/HFMI_SynergyUnit/yll/mario-blj}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
SEEDS="${SEEDS:-0,1,2}"
TIMESTEPS="${TIMESTEPS:-2000000}"
NUM_ENVS="${NUM_ENVS:-${SLURM_CPUS_PER_TASK:-8}}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/ladder}"
ROM="${ROM:-${PROJECT_ROOT}/roms/baserom.us.z64}"
COLLISION="${COLLISION:-${PROJECT_ROOT}/third_party/sm64-port/levels/castle_inside/areas/2/collision.inc.c}"
LIBSM64="${LIBSM64:-${PROJECT_ROOT}/third_party/libsm64/dist/libsm64.so}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUT_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

read -r RUNG SEED < <("${PYTHON}" -c "
import sys
from src.train import ladder
matrix = ladder.job_matrix(tuple(int(s) for s in '${SEEDS}'.split(',')))
index = int('${TASK_ID}')
if index >= len(matrix):
    sys.exit('array index %d exceeds the %d jobs in the ladder' % (index, len(matrix)))
print('%s %d' % matrix[index])
")

echo "host=$(hostname) task=${TASK_ID} rung=${RUNG} seed=${SEED} cpus=${NUM_ENVS}"

srun "${PYTHON}" scripts/train_blj.py \
  --rung="${RUNG}" \
  --seeds="${SEED}" \
  --timesteps="${TIMESTEPS}" \
  --num_envs="${NUM_ENVS}" \
  --out_dir="${OUT_DIR}" \
  --rom="${ROM}" \
  --collision="${COLLISION}" \
  --library="${LIBSM64}" \
  --checkpoint_every=100000
