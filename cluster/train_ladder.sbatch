#!/usr/bin/env bash
#SBATCH --job-name=mario-blj-ladder
#SBATCH --partition=standard
#SBATCH --exclude=hai002
#SBATCH --array=0-23
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
# Defaults are the published ladder: the four reward rungs, six seeds, 20M steps each, which is
# exactly the 24 runs reduced into results/learning_curves.json. The two cut rungs (curriculum and
# the action repeat sweep) are still in src/train/ladder.py and still selectable through RUNGS,
# but they are not part of the published result and are not in the default matrix.
#
#   sbatch cluster/train_ladder.sbatch
#
# is the command behind every ladder number in the README. Overriding RUNGS or SEEDS changes the
# size of the job matrix, so --array above has to be changed to match; the guard below refuses to
# run rather than silently dropping the jobs that fall past the end of the array.
RUNGS="${RUNGS:-terminal,speed,height,height_speed}"
SEEDS="${SEEDS:-0,1,2,3,4,5}"
TIMESTEPS="${TIMESTEPS:-20000000}"
NUM_ENVS="${NUM_ENVS:-${SLURM_CPUS_PER_TASK:-12}}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/ladder_v2}"
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
import os
import sys
from src.train import ladder
matrix = ladder.job_matrix(
    tuple(int(s) for s in '${SEEDS}'.split(',')),
    tuple(name for name in '${RUNGS}'.split(',') if name),
)
index = int('${TASK_ID}')
if index >= len(matrix):
    sys.exit('array index %d exceeds the %d jobs in the ladder' % (index, len(matrix)))
last = os.environ.get('SLURM_ARRAY_TASK_MAX')
if last is not None and int(last) + 1 < len(matrix):
    sys.exit(
        'the array covers %s tasks but this RUNGS/SEEDS selection is %d jobs; '
        'set --array=0-%d or the last %d jobs never run'
        % (int(last) + 1, len(matrix), len(matrix) - 1, len(matrix) - int(last) - 1)
    )
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
