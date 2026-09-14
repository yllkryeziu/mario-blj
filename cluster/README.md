# Running the ladder on the cluster

The ablation ladder is 7 runs (4 rungs, with rung 3 swept over 4 action repeat values) times
however many seeds you want. At 3 seeds that is 21 independent training jobs, which maps onto
a Slurm array one task per (rung, seed) pair.

Every job is CPU bound. libsm64 is a CPU physics library and the policy is a two layer MLP over
24 floats, so `train_ladder.sbatch` asks for `--cpus-per-task` and `--gres=none` and loads no
CUDA modules. It also passes `--exclude=hai002`, because `cuInit` returns 999 on that node and
it will take your tasks down with it. The usable nodes are hai001 and hai003 through hai008.

## Sync

```bash
rsync -av --delete \
  --exclude .git --exclude .venv --exclude __pycache__ \
  --exclude results --exclude 'third_party/libsm64/dist' \
  ~/Desktop/mario-blj/ berlin1:/fast/project/HFMI_SynergyUnit/yll/mario-blj/
```

The ROM is excluded from git but not from rsync, so `roms/baserom.us.z64` travels with the
repo. libsm64 reads Mario's animation and texture data out of it at runtime, and a job without
it fails at the first `reset`.

## Build

```bash
ssh berlin1
cd /fast/project/HFMI_SynergyUnit/yll/mario-blj
bash scripts/setup.sh
/opt/miniforge3/bin/python3 -m pip install --user \
  numpy gymnasium absl-py pytest 'stable-baselines3>=2.0' torch
```

Two things are worth checking before you burn an array on them:

```bash
ls -la third_party/libsm64/dist/libsm64.so
/opt/miniforge3/bin/python3 -m pytest src/train/ppo_test.py -q
```

The test suite needs neither the ROM nor the shared library, so it passing tells you the
harness and the rung table survived the trip. The `ls` tells you whether the physics did.

Build on a compute node rather than the login node if `make` is slow or gets killed:

```bash
srun --partition=standard --exclude=hai002 --cpus-per-task=8 --pty bash scripts/setup.sh
```

## Launch

```bash
sbatch cluster/train_ladder.sbatch
```

The array range in the script is `0-20`, which is the 21 jobs of the full ladder at seeds
`0,1,2`. Change the seed count and the range has to move with it. Ask the rung table for the
number rather than counting by hand:

```bash
/opt/miniforge3/bin/python3 -c "
from src.train import ladder
print(len(ladder.job_matrix((0, 1, 2, 3, 4))))"
```

Then override both at submit time. Set `SEEDS` in the calling shell rather than passing it
through `--export`, because `--export` is itself a comma separated list and would read the
seed list as separate variables. sbatch forwards the whole environment by default:

```bash
SEEDS=0,1,2,3,4 sbatch --array=0-34 cluster/train_ladder.sbatch
```

The mapping from array index to (rung, seed) comes from `ladder.job_matrix`, so the script and
the analysis agree on which task trained what. Seeds vary fastest, meaning all seeds of one
rung sit in a contiguous block of indices. A single rung is therefore a subrange:

```bash
sbatch --array=0-2 cluster/train_ladder.sbatch
```

Other knobs travel the same way, set in the shell ahead of `sbatch`: `TIMESTEPS`, `NUM_ENVS`,
`OUT_DIR`, `ROM`, `COLLISION`, `LIBSM64`. `NUM_ENVS` defaults to `SLURM_CPUS_PER_TASK`,
since one libsm64 process per environment is the only safe arrangement and one core per
process is the right ratio.

## Watch

```bash
squeue -u "$USER"
tail -f logs/ladder_*_0.out
```

Each job writes into `results/ladder/<rung>/seed_<seed>/`:

- `episodes.csv` appended as episodes finish. A job killed by the 24 hour wall clock still
  leaves a readable learning curve.
- `checkpoints/` every 100k agent steps, so a timeout costs you the tail of a run and not the
  whole run.
- `metrics.json` and `model.zip` written on clean completion.

## Collect

```bash
rsync -av berlin1:/fast/project/HFMI_SynergyUnit/yll/mario-blj/results/ladder/ \
  ~/Desktop/mario-blj/results/ladder/
```

The ladder reads off the per run summaries:

```bash
/opt/miniforge3/bin/python3 -c "
import glob
from src.train import ppo
for path in sorted(glob.glob('results/ladder/*/seed_*/metrics.json')):
    m = ppo.load_metrics(path)['metrics']
    print('%-12s seed %d  success %.3f  peak_vel %8.2f  first_success %s' % (
        m['rung'], m['seed'], m['success_rate'], m['peak_backward_velocity'],
        m['frames_to_first_success']))"
```

`frames_to_first_success` is the number that carries the finding. A rung where it is `None`
across every seed is a rung whose shaping was not enough.

## Resuming

A timed out job leaves its last checkpoint in `checkpoints/`. Resubmitting that one array index
restarts it from scratch rather than from the checkpoint, so raise `--time` or lower
`TIMESTEPS` instead of resubmitting into the same wall clock. `episodes.csv` is opened in
append mode, so a restart adds to the existing curve instead of truncating it, and the episode
counter restarts at 1. Watch for that when plotting.
