"""Measures how fast the environment steps, alone and inside the training loop.

The post quotes two throughput figures and neither had an artefact behind it, which made them the
only numbers in it that nothing measured. This writes both.

The two are worth separating because they are nearly equal, which is the interesting part. One
process stepping the environment with no policy attached is the physics cost on its own. The
training loop runs eight environment subprocesses against a learner in the parent, on a machine
with eight cores, so the workers contend with each other and with the learner and the parallelism
buys batched rollouts for PPO rather than raw throughput.

The loop rate has to be fitted rather than timed directly. Spawning eight subprocesses, opening
eight copies of libsm64 and loading the ROM into each costs a couple of seconds that a short run
cannot amortise, so a single timing reports whatever fraction of it happened to be startup: the
same configuration measures about 3,400 steps a second over four updates and about 6,200 over
thirty. Timing two lengths and taking the slope separates the fixed cost from the marginal rate,
and the residual is reported so a bad fit is visible rather than silent.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.env import endless_stairs
from src.env.blj_env import BljConfig, BljEnv
from src.train import ppo

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    """Builds the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default=os.path.join(_ROOT, "roms", "baserom.us.z64"),
                        help="ROM path.")
    parser.add_argument("--env_steps", type=int, default=20000,
                        help="Steps to time for the single process measurement.")
    parser.add_argument("--env_warmup", type=int, default=500,
                        help="Steps to run before timing, so the first reset is not counted.")
    parser.add_argument("--loop_steps", default="16384,40960,122880",
                        help="Agent step counts to time the training loop at. The rate is the "
                             "slope through the shortest and longest, so they must differ enough "
                             "for it to mean anything, and a third in between is what makes the "
                             "reported residual a check on the model rather than zero by "
                             "construction.")
    parser.add_argument("--rung", default="terminal", help="Rung to train for the loop timing.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for both measurements.")
    parser.add_argument("--out", default=os.path.join(_ROOT, "results", "throughput.json"),
                        help="Where to write the JSON.")
    parser.add_argument("--work_dir", default="",
                        help="Scratch directory for the throwaway training runs. Defaults to a "
                             "sibling of --out.")
    return parser.parse_args()


def time_environment(args: argparse.Namespace) -> dict:
    """Times one process stepping the environment with no policy attached.

    Actions are sampled uniformly rather than by a policy, because the quantity wanted is the cost
    of a physics tick plus the observation and reward work, and a uniform action reaches the same
    code as any other.

    Args:
        args: Parsed arguments.

    Returns:
        The step count, elapsed seconds and steps per second.
    """
    env = BljEnv(BljConfig(rom_path=args.rom))
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    count = int(env.action_space.n)

    for _ in range(args.env_warmup):
        _, _, terminated, truncated, _ = env.step(int(rng.integers(count)))
        if terminated or truncated:
            env.reset(seed=args.seed)

    env.reset(seed=args.seed + 1)
    started = time.perf_counter()
    for _ in range(args.env_steps):
        _, _, terminated, truncated, _ = env.step(int(rng.integers(count)))
        if terminated or truncated:
            env.reset(seed=args.seed + 1)
    elapsed = time.perf_counter() - started
    return {"steps": args.env_steps, "seconds": round(elapsed, 3),
            "stepsPerSecond": round(args.env_steps / elapsed, 1)}


def time_loop(args: argparse.Namespace) -> dict:
    """Times the training loop at several lengths and fits out the startup cost.

    Args:
        args: Parsed arguments.

    Returns:
        Each timing, the fitted marginal rate and fixed startup, and the largest residual the fit
        leaves on the timings it was given.
    """
    work = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "throughput")
    lengths = sorted(int(value) for value in args.loop_steps.split(","))
    runs = []
    for index, total in enumerate(lengths):
        started = time.perf_counter()
        ppo.train(rung_name=args.rung, seed=args.seed, total_timesteps=total,
                  out_dir=os.path.join(work, f"run{index}"), rom_path=args.rom,
                  collision_path=endless_stairs._DEFAULT_COLLISION,
                  checkpoint_every=10_000_000)
        elapsed = time.perf_counter() - started
        runs.append({"steps": total, "seconds": round(elapsed, 3),
                     "stepsPerSecond": round(total / elapsed, 1)})

    # Slope through the shortest and longest run, so the fixed cost cancels. Any run in between
    # is then a test of the straight line rather than part of it, which is what the residual
    # reports; with only two timings it is zero whatever the truth is.
    first, last = runs[0], runs[-1]
    rate = (last["steps"] - first["steps"]) / (last["seconds"] - first["seconds"])
    startup = first["seconds"] - first["steps"] / rate
    checked = runs[1:-1]
    residual = (max(abs(run["seconds"] - (startup + run["steps"] / rate)) for run in checked)
                if checked else None)
    return {"runs": runs, "stepsPerSecond": round(rate, 1),
            "startupSeconds": round(startup, 2),
            "fitResidualSeconds": None if residual is None else round(residual, 3),
            "fitCheckedRuns": len(checked)}


def main() -> None:
    """Writes the throughput summary."""
    args = parse_args()
    environment = time_environment(args)
    loop = time_loop(args)
    summary = {
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "cores": os.cpu_count()},
        "trainEnvs": 8,
        "environment": environment,
        "loop": loop,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.out}")
    print(f"  one process          {environment['stepsPerSecond']:>9,.0f} steps/s")
    for run in loop["runs"]:
        print(f"  loop, {run['steps']:>7,} steps {run['stepsPerSecond']:>9,.0f} steps/s "
              f"({run['seconds']:.1f}s)")
    residual = loop["fitResidualSeconds"]
    check = "unchecked, only two timings" if residual is None else f"residual {residual:.2f}s"
    print(f"  loop, fitted         {loop['stepsPerSecond']:>9,.0f} steps/s "
          f"plus {loop['startupSeconds']:.1f}s of startup, {check}")


if __name__ == "__main__":
    main()
