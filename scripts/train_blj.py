#!/usr/bin/env python3
"""Runs one or more rungs of the backwards long jump reward shaping ladder.

A single invocation trains every requested (rung, seed) pair in sequence and prints a
summary table at the end. On the cluster each Slurm array task calls this with exactly one
rung and one seed, so the sequencing here is for local runs and for small sweeps.

Typical local smoke run, one rung and one seed on two environments:

    python3 scripts/train_blj.py --rung=terminal --seeds=0 --timesteps=20000 --num_envs=2

The whole ladder at three seeds, which is what the finding is built from:

    python3 scripts/train_blj.py --rung=all --seeds=0,1,2 --timesteps=2000000

``--rung=all`` expands to the four rungs with rung 3 swept over ``--action_repeats``, so the
default is seven runs per seed.
"""

import pathlib
import sys

from absl import app, flags, logging

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.train import ladder, ppo

_RUNG = flags.DEFINE_multi_string(
    "rung",
    ["all"],
    "Rung to train. Repeatable. 'all' expands to the whole ladder. An explicit action "
    f"repeat variant such as repeat_r3 is also accepted. Known rungs: {sorted(ladder.RUNGS)}.",
)
_SEEDS = flags.DEFINE_list("seeds", ["0"], "Comma separated random seeds to train each rung at.")
_TIMESTEPS = flags.DEFINE_integer(
    "timesteps", 2_000_000, "Agent steps per run.", lower_bound=1
)
_NUM_ENVS = flags.DEFINE_integer(
    "num_envs", 8, "Parallel environments, one libsm64 process each.", lower_bound=1
)
_OUT_DIR = flags.DEFINE_string("out_dir", "results/ladder", "Root directory for run outputs.")
_ROM = flags.DEFINE_string("rom", "roms/baserom.us.z64", "Super Mario 64 US ROM path.")
_COLLISION = flags.DEFINE_string(
    "collision",
    "third_party/sm64-port/levels/castle_inside/areas/2/collision.inc.c",
    "Endless staircase collision.inc.c path.",
)
_ACTION_REPEATS = flags.DEFINE_list(
    "action_repeats",
    [str(value) for value in ladder.DEFAULT_ACTION_REPEAT_SWEEP],
    "Action repeat values swept when the sweep rung is expanded.",
)
_MAX_FRAMES = flags.DEFINE_integer(
    "max_frames", 3000, "Episode frame budget before truncation.", lower_bound=1
)
_SPAWN_JITTER = flags.DEFINE_float(
    "spawn_jitter", 0.0, "Magnitude of random spawn displacement in world units."
)
_CHECKPOINT_EVERY = flags.DEFINE_integer(
    "checkpoint_every",
    100_000,
    "Agent steps between checkpoints. Zero disables checkpointing.",
    lower_bound=0,
)
_LIBRARY = flags.DEFINE_string(
    "library", None, "Explicit libsm64 shared library path. Unset lets the environment find it."
)

SUMMARY_COLUMNS: tuple[tuple[str, int], ...] = (
    ("rung", 12),
    ("seed", 5),
    ("repeat", 7),
    ("episodes", 9),
    ("success", 8),
    ("recent", 7),
    ("peak_vel", 9),
    ("warps", 7),
    ("first_success_frames", 20),
)


def resolve_runs(names: list[str], repeats: tuple[int, ...]) -> tuple[ladder.Rung, ...]:
    """Turns the requested rung names into the concrete rungs to train.

    Args:
        names: Rung names from the flag. The single entry ``all`` expands to the whole
            ladder. Other entries resolve individually, with the bare sweep rung expanded
            over ``repeats``.
        repeats: Action repeat values for the sweep rung.

    Returns:
        The rungs to train, in order, with duplicates removed.

    Raises:
        KeyError: If a name is not a known rung.
        ValueError: If ``repeats`` is invalid.
    """
    if names == ["all"]:
        return ladder.ladder_runs(ladder.LADDER_ORDER, repeats)
    resolved: list[ladder.Rung] = []
    seen: set[str] = set()
    for name in names:
        if name == "all":
            candidates = ladder.ladder_runs(ladder.LADDER_ORDER, repeats)
        elif name == ladder.SWEEP_RUNG:
            candidates = ladder.expand_action_repeat(ladder.get_rung(name), repeats)
        else:
            candidates = (ladder.get_rung(name),)
        for rung in candidates:
            if rung.name not in seen:
                seen.add(rung.name)
                resolved.append(rung)
    return tuple(resolved)


def format_summary(results: list[ppo.RunMetrics]) -> str:
    """Renders the per run metrics as a fixed width table.

    Args:
        results: Metrics in the order the runs finished.

    Returns:
        The table as a single string, header included, without a trailing newline.
    """
    header = "".join(name.ljust(width) for name, width in SUMMARY_COLUMNS)
    lines = [header, "-" * len(header)]
    for metrics in results:
        first = (
            str(metrics.frames_to_first_success)
            if metrics.frames_to_first_success is not None
            else "never"
        )
        cells = (
            metrics.rung,
            str(metrics.seed),
            str(metrics.action_repeat),
            str(metrics.episodes),
            f"{metrics.success_rate:.3f}",
            f"{metrics.recent_success_rate:.3f}",
            f"{metrics.peak_backward_velocity:.1f}",
            f"{metrics.mean_warps:.2f}",
            first,
        )
        lines.append(
            "".join(cell.ljust(width)
                    for cell, (_, width) in zip(cells, SUMMARY_COLUMNS, strict=True))
        )
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    """Trains the requested rungs and seeds, then prints the summary table.

    Args:
        argv: Positional arguments. None are accepted.

    Raises:
        app.UsageError: If positional arguments are given, if a rung name is unknown, if
            a seed or action repeat is not an integer, or if the ROM or collision file is
            missing.
    """
    if len(argv) > 1:
        raise app.UsageError(f"unexpected positional arguments: {argv[1:]}")

    try:
        seeds = tuple(int(value) for value in _SEEDS.value)
        repeats = tuple(int(value) for value in _ACTION_REPEATS.value)
    except ValueError as error:
        raise app.UsageError(f"seeds and action_repeats must be integers: {error}") from error

    try:
        runs = resolve_runs(list(_RUNG.value), repeats)
    except (KeyError, ValueError) as error:
        raise app.UsageError(str(error)) from error
    if not runs or not seeds:
        raise app.UsageError("nothing to train, check --rung and --seeds")

    for label, path in (("rom", _ROM.value), ("collision", _COLLISION.value)):
        if not pathlib.Path(path).exists():
            raise app.UsageError(f"--{label} does not exist: {path}")

    logging.info(
        "training %d runs at %d seeds for %d timesteps each: %s",
        len(runs),
        len(seeds),
        _TIMESTEPS.value,
        ", ".join(rung.name for rung in runs),
    )

    results: list[ppo.RunMetrics] = []
    failures: list[str] = []
    for rung in runs:
        for seed in seeds:
            try:
                metrics = ppo.train(
                    rung_name=rung.name,
                    seed=seed,
                    total_timesteps=_TIMESTEPS.value,
                    out_dir=_OUT_DIR.value,
                    rom_path=_ROM.value,
                    collision_path=_COLLISION.value,
                    num_envs=_NUM_ENVS.value,
                    max_frames=_MAX_FRAMES.value,
                    spawn_jitter=_SPAWN_JITTER.value,
                    library_path=_LIBRARY.value,
                    checkpoint_every=_CHECKPOINT_EVERY.value,
                    action_repeats=repeats,
                )
            except Exception:
                logging.exception("rung %s seed %d failed", rung.name, seed)
                failures.append(f"{rung.name} seed {seed}")
                continue
            results.append(metrics)

    if results:
        print()
        print(format_summary(results))
        print()
        print(f"artifacts under {pathlib.Path(_OUT_DIR.value).resolve()}")
    if failures:
        raise app.UsageError(f"{len(failures)} run(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    app.run(main)
