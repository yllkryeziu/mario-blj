"""The reward shaping ablation ladder for the backwards long jump.

The experiment asks how much reward shaping an agent needs before it can perform the
backwards long jump. Each rung adds exactly one ingredient on top of the rung below it, so a
success at rung n and a failure at rung n-1 localizes the credit to the single ingredient
that changed. The ladder itself is the finding, which means the rung definitions have to stay
readable and have to stay pure data.

Rung 0 pays only the terminal bonus for reaching the top landing. Rung 1 adds a dense reward
on backwards speed. Rung 2 adds a bonus for advancing through the action chain that a jump
requires. Rung 3 keeps rung 2's reward and sweeps the action repeat, which changes the
effective horizon rather than the reward.

This module imports no ctypes and no gymnasium. It is safe to import on a machine where
libsm64 has not been built, which matters because the cluster submit path reads the rung
table to size a job array before any native library exists. The configuration types from
``src.env.blj_env`` are imported lazily inside :func:`build_config` for the same reason.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.env.blj_env import BljConfig

DEFAULT_ACTION_REPEAT_SWEEP: tuple[int, ...] = (1, 2, 3, 4)


@dataclasses.dataclass(frozen=True)
class Rung:
    """One rung of the ablation ladder.

    The four reward fields mirror ``src.env.blj_env.RewardConfig`` one for one rather than
    holding a ``RewardConfig`` instance, which keeps this module free of any import that
    touches the native library. :func:`build_config` does the translation.

    Attributes:
        name: Stable identifier for the rung, used in output paths and Slurm array mapping.
        description: One line saying what this rung adds over the rung below it.
        terminal: Reward paid once for reaching the top landing.
        speed_weight: Total payable for backwards speed, saturating at the speed that defeats
            the loop.
        curriculum_weight: Total payable for advancing the hand written stage recipe.
        height_weight: Total payable for climbing the whole staircase on foot.
        time_penalty: Reward subtracted each step to discourage stalling.
        action_repeat: Environment frames each agent action is held for.
    """

    name: str
    description: str
    terminal: float = 1.0
    speed_weight: float = 0.0
    curriculum_weight: float = 0.0
    height_weight: float = 0.0
    time_penalty: float = 0.0
    action_repeat: int = 1


RUNG_TERMINAL = Rung(
    name="terminal",
    description="Terminal bonus only, no shaping of any kind.",
    terminal=1.0,
)

RUNG_SPEED = Rung(
    name="speed",
    description="Terminal bonus plus a dense reward on backwards speed.",
    terminal=1.0,
    speed_weight=0.25,
)

RUNG_CURRICULUM = Rung(
    name="curriculum",
    description="Speed shaping plus a bonus for advancing the action chain.",
    terminal=1.0,
    speed_weight=0.25,
    curriculum_weight=0.25,
)

RUNG_REPEAT = Rung(
    name="repeat",
    description="Curriculum shaping with the action repeat swept over 1, 2, 3, 4.",
    terminal=1.0,
    speed_weight=0.25,
    curriculum_weight=0.25,
    action_repeat=1,
)

RUNG_HEIGHT = Rung(
    name="height",
    description="Terminal bonus plus record height climbed on foot. States the goal only.",
    terminal=1.0,
    height_weight=0.25,
)

RUNG_HEIGHT_SPEED = Rung(
    name="height_speed",
    description="Height progress plus the backwards speed hint, but no action recipe.",
    terminal=1.0,
    height_weight=0.25,
    speed_weight=0.25,
)

RUNGS: dict[str, Rung] = {
    rung.name: rung
    for rung in (RUNG_TERMINAL, RUNG_SPEED, RUNG_HEIGHT, RUNG_HEIGHT_SPEED, RUNG_CURRICULUM,
                 RUNG_REPEAT)
}

LADDER_ORDER: tuple[str, ...] = ("terminal", "speed", "height", "height_speed", "curriculum",
                                "repeat")

SWEEP_RUNG: str = "repeat"


def get_rung(name: str) -> Rung:
    """Looks a rung up by name.

    Args:
        name: Rung name, one of the keys of :data:`RUNGS`. A name of the form
            ``repeat_r3`` produced by :func:`expand_action_repeat` also resolves, so a
            caller holding only an expanded run name can recover its rung.

    Returns:
        The matching rung.

    Raises:
        KeyError: If the name is not a known rung and does not parse as an expanded
            action repeat variant of one.
    """
    if name in RUNGS:
        return RUNGS[name]
    base, separator, suffix = name.rpartition("_r")
    if separator and base in RUNGS and suffix.isdigit():
        return dataclasses.replace(RUNGS[base], name=name, action_repeat=int(suffix))
    raise KeyError(f"unknown rung {name!r}, expected one of {sorted(RUNGS)}")


def expand_action_repeat(
    rung: Rung,
    repeats: tuple[int, ...] = DEFAULT_ACTION_REPEAT_SWEEP,
) -> tuple[Rung, ...]:
    """Expands a rung into one variant per action repeat value.

    Rung 3 is a sweep rather than a single configuration. Each variant is named by
    appending ``_r<repeat>`` so that runs land in distinct output directories, with the
    exception of a single element sweep, which is returned unrenamed because there is
    nothing to disambiguate.

    Args:
        rung: Rung to expand.
        repeats: Action repeat values to sweep. Must be non empty and every value must be
            a positive integer.

    Returns:
        One rung per requested action repeat, in the order given.

    Raises:
        ValueError: If ``repeats`` is empty, contains a duplicate, or contains a value
            that is not a positive integer.
    """
    if not repeats:
        raise ValueError("repeats must not be empty")
    if len(set(repeats)) != len(repeats):
        raise ValueError(f"repeats must not contain duplicates, got {repeats}")
    for repeat in repeats:
        if repeat < 1:
            raise ValueError(f"action repeat must be a positive integer, got {repeat}")
    if len(repeats) == 1:
        return (dataclasses.replace(rung, action_repeat=repeats[0]),)
    return tuple(
        dataclasses.replace(rung, name=f"{rung.name}_r{repeat}", action_repeat=repeat)
        for repeat in repeats
    )


def ladder_runs(
    names: tuple[str, ...] = LADDER_ORDER,
    repeats: tuple[int, ...] = DEFAULT_ACTION_REPEAT_SWEEP,
) -> tuple[Rung, ...]:
    """Expands a selection of rung names into the concrete rungs to train.

    Every name resolves to exactly one rung except the sweep rung, which resolves to one
    rung per action repeat value. Requesting the whole ladder therefore yields three fixed
    rungs followed by the sweep.

    Args:
        names: Rung names to include, in the order they should be trained.
        repeats: Action repeat values used when expanding the sweep rung.

    Returns:
        The concrete rungs to train, in order.

    Raises:
        KeyError: If a name is not a known rung.
        ValueError: If ``repeats`` is invalid for the sweep rung.
    """
    runs: list[Rung] = []
    for name in names:
        rung = get_rung(name)
        if name == SWEEP_RUNG:
            runs.extend(expand_action_repeat(rung, repeats))
        else:
            runs.append(rung)
    return tuple(runs)


def run_names(
    names: tuple[str, ...] = LADDER_ORDER,
    repeats: tuple[int, ...] = DEFAULT_ACTION_REPEAT_SWEEP,
) -> tuple[str, ...]:
    """Lists the run names the ladder expands to.

    The cluster submit path uses this to size a job array without importing anything that
    needs the native library.

    Args:
        names: Rung names to include.
        repeats: Action repeat values used when expanding the sweep rung.

    Returns:
        One name per run, in order.
    """
    return tuple(rung.name for rung in ladder_runs(names, repeats))


def job_matrix(
    seeds: tuple[int, ...],
    names: tuple[str, ...] = LADDER_ORDER,
    repeats: tuple[int, ...] = DEFAULT_ACTION_REPEAT_SWEEP,
) -> tuple[tuple[str, int], ...]:
    """Builds the flat (run name, seed) list that a Slurm array indexes into.

    Array task n trains element n of this list. Runs vary slowest and seeds fastest, so
    all seeds of one rung sit in a contiguous block of array indices.

    Args:
        seeds: Random seeds to train each run with.
        names: Rung names to include.
        repeats: Action repeat values used when expanding the sweep rung.

    Returns:
        One (run name, seed) pair per array task, in array index order.

    Raises:
        ValueError: If ``seeds`` is empty.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    return tuple(
        (rung.name, seed)
        for rung in ladder_runs(names, repeats)
        for seed in seeds
    )


def build_config(
    rung: Rung,
    rom_path: str,
    collision_path: str,
    max_frames: int = 3000,
    spawn_jitter: float = 0.0,
    library_path: str | None = None,
) -> BljConfig:
    """Turns a rung into the environment configuration that realizes it.

    The import of ``src.env.blj_env`` happens here rather than at module scope so that the
    rung table stays importable without a built libsm64.

    Args:
        rung: Rung to realize.
        rom_path: Path to the Super Mario 64 ROM that libsm64 reads assets from.
        collision_path: Path to the endless staircase ``collision.inc.c``.
        max_frames: Episode frame budget before truncation.
        spawn_jitter: Magnitude of random spawn displacement, in world units.
        library_path: Explicit path to the libsm64 shared library, or None to let the
            environment find it.

    Returns:
        A ``BljConfig`` carrying this rung's reward weights and action repeat.
    """
    from src.env.blj_env import BljConfig, RewardConfig

    reward = RewardConfig(
        terminal=rung.terminal,
        speed_weight=rung.speed_weight,
        curriculum_weight=rung.curriculum_weight,
        height_weight=rung.height_weight,
        time_penalty=rung.time_penalty,
    )
    return BljConfig(
        rom_path=rom_path,
        collision_path=collision_path,
        reward=reward,
        action_repeat=rung.action_repeat,
        max_frames=max_frames,
        spawn_jitter=spawn_jitter,
        library_path=library_path,
    )
