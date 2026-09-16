"""Captures populations of Marios into the container ``sm64-port``'s swarm renderer replays.

``tools/dump_swarm.py`` writes a self contained pose dictionary for a WebGL viewer, which is the
right shape for a page that has to draw the geometry itself. This writes the other kind of file: a
flat per frame record that the patched game reads back and draws with its own renderer, in the real
castle, with the real model, lighting and shadows.

The reason the file can be this small is that nothing about the pose needs to be described. libsm64
already reports which animation the graphics node is playing and which frame of it, in
``SM64MarioState.animID`` and ``animFrame``, and ``include/mario_animation_ids.h`` is byte identical
between libsm64 and sm64-port, so the id means the same clip on both sides. A Mario is therefore
twenty bytes: where he is, which way he faces, and which frame of which clip he is on.

Container layout, little endian:

    magic       8 bytes, b"MBLJSWR2"
    frames      uint32
    marios      uint32
    records     frames * marios records, frame major, of

        position    3 x float32, world units
        angleY      int16, the game's own angle units, which wrap
        animID      int16, an index into the shared animation table
        animFrame   int16, the frame of that clip the renderer should hold
        flags       int16, bit 0 set while this Mario should be drawn

Each capture also writes the population's own audio, because the sound is the game's: the mixer in
libsm64 is a file scope global rather than part of the per Mario state, so every slot's
``play_sound`` calls land in one request queue and one tick per frame renders the whole crowd.

One run per shot, one shot per (rung, checkpoint) pair, plus ``--random`` for a population that has
no policy at all and samples the action space uniformly, and ``--untrained`` for a population
driven by a freshly initialised PPO — the network the training runs start from, weights drawn from
the same seed as the landing-only run the post follows. The post opens on the untrained network.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.env import endless_stairs
from src.env.audio import AudioRecorder
from src.env.blj_env import ACTION_SIZE
from src.env.native import MarioState
from src.env.swarm import RewardConfig, Swarm, SwarmConfig

MAGIC = b"MBLJSWR2"
RECORD = struct.Struct("<fffhhhh")
FLAG_VISIBLE = 1
MAX_MARIOS = 64
"""Matches SWARM_MAX_MARIOS in third_party/sm64-port/src/game/swarm_render.c."""

# libsm64 converts Mario's s16 yaw to radians with this constant rather than with pi, so dividing
# by the same one takes the angle back to the integer the game started with instead of to a
# neighbouring one.
LIBSM64_PI = 3.14159


@dataclasses.dataclass(frozen=True)
class Shot:
    """One population to capture.

    Attributes:
        name: Output stem, unique within the run.
        rung: Reward shaping rung, or "random" for the untrained population.
        steps: Training steps behind the policy, or None when there is no policy.
        path: Checkpoint to load, or None for uniform random actions.
    """

    name: str
    rung: str
    steps: int | None
    path: str | None


def parse_args() -> argparse.Namespace:
    """Builds the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rung", action="append", default=[], metavar="NAME=DIR",
                        help="A rung and the checkpoint directory holding its ppo_*_steps.zip "
                             "files. Repeatable, and the order is the order of the shots.")
    parser.add_argument("--steps", default="1000000,5000000,10000000,20000000",
                        help="Comma separated training step counts to capture per rung. Each is "
                             "matched to the nearest checkpoint the directory actually holds.")
    parser.add_argument("--random", action="store_true",
                        help="Also capture a population with no policy, sampling actions "
                             "uniformly.")
    parser.add_argument("--random_name", default="random", help="Output stem for --random.")
    parser.add_argument("--untrained", action="store_true",
                        help="Also capture a population driven by a freshly initialised PPO: the "
                             "network the training runs start from. This is the shot the post "
                             "opens on.")
    parser.add_argument("--untrained_name", default="untrained",
                        help="Output stem for --untrained.")
    parser.add_argument("--untrained_seed", type=int, default=2,
                        help="Weight initialisation seed for --untrained. The default matches "
                             "the landing-only run the post follows, so the cold open is that "
                             "run's own starting network.")
    parser.add_argument("--population", type=int, default=MAX_MARIOS,
                        help=f"Marios per shot, at most {MAX_MARIOS}.")
    parser.add_argument("--frames", type=int, default=450,
                        help="Frames recorded per shot, at 30 per second.")
    parser.add_argument("--warmup", type=int, default=45,
                        help="Frames simulated before recording starts, so a shot does not open "
                             "on sixty four Marios in one identical spawn pose.")
    parser.add_argument("--episode_frames", type=int, default=1200,
                        help="Frames an episode may last before it truncates and respawns.")
    parser.add_argument("--spawn_spread", type=float, default=150.0,
                        help="Lateral spawn jitter in world units, across the corridor.")
    parser.add_argument("--seed", type=int, default=20250916,
                        help="Seeds the spawn jitter, the policy sampling and the random actions.")
    parser.add_argument("--stochastic", action="store_true", default=True,
                        help="Sample from the policy rather than taking its argmax action.")
    parser.add_argument("--deterministic", dest="stochastic", action="store_false",
                        help="Take the policy's argmax action.")
    parser.add_argument("--audio_volume", type=float, default=0.4,
                        help="Master volume for the capture. A population sums into one mixer and "
                             "clips at full scale, so a crowd wants less than one Mario does.")
    parser.add_argument("--no_audio", dest="audio", action="store_false", default=True,
                        help="Skip the audio capture.")
    parser.add_argument("--rom", default=os.path.join("roms", "baserom.us.z64"),
                        help="Path to the Super Mario 64 US ROM.")
    parser.add_argument("--collision", default=endless_stairs._DEFAULT_COLLISION,
                        help="Path to the staircase's collision.inc.c.")
    parser.add_argument("--header", default=endless_stairs._DEFAULT_HEADER,
                        help="Path to surface_terrains.h.")
    parser.add_argument("--out_dir", required=True, help="Directory to write the shots into.")
    return parser.parse_args()


def step_count(path: str) -> int | None:
    """Reads the training step count out of a checkpoint filename.

    Args:
        path: Path to a ``ppo_<steps>_steps.zip``.

    Returns:
        The step count, or None when the name does not carry one.
    """
    match = re.search(r"ppo_(\d+)_steps", os.path.basename(path))
    return int(match.group(1)) if match else None


def label_steps(steps: int | None) -> str:
    """Renders a step count the way a shot's filename and the post's caption should read.

    Args:
        steps: Training steps, or None.

    Returns:
        A short label such as "20M", or "untrained".
    """
    if steps is None:
        return "untrained"
    millions = steps / 1e6
    return f"{millions:.0f}M" if abs(millions - round(millions)) < 0.05 else f"{millions:.1f}M"


def resolve_shots(args: argparse.Namespace) -> list[Shot]:
    """Turns the rung and step flags into the list of populations to capture.

    Each wanted step count is matched to the nearest checkpoint on disk rather than required
    exactly, because a ladder saves on a stride that rarely lands on a round number: the run that
    produced these saves every 499,980 steps, so "20M" is really 19,999,200.

    Args:
        args: Parsed arguments.

    Returns:
        The shots, in flag order, with the random population first when it was asked for.

    Raises:
        ValueError: If a rung entry is malformed or its directory holds no checkpoints.
    """
    shots: list[Shot] = []
    if args.random:
        shots.append(Shot(name=args.random_name, rung="random", steps=None, path=None))
    if args.untrained:
        shots.append(Shot(name=args.untrained_name, rung="untrained", steps=None, path=None))

    wanted = [int(float(entry)) for entry in args.steps.split(",") if entry.strip()]
    for entry in args.rung:
        name, separator, directory = entry.partition("=")
        if not separator:
            raise ValueError(f"--rung wants NAME=DIR, got {entry!r}")
        found = {step_count(path): os.path.join(directory, path)
                 for path in sorted(os.listdir(directory)) if step_count(path) is not None}
        if not found:
            raise ValueError(f"no ppo_*_steps.zip checkpoints in {directory}")
        for target in wanted:
            steps = min(found, key=lambda held: abs(held - target))
            shots.append(Shot(name=f"{name}-{label_steps(target)}", rung=name, steps=steps,
                              path=found[steps]))
    if not shots:
        raise ValueError("nothing to capture: pass --rung and/or --random")
    return shots


def quantise_angle(radians: float) -> int:
    """Converts a face angle back into the game's own wrapping s16 angle unit.

    Args:
        radians: Face angle as libsm64 reports it.

    Returns:
        The angle as a signed 16 bit integer.
    """
    steps = int(round(radians / LIBSM64_PI * 32768.0)) & 0xFFFF
    return steps - 0x10000 if steps >= 0x8000 else steps


def slot_states(swarm: Swarm) -> list[MarioState]:
    """Reaches for the per slot state the swarm ticks into.

    The swarm's public surface is observations, bookkeeping and mesh views, and none of those carry
    the animation key. This is the same read only reach through ``tools/dump_swarm.py`` makes for
    the pose dictionary.

    Args:
        swarm: A live swarm.

    Returns:
        One state per slot, in slot order, aliasing the structs libsm64 writes into.
    """
    return [slot.state for slot in swarm._slots]


def capture(args: argparse.Namespace, shot: Shot, order: int) -> dict:
    """Runs one population and writes its container and its audio.

    Args:
        args: Parsed arguments.
        shot: The population to capture.
        order: Index of this shot in the run, used to offset the seeds so two shots of the same
            rung do not get identical spawn jitter.

    Returns:
        A manifest row describing what was written and what the population did.
    """
    rng = np.random.default_rng(args.seed + order)
    model = None
    if shot.path is not None:
        from stable_baselines3 import PPO

        model = PPO.load(shot.path, device="cpu")
        model.set_random_seed(args.seed + order)
    elif shot.rung == "untrained":
        # A freshly initialised PPO rather than a loaded checkpoint: the network the training
        # runs start from. Only the observation and action spaces are needed to build it, but
        # those come from a live BljEnv, the same construction scripts/action_occupancy.py uses
        # for its untrained reference policy. The env closes again once the model exists;
        # predict never touches it.
        from stable_baselines3.common.vec_env import DummyVecEnv

        from src.env.blj_env import BljEnv
        from src.train.ladder import build_config, get_rung
        from src.train.ppo import build_model

        env = BljEnv(build_config(get_rung("terminal"), args.rom, args.collision))
        try:
            model = build_model(DummyVecEnv([lambda: env]), seed=args.untrained_seed)
        finally:
            env.close()
        # The weights stay the seed's draw; the sampling stream follows the same protocol as the
        # checkpoint shots, so the only difference between this shot and a 1M panel is training.
        model.set_random_seed(args.seed + order)

    swarm = Swarm(SwarmConfig(
        rom_path=args.rom,
        collision_path=args.collision,
        header_path=args.header,
        population=args.population,
        max_frames=args.episode_frames,
        reward=RewardConfig(terminal=1.0),
        spawn_spread=args.spawn_spread,
        seed=args.seed + order))
    recorder = AudioRecorder(swarm.game, volume=args.audio_volume) if args.audio else None
    actions = np.zeros(args.population, dtype=np.int64)
    records = bytearray()
    resets = 0
    successes = 0
    heights: list[float] = []
    started = time.perf_counter()

    try:
        states = slot_states(swarm)
        previous = swarm.members()
        for frame in range(args.warmup + args.frames):
            if model is None:
                actions = rng.integers(0, ACTION_SIZE, size=args.population)
            else:
                actions, _ = model.predict(swarm.observations(),
                                           deterministic=not args.stochastic)
            members = swarm.step(actions)
            recording = frame >= args.warmup
            if recorder is not None and recording:
                recorder.capture()
            for member, earlier in zip(members, previous, strict=True):
                if member.episodes > earlier.episodes:
                    resets += 1
                    successes += 1 if member.success else 0
                    heights.append(member.best_height)
            previous = members
            if not recording:
                continue
            for state in states:
                # libsm64 reports animID -1 for a Mario who has been created but not yet posed,
                # which is every Mario on the frame his episode restarts. There is no pose to draw
                # on those frames, so the record says so rather than leaving the renderer to invent
                # one, and it is written through unmasked so the renderer sees the same -1.
                anim = int(state.animID)
                records += RECORD.pack(
                    float(state.position[0]), float(state.position[1]), float(state.position[2]),
                    quantise_angle(float(state.faceAngle)),
                    anim, int(state.animFrame), FLAG_VISIBLE if anim >= 0 else 0)
        running = swarm.members()
    finally:
        swarm.close()

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{shot.name}.bin")
    with open(path, "wb") as handle:
        handle.write(MAGIC + struct.pack("<II", args.frames, args.population))
        handle.write(records)

    audio_path = None
    if recorder is not None and recorder.frames:
        audio_path = recorder.write(os.path.join(args.out_dir, f"{shot.name}.mp3"))

    row = {
        "name": shot.name,
        "rung": shot.rung,
        "steps": shot.steps,
        "stepsLabel": label_steps(shot.steps),
        "checkpoint": shot.path,
        "population": args.population,
        "frames": args.frames,
        "seconds": round(args.frames / 30.0, 3),
        "container": os.path.basename(path),
        "bytes": os.path.getsize(path),
        "audio": os.path.basename(audio_path) if audio_path else None,
        "episodesFinished": resets,
        "successes": successes,
        "meanBestHeight": round(float(np.mean(heights)), 1) if heights else None,
        "bestHeightInShot": round(max([member.best_height for member in running]
                                      + heights, default=0.0), 1),
        "peakBackwardInShot": round(min(member.peak_backward for member in running), 1),
        "captureSeconds": round(time.perf_counter() - started, 1),
    }
    print(f"  {shot.name:<20} {row['bytes']:>8} B  {resets:>3} episodes  {successes:>3} solved  "
          f"peak back {row['peakBackwardInShot']:>9}  {row['captureSeconds']:>5.1f}s")
    return row


def main() -> None:
    """Captures every shot and writes the manifest beside them."""
    args = parse_args()
    if not 1 <= args.population <= MAX_MARIOS:
        raise SystemExit(f"--population must be between 1 and {MAX_MARIOS}")
    shots = resolve_shots(args)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{len(shots)} shots, {args.population} Marios, {args.frames} frames "
          f"({args.frames / 30.0:.1f} s) each")
    rows = [capture(args, shot, order) for order, shot in enumerate(shots)]

    manifest = {
        "population": args.population,
        "frames": args.frames,
        "frameRate": 30.0,
        "warmup": args.warmup,
        "episodeFrames": args.episode_frames,
        "spawnSpread": args.spawn_spread,
        "seed": args.seed,
        "stochastic": args.stochastic,
        "collision": args.collision,
        "shots": rows,
    }
    out = os.path.join(args.out_dir, "manifest.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
        handle.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
