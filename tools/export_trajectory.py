"""Exports a recorded episode as a state trajectory the sm64-port build can render.

The environment runs libsm64, which returns geometry rather than pixels, so an episode can be
measured exactly but not screenshotted. sm64-port is the same decompilation with the game's real
renderer attached, and its Mario is driven by a controller rather than by a policy. This script
bridges the two: it replays a recorded episode through the environment, reads Mario's state back
each frame, and writes it as a flat binary the port's injector stamps into ``gMarioState`` after
physics has run. The port then draws the policy's own trajectory with the real castle, the real
Mario model and the real camera, and no open loop divergence is possible because every frame is
overwritten with the measured state instead of being re-simulated from inputs.

Run it as::

    PYTHONPATH=. python tools/export_trajectory.py --replay results/replay_model_endless.json \
        --out results/trajectory_model_endless.bin
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"MBLJTRJ1"
RECORD = struct.Struct("<3f3ff3i4I2i")
RECORD_SIZE = 64

_S16_PER_TURN = 65536.0


@dataclass(frozen=True)
class ExportConfig:
    """Resolved options for one export.

    Attributes:
        replay_path: Recorded episode to replay, as written by scripts/record_episode.py.
        out_path: Destination for the binary trajectory.
        rom_path: Base ROM libsm64 needs.
        collision_path: Collision file for the scene the episode was recorded in.
    """

    replay_path: Path
    out_path: Path
    rom_path: Path
    collision_path: Path


def radians_to_s16(radians: float) -> int:
    """Converts a yaw in radians back into the decompilation's s16 angle units.

    libsm64 converts Mario's ``faceAngle`` to radians on the way out, and the port wants the
    original units, so the conversion has to be undone exactly rather than approximately: one turn
    is 65536 units, and the result is wrapped into signed 16 bit range because that is how the
    game stores it.

    Args:
        radians: Yaw in radians.

    Returns:
        The same angle in s16 units, in [-32768, 32767].
    """
    units = int(round(radians * _S16_PER_TURN / (2.0 * math.pi))) & 0xFFFF
    return units - 0x10000 if units >= 0x8000 else units


def record_bytes(state: Any, extra: Any) -> bytes:
    """Packs one frame of Mario's state into the injector's 64 byte record.

    Args:
        state: The frame's MarioState, as the environment's game object reports it.
        extra: The matching MarioExtraState, which carries the action bookkeeping and the pitch
            and roll that MarioState leaves out.

    Returns:
        Exactly RECORD_SIZE bytes, little endian.
    """
    packed = RECORD.pack(
        state.position[0], state.position[1], state.position[2],
        state.velocity[0], state.velocity[1], state.velocity[2],
        state.forwardVelocity,
        extra.faceAnglePitch, radians_to_s16(state.faceAngle), extra.faceAngleRoll,
        state.action, extra.actionState & 0xFFFFFFFF, extra.actionTimer & 0xFFFFFFFF,
        extra.actionArg & 0xFFFFFFFF,
        state.animID, state.animFrame,
    )
    assert len(packed) == RECORD_SIZE, len(packed)
    return packed


def export(config: ExportConfig) -> dict[str, Any]:
    """Replays the recorded episode and writes the trajectory file.

    Args:
        config: Resolved export configuration.

    Returns:
        A summary dict with the frame count, the peak forward velocity and the y range covered.
    """
    from src.agent.drivers import action_index
    from src.env.blj_env import BljConfig, BljEnv, RewardConfig

    with open(config.replay_path, encoding="utf-8") as handle:
        replay = json.load(handle)
    queue = [action_index(row["inputs"]["stick_x"], row["inputs"]["stick_y"],
                          bool(row["inputs"]["a"]), bool(row["inputs"]["z"]))
             for row in replay["frames"]]
    logging.info("replaying %d recorded frames from %s", len(queue), config.replay_path)

    environment = BljEnv(BljConfig(
        rom_path=config.rom_path,
        collision_path=config.collision_path,
        reward=RewardConfig(terminal=1.0),
        max_frames=len(queue) + 1))
    records: list[bytes] = []
    peak = 0.0
    heights: list[float] = []
    try:
        environment.reset()
        game = environment.game
        for action in queue:
            environment.step(action)
            state = game.state
            extra = game.extra_state()
            records.append(record_bytes(state, extra))
            peak = min(peak, state.forwardVelocity)
            heights.append(state.position[1])
    finally:
        environment.close()

    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.out_path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<II", len(records), RECORD_SIZE))
        for record in records:
            handle.write(record)
    return {
        "frames": len(records),
        "peak_velocity": round(peak, 3),
        "y_range": [round(min(heights), 1), round(max(heights), 1)] if heights else [],
        "bytes": config.out_path.stat().st_size,
    }


def parse_args(argv: list[str] | None = None) -> ExportConfig:
    """Parses the command line.

    Args:
        argv: Argument list, or None to read sys.argv.

    Returns:
        The resolved configuration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="recorded episode to replay")
    parser.add_argument("--out", type=Path, required=True, help="destination trajectory file")
    parser.add_argument("--rom", type=Path, default=Path("roms/baserom.us.z64"))
    parser.add_argument("--collision", type=Path,
                        default=Path("third_party/sm64-port/levels/castle_inside/areas/2/"
                                     "collision.inc.c"))
    args = parser.parse_args(argv)
    return ExportConfig(replay_path=args.replay, out_path=args.out, rom_path=args.rom,
                        collision_path=args.collision)


def main() -> None:
    """Exports one trajectory and prints the summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = export(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
