"""Dumps states along the scripted expert's trajectory, for restart based exploration.

Random exploration will not stumble into a forty cycle speed chain. Restarting episodes from
states the expert visited is the standard remedy, and measuring how much it buys is the top rung
of the ladder. This script produces that pool of states.
"""

from __future__ import annotations

import json
import os

from absl import app, flags, logging

from src.agent.drivers import scripted_driver
from src.env.blj_env import BljConfig, BljEnv, RewardConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ROM = flags.DEFINE_string("rom", os.path.join(_ROOT, "roms", "baserom.us.z64"), "ROM path.")
_FRAMES = flags.DEFINE_integer("frames", 2600, "Frame limit for the demonstration.")
_STRIDE = flags.DEFINE_integer("stride", 5, "Keep one state every this many frames.")
_APPROACH = flags.DEFINE_string("approach", "down", "Cardinal the expert walks toward.")
_OUT = flags.DEFINE_string("out", os.path.join(_ROOT, "results", "reset_states.json"),
                           "Where to write the state pool.")


def main(argv: list[str]) -> None:
    """Runs the expert and writes every stride-th state."""
    del argv
    config = BljConfig(rom_path=_ROM.value, max_frames=_FRAMES.value,
                       reward=RewardConfig(terminal=1.0))
    env = BljEnv(config)
    driver = scripted_driver(_APPROACH.value)
    states = []
    try:
        observation, info = env.reset(seed=0)
        for frame in range(_FRAMES.value):
            action = driver(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            if frame % _STRIDE.value == 0:
                state = env.game.state
                states.append({
                    "frame": frame,
                    "position": [round(state.position[i], 3) for i in range(3)],
                    "velocity": [round(state.velocity[i], 3) for i in range(3)],
                    "forward_velocity": round(state.forwardVelocity, 4),
                    "face_angle": round(state.faceAngle, 5),
                    "action": int(state.action),
                })
            if terminated or truncated:
                break
        success = info["success"]
    finally:
        env.close()

    os.makedirs(os.path.dirname(_OUT.value), exist_ok=True)
    with open(_OUT.value, "w", encoding="utf-8") as handle:
        json.dump({"success": success, "stride": _STRIDE.value, "states": states}, handle, indent=1)
    logging.info("wrote %d states, demonstration success %s", len(states), success)
    print(f"{len(states)} states from a demonstration that "
          f"{'reached' if success else 'did not reach'} the top, wrote {_OUT.value}")


if __name__ == "__main__":
    app.run(main)
