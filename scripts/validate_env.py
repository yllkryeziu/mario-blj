"""Drives the environment with the scripted expert, to prove the task is solvable at all.

The project rule is to validate against a known-good reference before sweeping parameters. The
scripted expert in ``src.agent.scripted`` is that reference: it walks away from the rise, crouch
slides, long jumps, then holds the stick back and re-presses A on every landing. If it cannot beat
the staircase's instant warp loop, no learned policy will, and the fault is the environment's.
"""

from __future__ import annotations

import json
import os

from absl import app, flags, logging

from src.agent.drivers import scripted_driver
from src.env.blj_env import BljConfig, BljEnv, RewardConfig
from src.env.endless_stairs import minimum_escape_speed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ROM = flags.DEFINE_string("rom", os.path.join(_ROOT, "roms", "baserom.us.z64"), "ROM path.")
_FRAMES = flags.DEFINE_integer("frames", 6000, "Episode frame limit.")
_APPROACH = flags.DEFINE_string("approach", "down", "Cardinal the expert walks toward.")
_MAGNITUDE = flags.DEFINE_float("air_magnitude", 1.0, "Stick magnitude held during the air phase.")
_OUT = flags.DEFINE_string("out", os.path.join(_ROOT, "results", "env_validation.json"),
                           "Where to write the summary.")


def run(config: BljConfig, approach: str, air_magnitude: float, frames: int) -> dict:
    """Runs one expert episode.

    Args:
        config: Environment configuration.
        approach: Cardinal the expert walks toward before reversing.
        air_magnitude: Stick magnitude held during the air phase.
        frames: Step limit.

    Returns:
        A summary of the episode.
    """
    env = BljEnv(config)
    driver = scripted_driver(approach, air_magnitude=air_magnitude)
    try:
        observation, info = env.reset(seed=0)
        total = 0.0
        stage = 0
        history = []
        for _ in range(frames):
            action = driver(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            stage = max(stage, info["curriculum_stage"])
            history.append((info["frames"], info["forward_velocity"], info["height"]))
            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "approach": approach,
        "air_magnitude": air_magnitude,
        "return": round(total, 4),
        "frames": info["frames"],
        "peak_backward_velocity": round(info["peak_backward_velocity"], 2),
        "curriculum_stage": stage,
        "warps": info["warps"],
        "final_height": round(info["height"], 1),
        "success": info["success"],
        "highest_reached": round(max(row[2] for row in history), 1),
    }


def main(argv: list[str]) -> None:
    """Runs the expert over both stick polarities and writes the summary."""
    del argv
    config = BljConfig(
        rom_path=_ROM.value,
        reward=RewardConfig(terminal=1.0, speed_coefficient=0.01, curriculum_bonus=0.25),
        max_frames=_FRAMES.value)

    rows = []
    for approach in (_APPROACH.value, "up"):
        row = run(config, approach, _MAGNITUDE.value, _FRAMES.value)
        rows.append(row)
        logging.info("approach %s: peak %.2f stage %d warps %d high %.1f success %s",
                     approach, row["peak_backward_velocity"], row["curriculum_stage"],
                     row["warps"], row["highest_reached"], row["success"])
        print(f"approach {approach:>5}  peak |v| {row['peak_backward_velocity']:9.2f}  "
              f"stage {row['curriculum_stage']}  warps {row['warps']:3d}  "
              f"highest y {row['highest_reached']:8.1f}  success {row['success']}")

    escape = minimum_escape_speed(BljEnv(config)._scene.warp)
    print(f"\nwarp zone needs |forwardVel| > {escape:.0f} to be skipped")

    os.makedirs(os.path.dirname(_OUT.value), exist_ok=True)
    with open(_OUT.value, "w", encoding="utf-8") as handle:
        json.dump({"minimum_escape_speed": escape, "runs": rows}, handle, indent=2)
    print(f"wrote {_OUT.value}")


if __name__ == "__main__":
    app.run(main)
