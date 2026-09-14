"""Records one environment episode into the replay format the viewers read.

The driver is pluggable so the same recorder serves the scripted expert now and a trained policy
later. Watching the policy is the point of the exercise, so the recording carries the controller
state for every frame alongside Mario's own.
"""

from __future__ import annotations

import json
import os
from typing import Any

from absl import app, flags, logging

from src.agent.drivers import Driver, model_driver, scripted_driver
from src.env.blj_env import BljConfig, BljEnv, RewardConfig, decode_action
from src.env.native import ACT_LONG_JUMP

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ROM = flags.DEFINE_string("rom", os.path.join(_ROOT, "roms", "baserom.us.z64"), "ROM path.")
_FRAMES = flags.DEFINE_integer("frames", 2600, "Frame limit.")
_APPROACH = flags.DEFINE_string("approach", "down", "Cardinal the expert walks toward.")
_MODEL = flags.DEFINE_string("model", None, "Optional stable-baselines3 model zip to drive with.")
_NAME = flags.DEFINE_string("name", "endless_stairs", "Scene name in the replay payload.")
_OUT = flags.DEFINE_string("out", None, "Output path, defaults to results/replay_<name>.json.")
_STOCHASTIC = flags.DEFINE_boolean("stochastic", False,
                                   "Sample from the policy instead of taking the argmax action.")
_SEED = flags.DEFINE_integer("seed", 0, "Episode seed.")
_SEARCH = flags.DEFINE_integer("search", 1,
                               "Try this many seeds and keep the first successful episode.")


def record(config: BljConfig, driver: Driver, frames: int, name: str, seed: int = 0) -> dict:
    """Runs one episode and collects the replay payload.

    Args:
        config: Environment configuration.
        driver: Action source.
        frames: Frame limit.
        name: Scene name stored in the payload.
        seed: Episode seed, which fixes the spawn jitter and any sampling the driver does.

    Returns:
        The replay payload, ready to serialize.
    """
    env = BljEnv(config)
    try:
        observation, info = env.reset(seed=seed)
        trace: list[dict[str, Any]] = []
        cycles: list[dict[str, Any]] = []
        peak = 0.0
        previous_action = 0
        air_frames = 0
        warps = 0

        for frame in range(frames):
            action = driver(observation, info)
            stick_x, stick_y, press_a, press_z = decode_action(action)
            observation, _, terminated, truncated, info = env.step(action)
            state = env.game.state
            mario_action = info["mario_action_id"]

            if mario_action == ACT_LONG_JUMP and previous_action != ACT_LONG_JUMP:
                cycles.append({"frame": frame,
                               "launch_velocity": round(info["forward_velocity"], 3),
                               "air_frames": air_frames})
                air_frames = 0
            air_frames = air_frames + 1 if mario_action == ACT_LONG_JUMP else 0
            peak = min(peak, info["forward_velocity"])

            trace.append({
                "frame": frame,
                "action": mario_action,
                "action_name": info["mario_action"],
                "position": [round(state.position[i], 2) for i in range(3)],
                "velocity": [round(state.velocity[i], 2) for i in range(3)],
                "forward_velocity": round(info["forward_velocity"], 3),
                "face_angle": round(state.faceAngle, 4),
                "health": int(state.health),
                "stage": str(info["curriculum_stage"]),
                "warped": info["warps"] > warps,
                "inputs": {"stick_x": round(stick_x, 3), "stick_y": round(stick_y, 3),
                           "a": int(press_a), "b": 0, "z": int(press_z)},
            })
            warps = info["warps"]
            previous_action = mario_action
            if terminated or truncated:
                break

        scene = env.scene
        surfaces = [[[int(surface.vertices[corner][axis]) for axis in range(3)]
                     for corner in range(3)] for surface in scene.surfaces]
        mean_x = sum(row["position"][0] for row in trace) / max(1, len(trace))
        return {
            "scene": name,
            "air_magnitude": 1.0,
            "approach_stick": _APPROACH.value,
            "surfaces": surfaces,
            "profile_x": round(mean_x, 1),
            "peak_velocity": round(peak, 3),
            "success": info["success"],
            "warps": warps,
            "goal_y": scene.goal_y,
            "cycles": cycles,
            "frames": trace,
        }
    finally:
        env.close()


def main(argv: list[str]) -> None:
    """Records one episode and writes the replay JSON.

    Args:
        argv: Unparsed command line arguments, unused.

    Raises:
        ValueError: If --search asks for fewer than one episode, so nothing was recorded.
    """
    del argv
    config = BljConfig(
        rom_path=_ROM.value,
        reward=RewardConfig(terminal=1.0, speed_coefficient=0.01, curriculum_bonus=0.25),
        max_frames=_FRAMES.value)
    payload = None
    for attempt in range(_SEARCH.value):
        seed = _SEED.value + attempt
        driver = (model_driver(_MODEL.value, deterministic=not _STOCHASTIC.value, seed=seed)
                  if _MODEL.value else scripted_driver(_APPROACH.value))
        candidate = record(config, driver, _FRAMES.value, _NAME.value, seed=seed)
        logging.info("seed %d: peak %.2f warps %d success %s", seed,
                     candidate["peak_velocity"], candidate["warps"], candidate["success"])
        if payload is None or candidate["success"] and not payload["success"] or (
                not payload["success"] and candidate["peak_velocity"] < payload["peak_velocity"]):
            payload = candidate
        if payload["success"]:
            break

    if payload is None:
        raise ValueError(f"--search {_SEARCH.value} recorded no episodes, it must be at least 1")

    out = _OUT.value or os.path.join(_ROOT, "results", f"replay_{_NAME.value}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    logging.info("recorded %d frames, peak %.2f, success %s",
                 len(payload["frames"]), payload["peak_velocity"], payload["success"])
    print(f"{_NAME.value}: {len(payload['frames'])} frames, {len(payload['cycles'])} long jumps, "
          f"peak forwardVel {payload['peak_velocity']}, warps {payload['warps']}, "
          f"success {payload['success']}")
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    app.run(main)
