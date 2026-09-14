"""Sweeps the long jump chain over slope angle, A repress delay and stick deflection.

This is the direct sweep, with no policy object in the way: the loop reads Mario's action from the
previous frame and decides what to press this frame, which is the smallest thing that can hold a
chain going. It answers what the chain needs from the geometry and from the timing, and
``results/blj_sweep.json`` is its output.

Two things to know before reading its numbers. It reports the peak absolute speed rather than the
peak backward speed, so a run that never reversed still shows a number. And ``--stick`` defaults
to the raw N64 scale of +-64, which libsm64 multiplies by another 64: those rows are in the over
deflected regime that ``src.env.native`` documents, so their peaks are not comparable with the
environment's own. Pass ``--stick -1 0 1`` for the normalized range the environment uses.
"""

import argparse
import json
import os

from src.env.geometry import slope_course
from src.env.native import (
    ACT_CROUCH_SLIDE,
    ACT_LONG_JUMP,
    ACT_LONG_JUMP_LAND,
    MarioInputs,
    Sm64,
    action_name,
)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_chain(game: Sm64, angle: float, repress_delay: int, stick_y: float,
              frames: int, spawn_height: float) -> dict:
    """Runs one configuration of the chain on one slope and summarizes it.

    The button logic is a reaction to the previous frame's action, not a schedule. Z is held from
    frame 30 onward so the crouch is available, A is pressed once a crouch slide exists, and after
    a landing it waits ``repress_delay`` frames before pressing again. That delay is the knob the
    sweep exists to turn: the relaunch needs a fresh press, and how soon it can come is what
    decides whether the chain compounds or dies.

    Args:
        game: Live libsm64 handle. The scene and Mario are both replaced.
        angle: Slope angle of the course in degrees.
        repress_delay: Frames to wait after a landing before pressing A again.
        stick_y: Stick y held for the whole run, on whatever scale the caller passes.
        frames: Frames to run.
        spawn_height: y to spawn Mario at.

    Returns:
        The configuration, the peak absolute forwardVel, the peak height, the number of long jumps
        and a trace sampled every fifth frame.
    """
    game.load_surfaces(slope_course(angle))
    game.create_mario(0.0, spawn_height, 200.0)

    inputs = MarioInputs()
    inputs.camLookX, inputs.camLookZ = 0.0, 1.0

    peak_speed = 0.0
    peak_height = -1e9
    long_jumps = 0
    frames_since_land = 0
    previous_action = 0
    trace = []

    for frame in range(frames):
        inputs.stickX = 0.0
        inputs.stickY = stick_y
        inputs.buttonA = 0
        inputs.buttonB = 0
        inputs.buttonZ = 0

        action = previous_action
        if frame < 30:
            inputs.buttonZ = 0
        elif action == ACT_LONG_JUMP_LAND:
            inputs.buttonZ = 1
            if frames_since_land >= repress_delay:
                inputs.buttonA = 1
        elif action == ACT_CROUCH_SLIDE:
            inputs.buttonZ = 1
            inputs.buttonA = 1
        else:
            inputs.buttonZ = 1

        state = game.tick(inputs)
        if state.action == ACT_LONG_JUMP and previous_action != ACT_LONG_JUMP:
            long_jumps += 1
        frames_since_land = frames_since_land + 1 if state.action == ACT_LONG_JUMP_LAND else 0
        previous_action = state.action

        peak_speed = max(peak_speed, abs(state.forwardVelocity))
        peak_height = max(peak_height, state.position[1])
        if frame % 5 == 0:
            trace.append({
                "frame": frame,
                "action": action_name(state.action),
                "forward_velocity": round(state.forwardVelocity, 2),
                "y": round(state.position[1], 1),
                "z": round(state.position[2], 1),
            })

    return {
        "angle": angle,
        "repress_delay": repress_delay,
        "stick_y": stick_y,
        "peak_abs_speed": round(peak_speed, 2),
        "peak_height": round(peak_height, 1),
        "long_jumps": long_jumps,
        "trace": trace,
    }


def main() -> None:
    """Runs the full sweep, prints one line per configuration and writes the JSON.

    Raises:
        OSError: If the ROM, the shared library or the output path cannot be opened.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "roms", "baserom.us.z64"))
    parser.add_argument("--library", default=None)
    parser.add_argument("--angles", type=float, nargs="*", default=[10.0, 20.0, 30.0, 40.0])
    parser.add_argument("--delays", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--stick", type=float, nargs="*", default=[-64.0, 0.0, 64.0])
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--spawn-height", type=float, default=100.0)
    parser.add_argument("--out", default=os.path.join(RESULTS, "blj_sweep.json"))
    args = parser.parse_args()

    game = Sm64(args.rom, args.library)
    runs = []
    try:
        for angle in args.angles:
            for delay in args.delays:
                for stick_y in args.stick:
                    row = run_chain(game, angle, delay, stick_y, args.frames, args.spawn_height)
                    runs.append(row)
                    print(f"angle {angle:5.1f}  delay {delay}  stickY {stick_y:+6.1f}  "
                          f"jumps {row['long_jumps']:3d}  peak|v| {row['peak_abs_speed']:8.2f}  "
                          f"peak y {row['peak_height']:8.1f}")
    finally:
        game.close()

    best = max(runs, key=lambda r: r["peak_abs_speed"])
    print(f"\nbest: angle {best['angle']}, delay {best['repress_delay']}, "
          f"stickY {best['stick_y']}, peak |v| {best['peak_abs_speed']}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"runs": runs, "best": best}, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
