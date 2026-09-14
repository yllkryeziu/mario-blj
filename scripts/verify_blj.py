import argparse
import json
import os

from src.env.geometry import slope_course
from src.env.native import (ACT_CROUCH_SLIDE, ACT_LONG_JUMP, ACT_LONG_JUMP_LAND,
                            MarioInputs, Sm64, action_name)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_chain(game: Sm64, angle: float, repress_delay: int, stick_y: float,
              frames: int, spawn_height: float) -> dict:
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
