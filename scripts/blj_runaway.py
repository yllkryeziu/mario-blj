"""Asks which geometries let the speed chain run away, and by how much.

A backwards long jump chain either compounds or it does not, and which one happens is decided by
what Mario lands on. This sweep drives the same scripted policy across flat ground, four ramp
angles and ten staircases, at five air stick magnitudes by default, and records the peak speed of
every combination. A run counts as a runaway past -200, well outside anything ordinary movement
reaches.

The air stick magnitude is swept alongside the geometry because it is the one continuous knob the
policy has: 0 holds no direction in the air, 1 is full normalized deflection. ``envelope_degrees``
folds each scene down to the one number they can be compared on, the slope the treads describe,
which is what makes a staircase and a ramp comparable at all.
"""

import argparse
import json
import math
import os

from src.agent.scripted import ScriptedBlj, calibrate_stick, run_chain, stick_toward
from src.env.geometry import flat_area, ground_plane, ramp, staircase
from src.env.native import Sm64, action_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

RUNAWAY = -200.0


APRON = flat_area(-3000.0, 3000.0, -12000.0, 0.0)
SPAWN = (0.0, 100.0, -2000.0)


def geometries() -> dict:
    """Builds every scene in the sweep.

    The staircases vary rise and run separately rather than along one angle, because two flights
    of the same gradient built from different step sizes are not the same problem for a landing.
    Every scene sits on the same wide apron so that a run cannot end by walking off an edge.

    Returns:
        A mapping from scene label to its surfaces and spawn. The labels encode the parameters,
        which is what ``envelope_degrees`` reads back.
    """
    cases = {
        "flat": (APRON, SPAWN),
        "ramp_20deg": (APRON + ramp(24000.0, 6000.0, 20.0), SPAWN),
        "ramp_30deg": (APRON + ramp(24000.0, 6000.0, 30.0), SPAWN),
        "ramp_37deg": (APRON + ramp(24000.0, 6000.0, 37.0), SPAWN),
        "ramp_40deg": (APRON + ramp(24000.0, 6000.0, 40.0), SPAWN),
    }
    for rise, run in ((50.0, 150.0), (50.0, 100.0), (50.0, 60.0), (75.0, 150.0),
                      (75.0, 100.0), (75.0, 60.0), (100.0, 150.0), (100.0, 100.0),
                      (100.0, 60.0), (150.0, 100.0)):
        cases[f"stairs_rise{rise:.0f}_run{run:.0f}"] = (
            APRON + staircase(400, rise, run, 6000.0), SPAWN)
    return cases


def envelope_degrees(label: str) -> float | None:
    """Recovers the slope a scene label describes, so scenes can be ordered by steepness.

    Args:
        label: A label from :func:`geometries`.

    Returns:
        The ramp's angle, or the angle of the staircase's treads, or None for flat ground, which
        has no envelope rather than an envelope of zero.
    """
    if label.startswith("ramp"):
        return float(label.split("_")[1].removesuffix("deg"))
    if not label.startswith("stairs"):
        return None
    rise = float(label.split("rise")[1].split("_")[0])
    run = float(label.split("run")[1])
    return round(math.degrees(math.atan2(rise, run)), 1)


def summarize(label: str, magnitude: float, result: dict) -> dict:
    """Reduces one run to the row that goes in the report.

    Each cycle carries the air time of the jump before it, so the first one's is always zero and
    is dropped from the air frame statistics rather than averaged in.

    Args:
        label: Scene label.
        magnitude: Air stick magnitude the run used.
        result: The dict ``src.agent.scripted.run_chain`` returned.

    Returns:
        One flat row: the scene, its envelope, the cycle counts, the peak and best launch speeds,
        the air frame statistics, whether it ran away, and where the chain broke if it did.
    """
    cycles = result["cycles"]
    backwards = [c for c in cycles if c["launch_velocity"] < 0.0]
    air = [c["air_frames"] for c in cycles[1:]]
    left = result["left_loop"]
    return {
        "geometry": label,
        "envelope_degrees": envelope_degrees(label),
        "air_stick_magnitude": magnitude,
        "cycles": len(cycles),
        "backwards_cycles": len(backwards),
        "peak_velocity": result["peak_velocity"],
        "best_launch_velocity": min((c["launch_velocity"] for c in cycles), default=0.0),
        "min_air_frames": min(air) if air else None,
        "mean_air_frames": round(sum(air) / len(air), 2) if air else None,
        "runaway": result["peak_velocity"] < RUNAWAY,
        "left_loop": ({"frame": left["frame"], "action": action_name(left["action"]),
                       "forward_velocity": left["forward_velocity"]} if left else None),
    }


def main() -> None:
    """Runs the whole sweep on one libsm64 handle and writes the report.

    All of it shares one handle and one Mario, which is safe because every run reloads its own
    surfaces and respawns; it also means the stick calibration at the start applies to all of them.

    Raises:
        OSError: If the ROM, the shared library or the output path cannot be opened.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default=os.path.join(ROOT, "roms", "baserom.us.z64"))
    parser.add_argument("--library", default=None)
    parser.add_argument("--frames", type=int, default=1600)
    parser.add_argument("--magnitudes", type=float, nargs="*",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--out", default=os.path.join(RESULTS, "blj_runaway.json"))
    args = parser.parse_args()

    game = Sm64(args.rom, args.library)
    try:
        table = calibrate_stick(game, ground_plane(8000.0), (0.0, 100.0, 0.0))
        approach = stick_toward(table, 0.0, -1.0)
        print("stick calibration, 25 frames of full deflection from rest")
        for key, (dx, dz, face) in table.items():
            print(f"  {key:>5}  dx {dx:8.1f}  dz {dz:8.1f}  faceAngle {face:7.4f}")
        print(f"  approach stick '{approach}' walks away from the rise; "
              f"the air stick is its opposite\n")

        rows = []
        for label, (surfaces, spawn) in geometries().items():
            for magnitude in args.magnitudes:
                policy = ScriptedBlj(approach, air_magnitude=magnitude)
                result = run_chain(game, surfaces, spawn, policy, args.frames)
                row = summarize(label, magnitude, result)
                rows.append(row)
                print(f"{label:>22}  stick {magnitude:4.2f}  "
                      f"cycles {row['cycles']:3d}  best launch {row['best_launch_velocity']:9.2f}  "
                      f"peak {row['peak_velocity']:10.2f}  "
                      f"air frames min {str(row['min_air_frames']):>4}  "
                      f"{'RUNAWAY' if row['runaway'] else ''}")
    finally:
        game.close()

    runaways = [r for r in rows if r["runaway"]]
    print(f"\n{len(runaways)} of {len(rows)} configurations ran away past {RUNAWAY:.0f}")
    if runaways:
        best = min(runaways, key=lambda r: r["peak_velocity"])
        print(f"strongest: {best['geometry']} at stick {best['air_stick_magnitude']}, "
              f"peak forwardVel {best['peak_velocity']}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"calibration": {k: [round(v, 2) for v in row]
                                   for k, row in table.items()},
                   "approach_stick": approach, "runs": rows}, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
