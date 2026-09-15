"""Asks whether the transfer staircases support the chain at all, before blaming the policies.

``scripts/transfer_test.py`` drops every trained policy on eight flights of stairs and reports
where they fail. A failure there has two possible causes and the script cannot tell them apart: the
policy may be unable to do on new geometry what it does on the castle, or the geometry may not
admit a runaway chain in the first place. The second is a real possibility rather than a caveat,
because the chain's growth condition depends on the geometry directly. Each launch multiplies
speed by 1.5 and each airborne frame pays 2.0 to air drag, so a cycle of k air frames maps v to
1.5 * (v - 2k), which grows only past |v| > 6k. Air frames are set by how soon Mario meets the next
surface going backwards up the flight, which is exactly what rise and run decide.

So this runs the project's own hand written expert, the one ``results/blj_runaway.json`` reports
on ramps and abstract staircases, on the eight scenes the policies were tested on. It uses
``src.agent.scripted.run_chain``, which bypasses the environment: no reward, no episode limit and
no instant warp, so the peak speed it reports is the chain against the geometry alone. Where the
expert runs away and the policies do not, the policies are what failed. Where neither does, the
staircase is.
"""

import argparse
import json
import os
import statistics

from src.agent.scripted import ScriptedBlj, calibrate_stick, run_chain, stick_toward
from src.env.endless_stairs import load_scene, minimum_escape_speed, synthetic_scene
from src.env.geometry import ground_plane
from src.env.native import Sm64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# The same eight scenes ``scripts/transfer_test.py`` uses, by the same labels, so the two reports
# join on one column.
SCENES: dict[str, tuple[float, float, bool] | None] = {
    "castle": None,
    "rebuilt_rise26_run51": (25.6, 51.25, True),
    "rise50_run51": (50.0, 51.25, True),
    "rise26_run100": (25.6, 100.0, True),
    "rise75_run100": (75.0, 100.0, True),
    "rise100_run100": (100.0, 100.0, True),
    "rise100_run100_norisers": (100.0, 100.0, False),
    "rise75_run100_norisers": (75.0, 100.0, False),
    "rise50_run51_norisers": (50.0, 51.25, False),
    "rebuilt_rise26_run51_norisers": (25.6, 51.25, False),
}

RUNAWAY = -200.0


def summarize(label: str, magnitude: float, escape: float, result: dict) -> dict:
    """Reduces one run to the row that goes in the report.

    Args:
        label: Scene label.
        magnitude: Air stick magnitude the run used.
        escape: Single frame displacement the scene's instant warp demands, for the escape column.
        result: The dict :func:`src.agent.scripted.run_chain` returned.

    Returns:
        The cycle counts, the peak speed, the air frames per cycle that explain it, and whether
        the peak would have cleared the warp band had the warp been active.
    """
    cycles = result["cycles"]
    air = [cycle["air_frames"] for cycle in cycles[1:]] or [0]
    launches = [cycle["launch_velocity"] for cycle in cycles]
    backwards = [value for value in launches if value < 0.0]
    return {
        "scene": label,
        "air_stick_magnitude": magnitude,
        "cycles": len(cycles),
        "backwards_cycles": len(backwards),
        "peak_velocity": result["peak_velocity"],
        "best_launch_velocity": round(min(launches, default=0.0), 3),
        "min_air_frames": min(air),
        "mean_air_frames": round(statistics.mean(air), 2),
        "runaway": result["peak_velocity"] < RUNAWAY,
        "escapes_warp": result["peak_velocity"] < -escape,
        "escape_speed": escape,
    }


def main() -> None:
    """Runs the expert on every transfer scene and writes the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default=os.path.join(ROOT, "roms", "baserom.us.z64"))
    parser.add_argument("--library", default=None)
    parser.add_argument("--frames", type=int, default=1600)
    parser.add_argument("--magnitudes", type=float, nargs="*",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument("--out", default=os.path.join(RESULTS, "transfer_expert.json"))
    args = parser.parse_args()

    scenes = {}
    for label in args.scenes:
        treads = SCENES[label]
        scenes[label] = (load_scene() if treads is None
                         else synthetic_scene(treads[0], treads[1], risers=treads[2]))

    game = Sm64(args.rom, args.library)
    rows = []
    try:
        # Calibration replaces the level, so it has to happen before any scene is loaded; every
        # run_chain call reloads its own surfaces afterwards.
        table = calibrate_stick(game, ground_plane(8000.0), (0.0, 100.0, 0.0))
        for label, scene in scenes.items():
            # The approach stick walks away from the rise, so the air stick, its opposite, drives
            # Mario backwards up the flight. Every scene here ascends toward negative z.
            approach = stick_toward(table, -scene.ascends_toward[0], -scene.ascends_toward[1])
            escape = minimum_escape_speed(scene.warp)
            for magnitude in args.magnitudes:
                policy = ScriptedBlj(approach, air_magnitude=magnitude)
                result = run_chain(game, scene.surfaces, scene.spawn, policy, args.frames)
                row = summarize(label, magnitude, escape, result)
                rows.append(row)
                print(f"{label:>24}  stick {magnitude:4.2f}  cycles {row['cycles']:3d}  "
                      f"peak {row['peak_velocity']:10.2f}  "
                      f"air frames min {row['min_air_frames']:>3} "
                      f"mean {row['mean_air_frames']:6.2f}"
                      f"  {'RUNAWAY' if row['runaway'] else ''}"
                      f"{' ESCAPES' if row['escapes_warp'] else ''}", flush=True)
    finally:
        game.close()

    print(f"\n{'scene':<24}{'best peak':>12}{'air frames':>12}{'runaway':>10}{'escapes':>9}")
    best = {}
    for label in scenes:
        scene_rows = [row for row in rows if row["scene"] == label]
        pick = min(scene_rows, key=lambda row: row["peak_velocity"])
        best[label] = pick
        print(f"{label:<24}{pick['peak_velocity']:>12.2f}{pick['mean_air_frames']:>12.2f}"
              f"{str(any(row['runaway'] for row in scene_rows)):>10}"
              f"{str(any(row['escapes_warp'] for row in scene_rows)):>9}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"calibration": {key: [round(value, 2) for value in row]
                                   for key, row in table.items()},
                   "frames": args.frames, "runaway_threshold": RUNAWAY,
                   "best_per_scene": best, "runs": rows}, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
