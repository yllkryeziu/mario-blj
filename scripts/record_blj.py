"""Records a scripted chain on a synthetic scene into the replay format the 2D viewer reads.

This is the recorder for the scenes that are not the real staircase: flat ground, a ramp and a
synthetic flight of stairs, none of which has an instant warp. Since there is no warp there is no
environment either, only the raw libsm64 handle and the scripted policy, which is why the payload
this writes has no success flag or warp count while the one from ``scripts/record_episode.py``
does. Both feed ``tools/pack_replays.py``.
"""

import argparse
import json
import os

from src.agent.scripted import ScriptedBlj, calibrate_stick, run_chain, stick_toward
from src.env.geometry import flat_area, ground_plane, ramp, staircase
from src.env.native import Sm64, action_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

APRON = flat_area(-3000.0, 3000.0, -12000.0, 0.0)
SPAWN = (0.0, 100.0, -2000.0)

SCENES = {
    "stairs": APRON + staircase(400, 75.0, 100.0, 6000.0),
    "flat": APRON,
    "ramp": APRON + ramp(24000.0, 6000.0, 30.0),
}


def triangles(surfaces: list) -> list:
    """Copies surfaces out of ctypes into plain integers the JSON encoder accepts.

    Args:
        surfaces: The scene's surfaces.

    Returns:
        One triangle per surface, each three vertices of three integers, in winding order.
    """
    return [[[int(surface.vertices[i][axis]) for axis in range(3)] for i in range(3)]
            for surface in surfaces]


def main() -> None:
    """Records one run and writes the replay JSON.

    The stick is calibrated on a flat plane before the scene is loaded, which replaces the
    process wide surface set; ``run_chain`` loads the real scene again afterwards, so the order
    here matters.

    Raises:
        OSError: If the ROM, the shared library or the output path cannot be opened.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rom", default=os.path.join(ROOT, "roms", "baserom.us.z64"))
    parser.add_argument("--scene", default="stairs", choices=sorted(SCENES))
    parser.add_argument("--frames", type=int, default=700)
    parser.add_argument("--air-magnitude", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    surfaces = SCENES[args.scene]
    game = Sm64(args.rom)
    try:
        table = calibrate_stick(game, ground_plane(8000.0), (0.0, 100.0, 0.0))
        approach = stick_toward(table, 0.0, -1.0)
        policy = ScriptedBlj(approach, air_magnitude=args.air_magnitude)
        result = run_chain(game, surfaces, SPAWN, policy, args.frames)
    finally:
        game.close()

    for row in result["trace"]:
        row["action_name"] = action_name(row["action"])

    payload = {
        "scene": args.scene,
        "air_magnitude": args.air_magnitude,
        "approach_stick": approach,
        "surfaces": triangles(surfaces),
        "peak_velocity": result["peak_velocity"],
        "cycles": result["cycles"],
        "frames": result["trace"],
    }
    out = args.out or os.path.join(RESULTS, f"replay_{args.scene}.json")
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print(f"{args.scene}: {len(result['trace'])} frames, "
          f"{len(result['cycles'])} long jumps, peak forwardVel {result['peak_velocity']}")
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
