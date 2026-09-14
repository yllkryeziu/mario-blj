"""Packs the recorded replays into the one payload the 2D viewer page carries.

The recorders write a row per frame, which is convenient to produce and wasteful to inline: the
viewer plots whole series and never looks at a row. This transposes each replay into parallel
arrays, replaces the repeated action names with a table plus an index per frame, and precomputes
the floor profile the page draws the ground with, since it has no collision code of its own.

The profile is the only real work here. A replay carries the scene's triangles but the page needs
a height per z, so this walks the corridor at a fixed step and solves each triangle's plane for
the highest floor above that point, which is the same answer the game's floor lookup would give.
"""

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def floor_height_at(triangles: list, x: float, z: float) -> float | None:
    """Finds the highest floor above one point, the way the game's floor lookup does.

    A triangle is a candidate when the point is inside its projection onto the xz plane, tested
    by the sign of the three edge cross products, and it is a floor rather than a wall when its
    normal has a positive y. The plane equation then gives the height. Near vertical surfaces are
    rejected at a y normal of 0.01, which keeps the walls of the staircase out of the profile.

    Args:
        triangles: The scene's triangles, each three vertices of xyz.
        x: x of the query point.
        z: z of the query point.

    Returns:
        The height of the highest floor above the point, or None if there is no floor there.
    """
    best = None
    for (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) in triangles:
        if (z1 - z) * (x2 - x1) - (x1 - x) * (z2 - z1) < 0:
            continue
        if (z2 - z) * (x3 - x2) - (x2 - x) * (z3 - z2) < 0:
            continue
        if (z3 - z) * (x1 - x3) - (x3 - x) * (z1 - z3) < 0:
            continue
        nx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        ny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        nz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if ny == 0:
            continue
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        nx, ny, nz = nx / length, ny / length, nz / length
        if ny <= 0.01:
            continue
        height = -(x * nx + nz * z + -(nx * x1 + ny * y1 + nz * z1)) / ny
        if best is None or height > best:
            best = height
    return best


def profile(triangles: list, x: float = 0.0, step: float = 25.0) -> list:
    """Samples the floor height along z at one x, for the page's side elevation.

    The x is the path Mario actually took rather than the middle of the scene, because the
    staircase's corridor is off centre and a profile down the middle would draw the wrong ground.

    Args:
        triangles: The scene's triangles.
        x: The x to take the elevation at.
        step: Sampling interval along z.

    Returns:
        One [z, height] pair per sample that had a floor, in ascending z.
    """
    zs = [vertex[2] for triangle in triangles for vertex in triangle]
    z = min(zs)
    limit = max(zs)
    points = []
    while z <= limit:
        height = floor_height_at(triangles, x, z)
        if height is not None:
            points.append([round(z, 1), round(height, 1)])
        z += step
    return points


def pack(path: str) -> dict:
    """Transposes one replay into the parallel arrays the page reads.

    Action names go into a table with one index per frame, which is the one substantial saving
    here: the names repeat for hundreds of consecutive frames.

    Args:
        path: Path to a replay JSON from either recorder.

    Returns:
        The packed scene. Fields the synthetic recorder does not write, the profile x, the success
        flag and the per frame warp flags, fall back to their empty values.

    Raises:
        OSError: If the replay cannot be read.
        KeyError: If the replay is missing a field both recorders write.
    """
    with open(path, encoding="utf-8") as handle:
        replay = json.load(handle)
    frames = replay["frames"]
    actions = []
    lookup = {}
    codes = []
    for row in frames:
        label = row["action_name"]
        if label not in lookup:
            lookup[label] = len(actions)
            actions.append(label)
        codes.append(lookup[label])
    return {
        "scene": replay["scene"],
        "peak_velocity": replay["peak_velocity"],
        "profile": profile(replay["surfaces"], float(replay.get("profile_x", 0.0))),
        "profile_x": float(replay.get("profile_x", 0.0)),
        "success": bool(replay.get("success", False)),
        "warped": [int(bool(row.get("warped", False))) for row in frames],
        "actions": actions,
        "code": codes,
        "x": [row["position"][0] for row in frames],
        "y": [row["position"][1] for row in frames],
        "z": [row["position"][2] for row in frames],
        "vy": [row["velocity"][1] for row in frames],
        "v": [row["forward_velocity"] for row in frames],
        "stage": [row["stage"] for row in frames],
        "stick_x": [row["inputs"]["stick_x"] for row in frames],
        "stick_y": [row["inputs"]["stick_y"] for row in frames],
        "button_a": [row["inputs"]["a"] for row in frames],
        "button_b": [row["inputs"]["b"] for row in frames],
        "button_z": [row["inputs"]["z"] for row in frames],
        "cycles": replay["cycles"],
    }


def main() -> None:
    """Packs every ``replay_*.json`` in the results directory into one payload.

    Scenes are keyed by their own name, so re-recording one scene replaces it and leaves the
    others in place.

    Raises:
        OSError: If the results directory or the output path cannot be opened.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(RESULTS, "replays.json"))
    args = parser.parse_args()

    packed = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "replay_*.json"))):
        entry = pack(path)
        packed[entry["scene"]] = entry
        print(f"{entry['scene']:>8}  {len(entry['profile']):5d} profile points  "
              f"{len(entry['v']):5d} frames  peak {entry['peak_velocity']:9.2f}")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(packed, handle, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
