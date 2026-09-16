"""Reduces a crowd capture manifest to the scalars the blog post's panels quote.

``tools/export_swarm_render.py`` writes a manifest beside its containers that records, for every
shot, how each 64-policy population behaved during the fifteen seconds that were filmed. The
containers and their audio are hundreds of megabytes and stay out of git; this pulls out the
per-checkpoint counts the four-panel figures label themselves with and writes them where the site's
distill step reads from.

One property of the window is worth stating, because it decides what the counts mean. An episode
ends either at the goal or at the 1200 frame timeout, and the window is 450 frames, so no episode in
a shot can end by timing out. Every episode that finished inside the window finished by escaping:
``episodesFinished`` and ``successes`` are the same number in every row, and the figure is escapes
completed in fifteen seconds by a population of sixty four, not a success rate over attempts.
"""

from __future__ import annotations

import argparse
import json
import os

STEP_LABELS = ("1M", "5M", "10M", "20M")


def parse_args() -> argparse.Namespace:
    """Builds the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="manifest.json written by tools/export_swarm_render.py.")
    parser.add_argument("--out", required=True, help="Where to write the JSON.")
    parser.add_argument("--labels", default=",".join(STEP_LABELS),
                        help="Checkpoint labels, in panel order left to right.")
    return parser.parse_args()


def main() -> None:
    """Writes the swarm render summary."""
    args = parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    labels = args.labels.split(",")

    shots = {(shot["rung"], shot["stepsLabel"]): shot for shot in manifest["shots"]}
    rungs = []
    for rung in dict.fromkeys(shot["rung"] for shot in manifest["shots"]):
        if rung == "random":
            continue
        rungs.append(rung)

    solved, detail = {}, {}
    for rung in rungs:
        row = [shots[(rung, label)] for label in labels if (rung, label) in shots]
        if len(row) != len(labels):
            raise ValueError(f"{rung} has {len(row)} of {len(labels)} checkpoints in the manifest")
        solved[rung] = [shot["successes"] for shot in row]
        detail[rung] = [{
            "steps": shot["steps"],
            "label": shot["stepsLabel"],
            "checkpoint": shot["checkpoint"],
            "episodesFinished": shot["episodesFinished"],
            "successes": shot["successes"],
            "peakBackward": shot["peakBackwardInShot"],
            "bestHeight": shot["bestHeightInShot"],
            "meanBestHeight": shot["meanBestHeight"],
        } for shot in row]

    cold = shots.get(("random", "untrained"))
    summary = {
        "population": manifest["population"],
        "frames": manifest["frames"],
        "frameRate": manifest["frameRate"],
        "seconds": round(manifest["frames"] / manifest["frameRate"], 3),
        "warmupFrames": manifest["warmup"],
        "episodeFrames": manifest["episodeFrames"],
        "spawnSpread": manifest["spawnSpread"],
        "seed": manifest["seed"],
        "stochastic": manifest["stochastic"],
        "labels": labels,
        # Every termination inside the window is an escape, so this is the same array as
        # episodes finished. See the module docstring.
        "solved": solved,
        "rungs": detail,
        "coldOpen": None if cold is None else {
            "population": cold["population"],
            "frames": cold["frames"],
            "successes": cold["successes"],
            "bestHeight": cold["bestHeightInShot"],
            "peakBackward": cold["peakBackwardInShot"],
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.out}")
    for rung in rungs:
        counts = " / ".join(f"{count:>3}" for count in solved[rung])
        print(f"  {rung:<13} {counts}")


if __name__ == "__main__":
    main()
