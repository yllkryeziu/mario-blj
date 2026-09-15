"""Summarises the swarm captures and their audio into one figure-ready JSON.

The blog post cites numbers that live in two places the site build cannot reach: the swarm
container in a scratch directory, and the mp3s the capture wrote beside it. Both are large and
neither belongs in git, so this reduces them to the handful of scalars the prose actually quotes
and writes them into ``results/`` where the site's distill step reads from.

The audio half is worth explaining. libsm64 compiles in the decompilation's audio engine, so the
samples are the game's own, driven by the ``play_sound`` calls Mario's actions make.
``act_long_jump`` plays ``SOUND_MARIO_YAHOO``, and a working chain re-enters that action on nearly
every frame, so a swarm that has found the exploit is audibly different from one that has not. The
instant warp, by contrast, plays nothing at all, which is why the trapped rung ends up quieter
rather than louder.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SAMPLE_RATE = 32000
FRAME_RATE = 30.0
BANDS = (("sub", 0, 200), ("low_voice", 200, 800), ("high_voice", 800, 2000),
         ("bright", 2000, 5000), ("hiss", 5000, 16000))
WINDOW = 2048
HOP = 512


def parse_args() -> argparse.Namespace:
    """Builds the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swarm", required=True, help="Converged four rung swarm container.")
    parser.add_argument("--audio_dir", required=True,
                        help="Directory of swarm_<rung>-final.mp3 files, one per rung.")
    parser.add_argument("--progression_dir", default="",
                        help="Optional directory of per checkpoint mp3s for one rung.")
    parser.add_argument("--progression_prefix", default="swarm_terminal-s2-",
                        help="Filename prefix inside --progression_dir.")
    parser.add_argument("--progression_steps", default="0.50,1.00,2.00,3.00,4.00,5.00,6.00,7.00,"
                                                       "8.00,10.00,12.00,16.00,20.00",
                        help="Comma separated checkpoint labels, in order.")
    parser.add_argument("--excerpt_start", type=float, default=8.0,
                        help="Seconds into each capture the excerpt starts, past the reset burst.")
    parser.add_argument("--excerpt_seconds", type=float, default=3.0, help="Excerpt length.")
    parser.add_argument("--out", required=True, help="Where to write the JSON.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary.")
    return parser.parse_args()


def read_header(path: str) -> dict:
    """Reads a swarm container's JSON header.

    Args:
        path: Path to the container.

    Returns:
        The decoded header.

    Raises:
        ValueError: If the magic is not a swarm container's.
    """
    with open(path, "rb") as handle:
        magic = handle.read(9)
        if magic != b"MBLJSWRM1":
            raise ValueError(f"{path} is not an MBLJSWRM1 container, it starts {magic!r}")
        length = struct.unpack("<I", handle.read(4))[0]
        return json.loads(handle.read(length))


def decode_pcm(path: str, ffmpeg: str, start: float = 0.0, seconds: float = 0.0) -> np.ndarray:
    """Decodes an mp3 to a mono float array at the game's sample rate.

    Args:
        path: The mp3 to decode.
        ffmpeg: ffmpeg binary.
        start: Seconds to seek before reading.
        seconds: Seconds to read, or 0 for all of it.

    Returns:
        Mono samples as float64, the mean of the two channels.
    """
    command = [ffmpeg, "-v", "error"]
    if start:
        command += ["-ss", f"{start:.6f}"]
    if seconds:
        command += ["-t", f"{seconds:.6f}"]
    command += ["-i", path, "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-"]
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype="<i2").reshape(-1, 2).mean(axis=1).astype(np.float64)


def spectrum(mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Averages a Hann windowed magnitude spectrum over the whole signal.

    Args:
        mono: Mono samples.

    Returns:
        The frequency bins and the mean magnitude in each.
    """
    count = (len(mono) - WINDOW) // HOP
    strides = (mono.strides[0] * HOP, mono.strides[0])
    frames = np.lib.stride_tricks.as_strided(mono, (count, WINDOW), strides).copy()
    frames *= np.hanning(WINDOW)
    magnitude = np.abs(np.fft.rfft(frames, axis=1))
    return np.fft.rfftfreq(WINDOW, 1.0 / SAMPLE_RATE), magnitude.mean(axis=0)


def describe(mono: np.ndarray) -> dict:
    """Reduces one track to the scalars the post quotes.

    Args:
        mono: Mono samples.

    Returns:
        Duration, level, spectral centroid and the per band energy shares.
    """
    freq, magnitude = spectrum(mono)
    total = magnitude.sum()
    hop = int(SAMPLE_RATE / FRAME_RATE)
    count = len(mono) // hop
    envelope = np.sqrt((mono[:count * hop].reshape(count, hop) ** 2).mean(axis=1))
    return {
        "seconds": round(len(mono) / SAMPLE_RATE, 3),
        "rms": round(float(np.sqrt((mono ** 2).mean())), 1),
        "peak": round(float(np.abs(mono).max()), 1),
        "centroid_hz": round(float((freq * magnitude).sum() / total), 1),
        "loud_frame_fraction": round(float(np.mean(envelope > envelope.max() * 0.15)), 4),
        "bands": {name: round(float(magnitude[(freq >= lo) & (freq < hi)].sum() / total), 4)
                  for name, lo, hi in BANDS},
    }


def rung_outcomes(header: dict) -> list[dict]:
    """Pulls each rung's converged behaviour out of the container header.

    Args:
        header: A swarm container header.

    Returns:
        One row per rung, in container order.
    """
    rows = []
    for entry in header["checkpoints"]:
        outcome = entry["outcome"]
        rows.append({
            "rung": entry["name"].split("-")[0],
            "steps": entry["steps"],
            "episodes": outcome["episodes"],
            "successes": outcome["successes"],
            "bestHeight": round(outcome["best_height"], 1),
            "meanBestHeight": round(outcome["mean_best_height"], 1),
            "bestPeakBackward": round(outcome["best_peak_backward"], 1),
            "meanPeakBackward": round(outcome["mean_peak_backward"], 1),
            "meanReturn": round(outcome["mean_return"], 4),
        })
    return rows


def main() -> None:
    """Writes the media summary."""
    args = parse_args()
    header = read_header(args.swarm)
    rungs = rung_outcomes(header)
    for row in rungs:
        path = os.path.join(args.audio_dir, f"swarm_{row['rung']}-final.mp3")
        if os.path.exists(path):
            row["audio"] = describe(decode_pcm(path, args.ffmpeg))

    escaping = [row for row in rungs if row["successes"] > 0 and "audio" in row]
    trapped = [row for row in rungs if row["successes"] == 0 and "audio" in row]
    contrast = {}
    if escaping and trapped:
        for name, _, _ in BANDS:
            mean_escape = float(np.mean([row["audio"]["bands"][name] for row in escaping]))
            contrast[name] = round(trapped[0]["audio"]["bands"][name] / mean_escape, 3)

    progression = []
    if args.progression_dir:
        for label in args.progression_steps.split(","):
            path = os.path.join(args.progression_dir, f"{args.progression_prefix}{label}M.mp3")
            if not os.path.exists(path):
                print(f"progression: no capture at {label}M, skipping")
                continue
            mono = decode_pcm(path, args.ffmpeg, args.excerpt_start, args.excerpt_seconds)
            progression.append({"steps": float(label) * 1e6,
                                "rms": round(float(np.sqrt((mono ** 2).mean())), 1)})

    summary = {
        "sampleRate": SAMPLE_RATE,
        "frameRate": FRAME_RATE,
        "excerpt": {"startSeconds": args.excerpt_start, "seconds": args.excerpt_seconds},
        "rungs": rungs,
        "trappedOverEscapingByBand": contrast,
        "progression": progression,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.out}")
    for row in rungs:
        audio = row.get("audio", {})
        print(f"  {row['rung']:<13} {row['successes']:>4}/{row['episodes']:<4} succ  "
              f"peak_back {row['bestPeakBackward']:>9}  "
              f"rms {audio.get('rms', '?'):>7}  centroid {audio.get('centroid_hz', '?'):>7}")
    if progression:
        print("  progression rms:", " ".join(f"{p['rms']:.0f}" for p in progression))


if __name__ == "__main__":
    main()
