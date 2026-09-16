"""Summarises the swarm captures and their audio into one figure-ready JSON.

The blog post cites numbers that live in two places the site build cannot reach: the swarm
container in a scratch directory, and the mp3s the capture wrote beside it. Both are large and
neither belongs in git, so this reduces them to the handful of scalars the prose actually quotes
and writes them into ``results/`` where the site's distill step reads from.

The audio half is worth explaining. libsm64 compiles in the decompilation's audio engine, so the
samples are the game's own, driven by the ``play_sound`` calls Mario's actions make. The instant
warp plays nothing at all, which is why the trapped rung ends up quieter rather than louder.

``act_long_jump`` plays ``SOUND_MARIO_YAHOO`` on entry and a working chain re-enters it on nearly
every frame, which reads as an argument that the exploit should be the loudest thing in the run.
It is the opposite, and ``--episode_audio`` is here to measure that rather than argue it.
``SOUND_MARIO_YAHOO`` carries ``SOUND_DISCRETE`` (``include/sounds.h``), whose contract is that
every ``play_sound`` call restarts the sample; ``set_mario_action`` clears the played flag on every
transition, so each press asks for the yell again; and ``process_sound_request`` will not stack a
second request from a source already holding a slot in that bank, so it overwrites the pending one.
A press therefore never gets more than a frame or two of its own sample played before the next
press cuts it off. Mario is yelling the whole way up the staircase and what comes out is the attack
of a yell, twelve times, and none of the rest of it.
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
    parser.add_argument("--episode_audio", default="",
                        help="The filmed episode's audio, as anything ffmpeg can decode. Given "
                             "with --replay, the summary gains per phase levels for that episode.")
    parser.add_argument("--replay", default="results/replay_model_endless.json",
                        help="The filmed episode's measured frames, which the phase windows are "
                             "derived from rather than typed in.")
    parser.add_argument("--amplify_ratio", type=float, default=1.3,
                        help="Frame over frame speed ratio that counts as a press amplifying, "
                             "which separates the chain from the 0.98 of ground friction cleanly; "
                             "the real values are either about 1.4 or about 0.98.")
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



def episode_phases(replay: dict, amplify_ratio: float, gap: int = 4) -> dict:
    """Locates the parts of the filmed episode the post talks about separately.

    Every window is found from the measured frames rather than written down, so re-recording the
    episode moves them. The chain gets two windows, because there are two defensible edges to it
    and the post quotes both. ``chain`` is the run of presses over which speed grows monotonically
    all the way to the peak, which is the window the press table shows and the tighter of the two.
    ``chainFull`` starts at the first amplifying press of the cluster that produced the peak, four
    presses earlier: those presses do amplify, but two consecutive friction frames after the fourth
    cost more than it gained, so speed is not yet monotone there. Reporting both means the audio
    claim can be read against either edge, and it survives both.

    The reference jump is the maximal run of ``long_jump`` immediately before the chain, which is
    one ordinary jump, and the flight is everything after the chain, which is the ascent it bought.

    Args:
        replay: A decoded replay recording.
        amplify_ratio: Speed ratio above which a frame counts as an amplifying press.
        gap: Largest frame gap between consecutive presses of one chain. The chain alternates
            press and friction frame, so its real gaps are two to four.

    Returns:
        One entry per phase, each with its first and last frame, and each chain window's press
        count and amplification range alongside.
    """
    frames = replay["frames"]
    speed = [abs(frame["forward_velocity"]) for frame in frames]
    peak = max(range(len(frames)), key=lambda i: speed[i])

    amplifying = [i for i in range(1, len(frames))
                  if frames[i - 1]["forward_velocity"]
                  and frames[i]["forward_velocity"] / frames[i - 1]["forward_velocity"]
                  > amplify_ratio]
    clusters: list[list[int]] = []
    for index in amplifying:
        if clusters and index - clusters[-1][-1] <= gap:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    full = min(clusters, key=lambda c: (not c[0] <= peak <= c[-1], -len(c)))

    # The monotone window: walk back from the peak over every frame that presses A, not only the
    # ones that amplified, for as long as each is slower than the press after it. Including the
    # discarded presses is what moves the edge, because a press that pressed A without holding Z
    # pays friction and can leave the chain slower than it was four frames earlier. This is the
    # edge the press table uses.
    a_pressed = [i for i in range(full[0], peak + 1) if frames[i]["inputs"]["a"] == 1]
    monotone = [peak]
    for index in reversed([i for i in a_pressed if i < peak]):
        if speed[index] < speed[monotone[0]]:
            monotone.insert(0, index)
        else:
            break


    jump_last = monotone[0] - 1
    while jump_last > 0 and frames[jump_last]["action_name"] != "long_jump":
        jump_last -= 1
    jump_first = jump_last
    while jump_first > 0 and frames[jump_first - 1]["action_name"] == "long_jump":
        jump_first -= 1

    def chain_window(first: int, last: int) -> dict:
        """Describes one chain window.

        The window runs from the first frame that presses A to the peak, so it can open on a press
        that was discarded; the press count is of the presses inside it that actually amplified.

        Args:
            first: First frame of the window.
            last: Last frame of the window, inclusive.

        Returns:
            The window's edges, its amplifying press count and their amplification range.
        """
        inside = [i for i in amplifying if first <= i <= last]
        ratios = [round(frames[i]["forward_velocity"] / frames[i - 1]["forward_velocity"], 3)
                  for i in inside]
        return {"first": first, "last": last, "presses": len(inside),
                "ratios": [min(ratios), max(ratios)], "peakFrame": peak}

    return {
        "chain": chain_window(monotone[0], peak),
        "chainFull": chain_window(full[0], peak),
        "jumpBefore": {"first": jump_first, "last": jump_last},
        "flightAfter": {"first": full[-1] + 1, "last": len(frames) - 1},
        "episode": {"first": 0, "last": len(frames) - 1},
    }


def window_level(mono: np.ndarray, first: int, last: int) -> dict:
    """Measures one window of the episode, bucketed one bucket per game frame.

    Args:
        mono: The whole episode's mono samples, starting at replay frame 0.
        first: First replay frame in the window.
        last: Last replay frame in the window, inclusive.

    Returns:
        The window's length, its RMS and its peak, or nulls if it falls outside the recording.
    """
    hop = SAMPLE_RATE / FRAME_RATE
    begin, end = int(first * hop), min(int((last + 1) * hop), len(mono))
    if end <= begin:
        return {"frames": last - first + 1, "rms": None, "peak": None}
    window = mono[begin:end]
    return {"frames": last - first + 1,
            "rms": round(float(np.sqrt((window ** 2).mean())), 1),
            "peak": round(float(np.abs(window).max()), 1)}


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

    episode = None
    if args.episode_audio:
        with open(args.replay, encoding="utf-8") as handle:
            replay = json.load(handle)
        phases = episode_phases(replay, args.amplify_ratio)
        mono = decode_pcm(args.episode_audio, args.ffmpeg)
        episode = {"source": os.path.basename(args.episode_audio),
                   "seconds": round(len(mono) / SAMPLE_RATE, 3), "phases": {}}
        for name, window in phases.items():
            episode["phases"][name] = {**window,
                                       **window_level(mono, window["first"], window["last"])}

    summary = {
        "sampleRate": SAMPLE_RATE,
        "frameRate": FRAME_RATE,
        "excerpt": {"startSeconds": args.excerpt_start, "seconds": args.excerpt_seconds},
        "rungs": rungs,
        "trappedOverEscapingByBand": contrast,
        "progression": progression,
        "episodeAudio": episode,
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
    if episode:
        print(f"  filmed episode, {episode['seconds']}s:")
        for name, phase in episode["phases"].items():
            print(f"    {name:<12} frames {phase['first']:>3}-{phase['last']:<3} "
                  f"({phase['frames']:>3})  rms {phase['rms']:>7}  peak {phase['peak']:>8}")


if __name__ == "__main__":
    main()
