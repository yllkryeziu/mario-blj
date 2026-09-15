"""Cuts an injected sm64-port render into the clip set a post embeds.

The render this reads is not a stylised view of the policy, it is the policy: ``patches/`` teaches
sm64-port to stamp the recorded per frame ``gMarioState`` back over its own after physics runs, so
the frames are the real game drawing the real episode with the real castle, and the audio beside
them is the real game's audio for the same frames.

Three decisions are baked in here, each of them measured rather than guessed.

**No GIFs.** The same thirty three frame clip at 640 pixels wide costs 3.4 MB as a GIF, 374 KB as
an animated WebP and 148 KB as h264. GIF loses by 23x to the format every browser has hardware
decoders for, so every clip ships as h264 and the short ones additionally ship as animated WebP,
which is the format that still works inside a plain ``<img>`` for a reader who wants a loop with
no player chrome. Long clips get no WebP at all: the encoder's wall clock grows fast enough that
the 668 frame hero clip hangs for minutes, which is why ``--webp_max_frames`` exists.

**One audio seek, no resampling.** ``patches/sm64-port-audio-dump.patch`` overrides the game's
audio buffer size with the 528, 528, 544 cycle while dumping, whose mean is exactly 533.33 samples
per block and therefore 32000/30 samples per frame, so the dumped stream is locked to a 30 fps
render by construction. A clip beginning at rendered frame ``first`` is then simply
``(first - --dump_first) / --fps`` seconds into the raw stream. Measured drift over the whole 668
frame episode: 0.3 milliseconds.

**A crop, not a scale.** The dump is 1280x890 with a two pixel seam at the bottom, so the video
filter crops to 1280x888 before scaling. ``eq=gamma=2.0`` is there because the staircase interior
is lit for a CRT and reads almost black in a browser; it is a display choice and it is applied
identically to every clip.

Telemetry comes from ``tools/make_ass.py``, once per output size, and libass burns it in.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import make_ass

# The clip set, as (name, first rendered frame, last rendered frame). The frame numbers are the
# render's: the 2016M TAS movie drives the controller until the policy takes over at 6732, so a
# replay index i is rendered frame 6732 + i, and the pre-roll frames before that are what the
# spawn clip opens on.
DEFAULT_CLIPS = (
    ("00-full", 6725, 7392),
    ("01-spawn", 6725, 6762),
    ("02-approach", 6752, 6792),
    ("03-pump", 6918, 6956),
    ("04-fallback", 7038, 7150),
    ("05-launch", 7284, 7322),
    ("06-goal", 7344, 7392),
)


@dataclasses.dataclass(frozen=True)
class Encoded:
    """One clip's measured result."""

    name: str
    frames: int
    seconds: float
    mp4_bytes: int
    webp_bytes: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Builds the command line.

    Args:
        argv: Arguments without the program name.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames_dir", required=True,
                        help="Directory of frame_%%06d.ppm the dump wrote.")
    parser.add_argument("--replay", required=True,
                        help="Replay JSON scripts/record_episode.py wrote.")
    parser.add_argument("--out_dir", required=True, help="Directory to write clips into.")
    parser.add_argument("--audio", default=None,
                        help="Raw s16le stereo audio the dump wrote, or None for silent clips.")
    parser.add_argument("--audio_rate", type=int, default=32000, help="Audio sample rate.")
    parser.add_argument("--dump_first", type=int, default=6725,
                        help="Rendered frame the dump and the audio both start on.")
    parser.add_argument("--inject_start", type=int, default=6732,
                        help="Rendered frame the policy took the controller over on.")
    parser.add_argument("--clips", default="",
                        help="Clips as name:first:last, comma separated. Empty for the default "
                             "set this file documents.")
    parser.add_argument("--fps", type=int, default=30, help="Render and output frame rate.")
    parser.add_argument("--crop", default="1280:888:0:0",
                        help="ffmpeg crop applied before scaling, as w:h:x:y.")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Gamma lift, because the castle interior is lit for a CRT.")
    parser.add_argument("--video_size", default="960:666", help="h264 output size, as w:h.")
    parser.add_argument("--webp_size", default="640:444", help="Animated WebP size, as w:h.")
    parser.add_argument("--crf", type=int, default=20, help="libx264 quality.")
    parser.add_argument("--webp_quality", type=int, default=72, help="libwebp_anim quality.")
    parser.add_argument("--webp_max_frames", type=int, default=200,
                        help="Clips longer than this get no WebP, because the encoder's wall "
                             "clock makes it unaffordable and the payload unshippable.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary.")
    return parser.parse_args(argv)


def parse_clips(spec: str) -> tuple[tuple[str, int, int], ...]:
    """Reads the clip list off the command line.

    Args:
        spec: Clips as ``name:first:last``, comma separated, possibly empty.

    Returns:
        The clips, or the documented default set when the spec is empty.

    Raises:
        ValueError: If an entry is not three colon separated fields.
    """
    if not spec.strip():
        return DEFAULT_CLIPS
    clips = []
    for entry in spec.split(","):
        fields = entry.strip().split(":")
        if len(fields) != 3:
            raise ValueError(f"clip {entry!r} is not name:first:last")
        clips.append((fields[0], int(fields[1]), int(fields[2])))
    return tuple(clips)


def write_track(args: argparse.Namespace, name: str, first: int, last: int,
                size: str) -> str:
    """Writes one clip's telemetry track at one output size.

    Args:
        args: Parsed command line.
        name: Clip name.
        first: First rendered frame.
        last: Last rendered frame.
        size: Output size as ``w:h``.

    Returns:
        The path of the written ASS file.
    """
    width, height = (int(value) for value in size.split(":"))
    path = os.path.join(args.out_dir, "ass", f"{name}-{width}.ass")
    make_ass.main(["--replay", args.replay, "--out", path, "--first", str(first),
                   "--last", str(last), "--width", str(width), "--height", str(height),
                   "--fps", str(args.fps), "--inject_start", str(args.inject_start)])
    return path


def encode(args: argparse.Namespace, name: str, first: int, last: int) -> Encoded:
    """Encodes one clip as h264, and as animated WebP when it is short enough.

    Args:
        args: Parsed command line.
        name: Clip name.
        first: First rendered frame.
        last: Last rendered frame.

    Returns:
        What the clip cost.

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    count = last - first + 1
    pattern = os.path.join(args.frames_dir, "frame_%06d.ppm")
    video_track = write_track(args, name, first, last, args.video_size)
    mp4 = os.path.join(args.out_dir, f"{name}.mp4")

    command = [args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", str(args.fps), "-start_number", str(first), "-i", pattern]
    if args.audio:
        # Bound the audio read with -t rather than leaning on -shortest. -shortest is applied
        # after the aac encoder has already been handed more than it needs, and on a 41 frame clip
        # that left 0.6 s of audio hanging past the end of the video. A seek plus an explicit
        # duration is exact, and both numbers are frame counts over the frame rate.
        offset = (first - args.dump_first) / float(args.fps)
        command += ["-f", "s16le", "-ar", str(args.audio_rate), "-ac", "2",
                    "-ss", f"{offset:.6f}", "-t", f"{count / float(args.fps):.6f}",
                    "-i", args.audio]
    command += ["-frames:v", str(count)]
    command += ["-vf", (f"crop={args.crop},eq=gamma={args.gamma},"
                        f"scale={args.video_size},ass={video_track}"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf)]
    if args.audio:
        command += ["-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    command += ["-movflags", "+faststart", mp4]
    subprocess.run(command, check=True)

    webp_bytes = 0
    if count <= args.webp_max_frames:
        webp_track = write_track(args, name, first, last, args.webp_size)
        webp = os.path.join(args.out_dir, f"{name}.webp")
        subprocess.run([args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-framerate", str(args.fps), "-start_number", str(first), "-i", pattern,
                        "-frames:v", str(count),
                        "-vf", (f"crop={args.crop},eq=gamma={args.gamma},"
                                f"scale={args.webp_size},ass={webp_track}"),
                        "-c:v", "libwebp_anim", "-loop", "0",
                        "-q:v", str(args.webp_quality), "-compression_level", "4", webp],
                       check=True)
        webp_bytes = os.path.getsize(webp)
    return Encoded(name=name, frames=count, seconds=count / float(args.fps),
                   mp4_bytes=os.path.getsize(mp4), webp_bytes=webp_bytes)


def main(argv: list[str]) -> int:
    """Cuts the clip set.

    Args:
        argv: Arguments without the program name.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    os.makedirs(os.path.join(args.out_dir, "ass"), exist_ok=True)
    results = [encode(args, *clip) for clip in parse_clips(args.clips)]
    for result in results:
        webp = f"{result.webp_bytes // 1024:6d} KB" if result.webp_bytes else "     --"
        print(f"{result.name:<14s} {result.frames:4d} frames  {result.seconds:5.1f}s  "
              f"mp4 {result.mp4_bytes // 1024:6d} KB  webp {webp}")
    totals: dict[str, Any] = {
        "clips": len(results),
        "mp4_bytes": sum(result.mp4_bytes for result in results),
        "webp_bytes": sum(result.webp_bytes for result in results),
    }
    print(f"total  mp4 {totals['mp4_bytes'] // 1024} KB, "
          f"webp {totals['webp_bytes'] // 1024} KB")
    manifest = os.path.join(args.out_dir, "clips.json")
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"clips": [dataclasses.asdict(result) for result in results], **totals},
                  handle, indent=1)
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
