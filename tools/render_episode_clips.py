"""Renders one recorded episode as the post's two hero clips: the slow motion and the whole run.

The environment drives libsm64, which returns geometry rather than pixels, so a measured episode
can never be screenshotted directly. ``patches/`` closes that gap: the injector stamps a recorded
trajectory over ``gMarioState`` after physics has run, so the game draws the episode it was given
and nothing is re-simulated from inputs. What comes out cannot drift from what was measured.

Two things had to be added for this to be watchable.

The camera is the first. The game's is written around a Mario who moves a few dozen units a frame,
and the final chain moves him 1117 units in one. Filming the injected run with it produces seconds
at a time inside the staircase's walls, where back face culling leaves a black wedge across the
frame, and then a snap to a close up of his face when it catches up. ``SM64_CAM_CHASE`` replaces it
with a rigid offset: always the same distance behind and above him, always inside the corridor
because he is.

The warp band is the second. The trap has no appearance of its own -- walking into it simply puts
Mario back where he came from -- so ``SM64_WARP_BAND`` paints the collision triangles the level
types as instant warps. With the band drawn, the two clips argue for themselves: at replay frame
214 Mario is already travelling backwards faster than the 154 units a frame the band takes to
cross, and he is thrown back anyway because he lands inside it; at frame 570 he crosses it.

The readout is burned in by ``tools/make_ass.py`` through libass rather than drawn by this file, and
it counts replay frames, which is what the post's chapter marks seek by.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CHASE = "0,260,760,60,55"
"""dx, dy, dz, look dy, fov. Behind, above, and aimed at his chest."""

# The frame dumper counts buffer swaps, and the swap it labels N carries the state the game
# simulated for gGlobalTimer N+1. The injector's first frame lands on gGlobalTimer
# SM64_INJECT_START, so replay frame 0 is dumped as SM64_INJECT_START - 1, and dumping from there
# makes replay index and dump number differ by exactly that constant.
DUMP_LEADS_TIMER_BY = 1

BEATS = (
    ("frame 214 lands inside the band, 217 clears it", 204, 222),
    ("the chain: x1.33 to x1.50, then over the band", 548, 580),
)
"""Slow motion beats, as (title, first replay frame, last replay frame).

The first beat is the whole argument in nineteen frames. At replay frame 214 Mario is travelling
backwards at 176 units a frame, comfortably past the 154 the band is wide, and he is thrown back
regardless, because the frame he lands on puts him at z 995.6, inside it. Three frames later, at
381 units a frame, he steps z 1109 to z 878 and never touches it. The chain survives the warp that
the first landing cost him -- displacement resets where he is and not how fast he is going, which
is why 217 is faster than 214 by a factor of 2.17 despite being thrown down the flight in between
-- so the landing that escapes is two treads lower than the landing that did not. What decides it
is where a landing falls, not how fast it is and not how high.

The second beat is the chain that finishes the episode: twelve amplifying presses from 551 to 576,
x1.334 to x1.498, the crossing at 570 and the peak of -1117.80 at 576. The window opens at 548 so
the chain is on screen from its first press rather than from the middle.

The title carries no press count on purpose. This window holds twelve amplifying presses, but the
post's press table is scoped to the monotone run from 560 and counts eight, and two numbers for one
chain on one page is worse than none on the clip. Both windows are measured in
``results/media_summary.json`` as ``chain`` and ``chainFull``.

Twelve presses in this window and not twenty six frames' worth, because only half the frames
amplify, and which half is decided
by the inputs rather than by anything about the landing. ``act_long_jump_land`` opens by clearing
INPUT_A_PRESSED unless INPUT_Z_DOWN is held, so the press only counts if A and Z arrive together.
In this episode that conjunction separates the chain perfectly: all twelve frames with both
amplify, and all seventeen without both do not -- including two where A is pressed without Z, which
come out at exactly 0.980, the ground friction and nothing else.

It is also the quietest part of the run, which is the detail worth keeping. Measured on the
episode's own audio in ``results/media_summary.json``, the chain is 3754.6 RMS against 5086.7 for
the single ordinary long jump immediately before it and 5597.1 for the flight it buys.
SOUND_MARIO_YAHOO is a discrete sound, so every press restarts it and the next press cuts it off a
frame or two in. Twelve yells in twenty six frames are quieter than one yell in twenty three.
"""


def parse_args() -> argparse.Namespace:
    """Builds the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", default=os.path.join("results", "replay_model_endless.json"),
                        help="Replay JSON scripts/record_episode.py wrote.")
    parser.add_argument("--trajectory",
                        default=os.path.join("results", "trajectory_model_endless.bin"),
                        help="The same episode as the injector's binary format.")
    parser.add_argument("--out_dir", required=True, help="Directory to write the clips into.")
    parser.add_argument("--frames_dir", default="",
                        help="Where the PPM dump lives. Defaults to a frames subdir of --out_dir. "
                             "Reused when it already holds the whole episode, which makes "
                             "re-cutting free.")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Leave the dump on disk. It is about two gigabytes.")
    parser.add_argument("--game", default=os.path.join("third_party", "sm64-port", "build",
                                                       "us_pc", "sm64.us"),
                        help="The patched sm64-port binary.")
    parser.add_argument("--replay_dir", default=os.path.join("data", "tas", "replay2016M"),
                        help="Directory holding cont.m64, which the binary reads from its own cwd.")
    parser.add_argument("--tas", default=os.path.join("data", "tas", "tas_validation",
                                                      "sm64-0star-2016M.m64"),
                        help="Movie copied in as cont.m64 to walk the game to the staircase.")
    parser.add_argument("--tas_last", type=int, default=6731,
                        help="Frame the movie stops steering on.")
    parser.add_argument("--inject_start", type=int, default=6732,
                        help="gGlobalTimer value the trajectory's first frame is stamped on.")
    parser.add_argument("--chase", default=DEFAULT_CHASE, help="SM64_CAM_CHASE value.")
    parser.add_argument("--band", default="1", help="SM64_WARP_BAND value: 1, or r,g,b,a.")
    parser.add_argument("--width", type=int, default=960, help="Render and output width.")
    parser.add_argument("--height", type=int, default=720, help="Render and output height.")
    parser.add_argument("--gamma", type=float, default=1.8, help="Gamma applied on encode.")
    parser.add_argument("--slow", type=int, default=8,
                        help="Slow motion factor for the escape clip. Eight holds each frame for a "
                             "quarter of a second, which is about how long it takes to read a "
                             "velocity and a ratio off the readout.")
    parser.add_argument("--episode_crf", type=int, default=26, help="x264 quality for the run.")
    parser.add_argument("--escape_crf", type=int, default=21,
                        help="x264 quality for the slow motion, which is nearly a still per frame "
                             "and compresses far better.")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg", help="ffmpeg.")
    return parser.parse_args()


def episode_length(path: str) -> int:
    """Reads how many frames the replay holds.

    Args:
        path: Replay JSON path.

    Returns:
        The frame count.
    """
    with open(path, encoding="utf-8") as handle:
        return len(json.load(handle)["frames"])


def dump(args: argparse.Namespace, frames_dir: str, first: int, frames: int) -> str:
    """Runs the injected render once, leaving one PPM per frame and one raw audio stream.

    Args:
        args: Parsed arguments.
        frames_dir: Directory to dump into.
        first: Dump frame number replay frame 0 lands on.
        frames: Episode length.

    Returns:
        Path to the raw audio the run wrote.

    Raises:
        RuntimeError: If the run did not write the whole episode.
    """
    audio = os.path.join(os.path.dirname(frames_dir), "episode_audio.raw")
    held = len([name for name in os.listdir(frames_dir)]) if os.path.isdir(frames_dir) else 0
    if held >= frames and os.path.exists(audio):
        print(f"  reusing {held} frames in {frames_dir}")
        return audio

    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    root = os.path.abspath(os.curdir)
    environment = dict(os.environ)
    environment.update({
        "SM64_FAST": "1",
        "SM64_TAS_LAST": str(args.tas_last),
        "SM64_INJECT_START": str(args.inject_start),
        "SM64_INJECT_FILE": os.path.join(root, args.trajectory),
        "SM64_WARP_BAND": args.band,
        "SM64_CAM_CHASE": args.chase,
        "SM64_WINDOW_W": str(args.width),
        "SM64_WINDOW_H": str(args.height),
        "SM64_DUMP_DIR": os.path.abspath(frames_dir),
        "SM64_DUMP_FIRST": str(first),
        "SM64_DUMP_LAST": str(first + frames - 1),
        "SM64_DUMP_AUDIO": os.path.abspath(audio),
    })
    result = subprocess.run([os.path.join(root, args.game)], cwd=args.replay_dir,
                            env=environment, capture_output=True, text=True)
    for line in result.stderr.splitlines():
        if line.startswith(("inject:", "warp band:", "camera:", "audio dump:")):
            print(f"  {line}")
    written = len([name for name in os.listdir(frames_dir) if name.endswith(".ppm")])
    if written < frames:
        raise RuntimeError(f"dumped {written} of {frames} frames (exit {result.returncode})")
    return audio


def telemetry(args: argparse.Namespace, out: str, first: int, last: int, fps: float) -> str:
    """Writes the burned in readout for one clip.

    Args:
        args: Parsed arguments.
        out: ASS path to write.
        first: First rendered frame in the clip.
        last: Last rendered frame in the clip.
        fps: Rate the clip's source frames are presented at, which for slow motion is not 30.

    Returns:
        The path written.
    """
    subprocess.run([sys.executable, os.path.join("tools", "make_ass.py"),
                    "--replay", args.replay, "--out", out,
                    "--inject_start", str(args.inject_start - DUMP_LEADS_TIMER_BY),
                    "--first", str(first), "--last", str(last), "--fps", f"{fps:.6f}",
                    "--width", str(args.width), "--height", str(args.height)],
                   check=True, capture_output=True, text=True)
    return out


def escaped(path: str) -> str:
    """Escapes a path for use inside an ffmpeg filter argument.

    Args:
        path: A filesystem path.

    Returns:
        The path with the characters ffmpeg's filter parser treats as syntax backslashed.
    """
    return path.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def encode_episode(args: argparse.Namespace, frames_dir: str, audio: str, first: int, frames: int,
                   out: str) -> None:
    """Encodes the whole run at speed, with the readout and the game's own audio.

    Args:
        args: Parsed arguments.
        frames_dir: Directory of PPMs.
        audio: Raw audio the dump wrote.
        first: Dump frame number replay frame 0 lands on.
        frames: Episode length.
        out: Path to write.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    ass = telemetry(args, os.path.join(args.out_dir, "episode.ass"), first, first + frames - 1,
                    30.0)
    command = [args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", "30", "-start_number", str(first),
               "-i", os.path.join(frames_dir, "frame_%06d.ppm"),
               "-f", "s16le", "-ar", "32000", "-ac", "2", "-i", audio,
               "-frames:v", str(frames),
               "-vf", f"eq=gamma={args.gamma},ass={escaped(ass)}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.episode_crf),
               "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-shortest",
               "-movflags", "+faststart", out]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"episode: ffmpeg failed\n{result.stderr.strip()}")


def encode_escape(args: argparse.Namespace, frames_dir: str, first: int, out: str) -> None:
    """Encodes the slow motion beats, concatenated, silent.

    Silent on purpose. Stretching the game's audio by eight turns Mario's shout into a drone, and
    the point of this clip is a number changing by a factor of 1.48 nine times, which is read
    rather than heard. The run that follows it in the post carries the sound.

    Args:
        args: Parsed arguments.
        frames_dir: Directory of PPMs.
        first: Dump frame number replay frame 0 lands on.
        out: Path to write.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    rate = 30.0 / args.slow
    parts = []
    for index, (title, begin, end) in enumerate(BEATS):
        ass = telemetry(args, os.path.join(args.out_dir, f"escape{index}.ass"),
                        first + begin, first + end, rate)
        caption = os.path.join(args.out_dir, f"escape{index}.txt")
        with open(caption, "w", encoding="utf-8") as handle:
            handle.write(title)
        part = os.path.join(args.out_dir, f"escape{index}.mp4")
        # The title goes through textfile rather than text so that a beat can be worded in plain
        # English, commas and colons included, without being escaped past legibility.
        # Bottom right, because the readout libass burns in sits bottom left and the caption the
        # readout raises when Mario is inside the band sits top centre.
        drawtext = (f"drawtext=textfile={escaped(caption)}:x=w-text_w-22:y=h-50:fontsize=25"
                    f":fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10")
        # -frames:v bounds the OUTPUT, and the output runs at thirty while the input is read at a
        # rate slower by --slow, so the bound is the source frame count times the slow factor.
        # Bounding it by the source count instead would silently produce a clip of the right frame
        # count and the wrong length, which is how this first went wrong.
        command = [args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                   "-framerate", f"{rate:.6f}", "-start_number", str(first + begin),
                   "-i", os.path.join(frames_dir, "frame_%06d.ppm"),
                   "-frames:v", str((end - begin + 1) * args.slow),
                   "-vf", f"eq=gamma={args.gamma},ass={escaped(ass)},{drawtext}",
                   "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-crf", str(args.escape_crf), "-movflags", "+faststart", part]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"escape beat {index}: ffmpeg failed\n{result.stderr.strip()}")
        parts.append(part)

    listing = os.path.join(args.out_dir, "escape_parts.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        for part in parts:
            handle.write(f"file '{os.path.abspath(part)}'\n")
    result = subprocess.run([args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                             "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy",
                             "-movflags", "+faststart", out], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"escape: concat failed\n{result.stderr.strip()}")
    for path in [*parts, listing]:
        os.remove(path)


def poster(args: argparse.Namespace, frames_dir: str, frame: int, out: str) -> None:
    """Writes one frame as a jpg poster.

    Args:
        args: Parsed arguments.
        frames_dir: Directory of PPMs.
        frame: Dump frame number to use.
        out: Path to write.
    """
    subprocess.run([args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", os.path.join(frames_dir, f"frame_{frame:06d}.ppm"),
                    "-vf", f"eq=gamma={args.gamma}", "-q:v", "4", out], check=True)


def main() -> None:
    """Dumps the episode once and cuts both clips out of it."""
    args = parse_args()
    frames = episode_length(args.replay)
    first = args.inject_start - DUMP_LEADS_TIMER_BY
    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = args.frames_dir or os.path.join(args.out_dir, "frames")
    cont = os.path.join(args.replay_dir, "cont.m64")
    if not os.path.exists(cont):
        shutil.copyfile(args.tas, cont)

    print(f"episode: {frames} frames, replay frame 0 is dump frame {first}")
    audio = dump(args, frames_dir, first, frames)

    escape = os.path.join(args.out_dir, "escape.mp4")
    encode_escape(args, frames_dir, first, escape)
    # The crossing rather than the middle: the middle of the slow motion is the gap between the two
    # beats, which is a Mario standing still on a staircase.
    poster(args, frames_dir, first + BEATS[1][1] + 14, os.path.join(args.out_dir, "escape.jpg"))
    print(f"  escape.mp4   {os.path.getsize(escape) // 1024:>6} KB  "
          f"{sum(end - begin + 1 for _, begin, end in BEATS) * args.slow / 30.0:.1f}s, silent")

    episode = os.path.join(args.out_dir, "episode.mp4")
    encode_episode(args, frames_dir, audio, first, frames, episode)
    poster(args, frames_dir, first + BEATS[1][2], os.path.join(args.out_dir, "episode.jpg"))
    print(f"  episode.mp4  {os.path.getsize(episode) // 1024:>6} KB  {frames / 30.0:.1f}s, "
          f"with audio")

    for stray in os.listdir(args.out_dir):
        if stray.endswith((".ass", ".txt")):
            os.remove(os.path.join(args.out_dir, stray))
    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.remove(audio)


if __name__ == "__main__":
    main()
