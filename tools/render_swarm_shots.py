"""Renders every swarm container in a manifest to an mp4 with the population's own audio.

``tools/export_swarm_render.py`` captures what a population did; this draws it. The patched
``sm64-port`` reads the container, hangs one graphics node per Mario off the object parent and lets
the game's own renderer draw all of them in a single pass, in the real endless staircase, with the
real model, lighting, shadows and depth. Nothing here composites separately rendered Marios, which
is why occlusion is right and why sixty four of them cost one render pass rather than sixty four.

Three environment settings make a shot reproducible rather than framed by hand:

* ``SM64_CAM`` locks the camera to one vantage, so four checkpoints of the same rung differ only in
  what the policies do. The default is behind the castle's bottom landing at hip height, looking up
  the flight, which is the one vantage that holds both a population still standing on the landing
  and a population already climbing past the warp.
* ``SM64_WARP_BAND`` paints the instant warp floors, discovered from the collision data itself, so
  the trap is on screen instead of being asserted in a caption.
* ``SM64_HIDE_DOORS`` drops star doors from the draw. The camera has to sit behind the landing to
  hold a whole population, and the 70 star door stands at z 3772, so without this the shot is a
  slab of wood. Only star doors: the movie that walks the game here opens ordinary and warp doors
  on the way, and hiding those desynchronises the route into the castle.
* ``SM64_FAST`` detaches the render from the display clock. The movie that walks the game to the
  staircase is nearly seven thousand frames, so at thirty frames a second every shot would spend
  four minutes rendering footage nobody keeps. Uncapped, a whole shot takes a few seconds.

The audio is muxed from the mp3 the capture wrote, not from the game running here. These Marios are
drawn rather than simulated, so this process is silent; the sound was recorded from libsm64 while
the population was actually playing, one mixer tick per simulated frame.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

# One vantage for every shot, from behind the bottom landing looking up the flight. Nine rounds of
# test renders mapped the room, and each of the four constraints below came out of a frame that was
# wrong in a way a caption could not fix:
#
# * The eye cannot be much above the floor. At y 3700, 526 above the landing's 3174, the crowd
#   standing on the landing is cut off by the bottom edge at panel size; y 3450 holds all of it.
# * The eye cannot come much closer than the landing's own back edge at z 3824. Inside the room the
#   nearest of the sixty four fill the lens, which is the note that sent me looking the first time.
# * Nor much further back than z 4100. Past that the near archway's lintel and posts cross the
#   frame, and past z 4400 the camera leaves the room entirely and back face culling shows the
#   castle exterior.
# * Which leaves a window of a few hundred units, all of it behind the 70 star door at z 3772. The
#   door is why this shot used to be stuck inside the room; SM64_HIDE_DOORS is why it no longer is.
#
# Filming from inside the flight would give a better picture of the trap, because the warp surfaces
# are floors and only a steep pitch shows their top faces. It is still wrong: a population that has
# learned nothing never leaves the landing, so the terminal rung's first two checkpoints would come
# out as fifteen seconds of an empty staircase, which at panel size reads as a broken file. So the
# band is a thin cyan line at the head of the flight here rather than a slab, which is the cost of
# never losing the population. The band is established unmistakably in the cold open, the slow
# motion escape and the hero run; what these panels have to show is where a crowd gets to.
DEFAULT_CAM = "-204,3450,3900,-204,3700,1800,55"
"""Eye xyz, target xyz, fov degrees. See the module docstring for why this one."""

# The frame dumper counts buffer swaps, and the swap it labels N carries the state the game
# simulated for gGlobalTimer N+1, so a swarm that starts at SM64_SWARM_START first appears in the
# dump one frame earlier than that. Measured, not assumed: with SM64_SWARM_START=6740 the first
# frame without a HUD and with sixty four Marios in it is frame_006739.
DUMP_LEADS_TIMER_BY = 1


def parse_args() -> argparse.Namespace:
    """Builds the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                        help="manifest.json written by tools/export_swarm_render.py.")
    parser.add_argument("--out_dir", required=True, help="Directory to write the mp4s into.")
    parser.add_argument("--work_dir", default="",
                        help="Scratch directory for the PPM dumps. Defaults to a shots subdir of "
                             "--out_dir, and is deleted per shot once the mp4 is encoded.")
    parser.add_argument("--only", action="append", default=[],
                        help="Render just this shot name. Repeatable.")
    parser.add_argument("--prefix", default="swarm-",
                        help="Filename prefix for the rendered shots.")
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
    parser.add_argument("--swarm_start", type=int, default=6740,
                        help="gGlobalTimer value the container's first frame applies at.")
    parser.add_argument("--cam", default=DEFAULT_CAM, help="SM64_CAM value.")
    parser.add_argument("--band", default="1", help="SM64_WARP_BAND value: 1, or r,g,b,a.")
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Gamma applied on encode. The game's framebuffer is much darker than "
                             "a display's, so a raw dump reads as an unlit room.")
    parser.add_argument("--crf", type=int, default=20, help="x264 quality.")
    parser.add_argument("--out_width", type=int, default=0,
                        help="Scale the encode to this width, 0 to keep the render size. A four "
                             "panel row gives each panel a couple of hundred CSS pixels, so the "
                             "panels are rendered large for the poster and encoded smaller.")
    parser.add_argument("--clip_frames", type=int, default=0,
                        help="Encode only the first N frames, 0 for all of them. The capture is "
                             "deliberately longer than the clip so the loop length is a decision "
                             "made here rather than one baked into every container.")
    parser.add_argument("--audio_bitrate", default="96k", help="AAC bitrate for the muxed audio.")
    parser.add_argument("--label", dest="label", action="store_true", default=True,
                        help="Burn the checkpoint label into the corner.")
    parser.add_argument("--no_label", dest="label", action="store_false",
                        help="Leave the frame clean and let the page caption it.")
    parser.add_argument("--poster", dest="poster", action="store_true", default=True,
                        help="Write a jpg poster from the middle of each shot.")
    parser.add_argument("--no_poster", dest="poster", action="store_false")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg", help="ffmpeg.")
    return parser.parse_args()


def shot_label(row: dict) -> str:
    """Renders the caption burned into a shot's corner.

    Args:
        row: A manifest shot row.

    Returns:
        The label, empty for the untrained population, which the page introduces in prose.
    """
    if row["steps"] is None:
        return ""
    return f"{row['rung'].replace('_', '+')}  {row['stepsLabel']} steps"


def dump_frames(args: argparse.Namespace, row: dict, frames_dir: str) -> int:
    """Runs the patched game once and leaves one PPM per frame of the shot.

    Args:
        args: Parsed arguments.
        row: A manifest shot row.
        frames_dir: Directory to dump into. Created, and expected to be empty.

    Returns:
        The frame number the dump starts at, which the encoder needs for its input pattern.

    Raises:
        RuntimeError: If the run wrote fewer frames than the container declares.
    """
    first = args.swarm_start - DUMP_LEADS_TIMER_BY
    last = first + row["frames"] - 1
    os.makedirs(frames_dir, exist_ok=True)
    root = os.path.abspath(os.curdir)
    environment = dict(os.environ)
    environment.update({
        "SM64_FAST": "1",
        "SM64_TAS_LAST": str(args.tas_last),
        "SM64_WINDOW_W": str(args.width),
        "SM64_WINDOW_H": str(args.height),
        "SM64_SWARM_FILE": os.path.join(root, os.path.dirname(args.manifest), row["container"]),
        "SM64_SWARM_START": str(args.swarm_start),
        "SM64_WARP_BAND": args.band,
        "SM64_CAM": args.cam,
        "SM64_HIDE_DOORS": "1",
        "SM64_DUMP_DIR": os.path.abspath(frames_dir),
        "SM64_DUMP_FIRST": str(first),
        "SM64_DUMP_LAST": str(last),
    })
    result = subprocess.run([os.path.join(root, args.game)], cwd=args.replay_dir, env=environment,
                            capture_output=True, text=True)
    written = len([name for name in os.listdir(frames_dir) if name.endswith(".ppm")])
    if written < row["frames"]:
        tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"{row['name']}: wrote {written} of {row['frames']} frames "
                           f"(exit {result.returncode})\n{tail}")
    for line in result.stderr.splitlines():
        if line.startswith(("swarm:", "warp band:", "camera:")):
            print(f"    {line}")
    return first


def clip_length(args: argparse.Namespace, row: dict) -> int:
    """Returns how many of a shot's frames the encode should keep.

    Args:
        args: Parsed arguments.
        row: A manifest shot row.

    Returns:
        The frame count, clamped to what the container holds.
    """
    if args.clip_frames <= 0:
        return row["frames"]
    return min(args.clip_frames, row["frames"])


def encode(args: argparse.Namespace, row: dict, frames_dir: str, first: int, out: str) -> None:
    """Muxes a shot's frames and audio into one mp4.

    Args:
        args: Parsed arguments.
        row: A manifest shot row.
        frames_dir: Directory of PPMs.
        first: Frame number the dump starts at.
        out: Path to write.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    label = shot_label(row) if args.label else ""
    filters = [f"eq=gamma={args.gamma}"]
    if args.out_width > 0:
        # Scaled after the gamma lift and before the label, so the label is drawn at its final
        # size and stays legible instead of being resampled along with the picture.
        filters.append(f"scale={args.out_width}:-2")
    if label:
        # Boxed rather than plain, because the staircase behind the label is red carpet in some
        # frames and a Mario's white glove in others, and unboxed text disappears into both.
        filters.append(f"drawtext=text='{label}':x=18:y=h-46:fontsize=24:fontcolor=white"
                       f":box=1:boxcolor=black@0.5:boxborderw=8")
    command = [args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", "30", "-start_number", str(first),
               "-i", os.path.join(frames_dir, "frame_%06d.ppm")]
    audio = row.get("audio")
    audio_path = os.path.join(os.path.dirname(args.manifest), audio) if audio else None
    if audio_path and os.path.exists(audio_path):
        command += ["-i", audio_path]
    command += ["-frames:v", str(clip_length(args, row)), "-vf", ",".join(filters),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
                "-movflags", "+faststart"]
    if audio_path and os.path.exists(audio_path):
        command += ["-c:a", "aac", "-b:a", args.audio_bitrate, "-ac", "2", "-shortest"]
    command.append(out)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{row['name']}: ffmpeg failed\n{result.stderr.strip()}")


def write_poster(args: argparse.Namespace, row: dict, frames_dir: str, first: int,
                 out: str) -> None:
    """Writes a poster frame from the middle of a shot.

    The middle rather than the first frame, because the first frame of a population that has
    learned the exploit is sixty four Marios standing still on the landing, which looks like the
    population that never learned anything.

    Args:
        args: Parsed arguments.
        row: A manifest shot row.
        frames_dir: Directory of PPMs.
        first: Frame number the dump starts at.
        out: Path to write.
    """
    middle = first + clip_length(args, row) // 2
    source = os.path.join(frames_dir, f"frame_{middle:06d}.ppm")
    filters = [f"eq=gamma={args.gamma}"]
    label = shot_label(row) if args.label else ""
    if label:
        filters.append(f"drawtext=text='{label}':x=18:y=h-46:fontsize=24:fontcolor=white"
                       f":box=1:boxcolor=black@0.5:boxborderw=8")
    subprocess.run([args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", source,
                    "-vf", ",".join(filters), "-q:v", "4", out], check=True)


def main() -> None:
    """Renders every shot the manifest names."""
    args = parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = manifest["shots"]
    if args.only:
        wanted = set(args.only)
        rows = [row for row in rows if row["name"] in wanted]
        missing = wanted - {row["name"] for row in rows}
        if missing:
            raise SystemExit(f"no such shot: {', '.join(sorted(missing))}")

    os.makedirs(args.out_dir, exist_ok=True)
    work_root = args.work_dir or os.path.join(args.out_dir, "frames")
    cont = os.path.join(args.replay_dir, "cont.m64")
    if not os.path.exists(cont):
        shutil.copyfile(args.tas, cont)

    print(f"{len(rows)} shots at {args.width}x{args.height}, camera {args.cam}")
    written = []
    for row in rows:
        started = time.perf_counter()
        name = f"{args.prefix}{row['name']}".replace("_", "-")
        frames_dir = os.path.join(work_root, row["name"])
        shutil.rmtree(frames_dir, ignore_errors=True)
        try:
            first = dump_frames(args, row, frames_dir)
            out = os.path.join(args.out_dir, f"{name}.mp4")
            encode(args, row, frames_dir, first, out)
            poster = None
            if args.poster:
                poster = os.path.join(args.out_dir, f"{name}.jpg")
                write_poster(args, row, frames_dir, first, poster)
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)
        size = os.path.getsize(out) // 1024
        written.append({"shot": row["name"], "video": os.path.basename(out),
                        "poster": os.path.basename(poster) if poster else None,
                        "frames": clip_length(args, row),
                        "seconds": round(clip_length(args, row) / 30.0, 2),
                        "kilobytes": size})
        print(f"  {name:<28} {size:>6} KB  {time.perf_counter() - started:>5.1f}s")

    shutil.rmtree(work_root, ignore_errors=True)
    index = os.path.join(args.out_dir, "rendered.json")
    with open(index, "w", encoding="utf-8") as handle:
        json.dump({"camera": args.cam, "renderWidth": args.width, "renderHeight": args.height,
                   "encodeWidth": args.out_width or args.width, "frameRate": 30.0,
                   "gamma": args.gamma, "crf": args.crf, "shots": written}, handle, indent=1)
        handle.write("\n")
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
