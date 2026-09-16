"""Writes an ASS subtitle track that burns per frame policy telemetry onto a render.

``scripts/record_episode.py`` measures what the policy did and ``patches/`` teaches sm64-port to
stamp that measurement back over ``gMarioState`` and dump the result as frames, so the pixels in a
render are the real game drawing the real episode. What the pixels cannot say is *why* a frame
matters: a reader watching Mario slide backwards down a staircase has no way to see that his
forward velocity just multiplied by 1.47 for the eighth time in sixteen frames, or that he is
inside the warp band that is supposed to send him back down.

So the numbers go on the picture. This tool reads the same replay JSON the recorder wrote and
emits one ASS dialogue event per rendered frame, which ffmpeg's ``ass`` filter then burns in
through libass. Two reasons to go through a subtitle track rather than ffmpeg's ``drawtext``:
``drawtext`` re-reads and re-lays-out its text every frame, which for a 668 frame clip means 668
expression evaluations and a filter graph that is unreadable, while an ASS file is a plain list
this tool can write, diff and check. And libass gives real per event styling, which is what makes
the one visual decision here possible: the telemetry turns amber the moment ``|forward velocity|``
crosses the 154 units per frame the warp escape needs, so the threshold the whole project is about
is visible rather than asserted.

Two frame numberings meet here and the tool has to be told which one the reader sees. The injected
render begins by letting the 2016M TAS movie drive the controller so Mario is in the right room in
the right state, and only then takes over, so ``rendered_frame = --inject_start + replay_index``
and frames before the handover are labelled as the pre-roll they are. ``--first`` and ``--last``
are always rendered frames, because that is what a dump directory is named by. What the readout
*prints* is the replay frame by default, because the post's chapter marks seek the video by replay
frame and a readout counting the render's own frames would contradict them on screen.
"""

import argparse
import json
import sys
from typing import Any

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, \
BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: tel,{family},{size},&H00FFFFFF,&H00000000,&HB4000000,0,0,3,{pad},0,1,{left},20,{bottom},0
Style: hot,{family},{size},&H0060E0FF,&H00000000,&HB4000000,1,0,3,{pad},0,1,{left},20,{bottom},0
Style: cap,{family},{caption},&H00FFFFFF,&H00000000,&HC8202020,0,0,3,{pad},0,8,20,20,18,0

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text
"""

# The style block's sizes are quoted for a 667 pixel tall render, which is what the 1280x890
# injected dump becomes after its letterbox is cropped and it is scaled to 960 wide. Everything
# scales off that so a clip rendered at another height gets proportional text.
REFERENCE_HEIGHT = 667.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Builds the command line.

    Args:
        argv: Arguments without the program name.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", required=True,
                        help="Replay JSON scripts/record_episode.py wrote.")
    parser.add_argument("--out", required=True, help="ASS file to write.")
    parser.add_argument("--inject_start", type=int, default=6732,
                        help="Rendered frame the policy took the controller over on.")
    parser.add_argument("--first", type=int, required=True,
                        help="First rendered frame in the clip.")
    parser.add_argument("--last", type=int, required=True,
                        help="Last rendered frame in the clip.")
    parser.add_argument("--fps", type=float, default=30.0, help="Clip frame rate.")
    parser.add_argument("--width", type=int, default=960, help="Clip width in pixels.")
    parser.add_argument("--height", type=int, default=667, help="Clip height in pixels.")
    parser.add_argument("--escape_speed", type=float, default=154.0,
                        help="Speed at which the telemetry turns amber, in units per frame.")
    parser.add_argument("--warp_y", type=float, nargs=2, default=(3917.0, 3994.0),
                        help="Warp band's world y range, as min max.")
    parser.add_argument("--growth_floor", type=float, default=1.0,
                        help="Speed below which a frame to frame ratio is noise, not a pump.")
    parser.add_argument("--frame_label", choices=("replay", "rendered"), default="replay",
                        help="Which frame number the readout prints. The post's chapter marks seek "
                             "the video by replay frame, so the readout has to count the same way "
                             "or the two disagree on screen. 'rendered' prints the render's own "
                             "frame number instead, which is what a dump directory is named by.")
    return parser.parse_args(argv)


def stamp(seconds: float) -> str:
    """Formats a time as the ``h:mm:ss.cc`` string ASS expects.

    Args:
        seconds: Time from the start of the clip.

    Returns:
        The formatted timestamp.
    """
    hundredths = int(round(seconds * 100))
    hours, rest = divmod(hundredths, 360000)
    minutes, rest = divmod(rest, 6000)
    whole, cents = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{cents:02d}"


def build_events(frames: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    """Builds one dialogue event per rendered frame in the clip.

    Args:
        frames: The replay's per frame records.
        args: Parsed command line.

    Returns:
        The dialogue lines, in clip order.
    """
    low, high = args.warp_y
    events: list[str] = []
    for rendered in range(args.first, args.last + 1):
        index = rendered - args.inject_start
        start = stamp((rendered - args.first) / args.fps)
        end = stamp((rendered - args.first + 1) / args.fps)
        if index < 0:
            events.append(f"Dialogue: 0,{start},{end},tel,0,0,0,,"
                          r"{\an1}" f"frame {rendered}" r"\Nreplay: 2016M TAS (pre-roll)")
            continue
        if index >= len(frames):
            break
        number = index if args.frame_label == "replay" else rendered
        row = frames[index]
        speed = float(row["forward_velocity"])
        height = float(row["position"][1])
        previous = float(frames[index - 1]["forward_velocity"]) if index else 0.0
        growing = abs(previous) > args.growth_floor and abs(speed) / abs(previous) > 1.05
        ratio = rf"   \h\h×{abs(speed) / abs(previous):.3f}" if growing else ""
        style = "hot" if abs(speed) >= args.escape_speed else "tel"
        events.append(f"Dialogue: 0,{start},{end},{style},0,0,0,,"
                      r"{\an1}" f"frame {number}" r"\N" f"fwd vel {speed:9.2f}{ratio}"
                      r"\N" f"y{height:14.1f}")
        if low <= height <= high:
            events.append(f"Dialogue: 1,{start},{end},cap,0,0,0,,"
                          f"inside the warp band  y [{low:.0f}, {high:.0f}]")
    return events


def main(argv: list[str]) -> int:
    """Writes the track.

    Args:
        argv: Arguments without the program name.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    with open(args.replay, encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]
    scale = args.height / REFERENCE_HEIGHT
    header = HEADER.format(width=args.width, height=args.height, family="Andale Mono",
                           size=int(round(22 * scale)), caption=int(round(20 * scale)),
                           pad=int(round(6 * scale)), left=int(round(28 * scale)),
                           bottom=int(round(26 * scale)))
    events = build_events(frames, args)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join([header, *events]) + "\n")
    print(f"{args.out}: {len(events)} events, frames {args.first}..{args.last}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
