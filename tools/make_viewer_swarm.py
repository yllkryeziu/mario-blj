"""Builds the single file swarm viewer, a room full of trained Marios on the real staircase.

``tools/make_viewer3d.py`` shows one Mario at a time and spends about 6 KB of base64 on every
frame of him, which is the honest price of keeping his deformed mesh. A swarm cannot pay that
price: forty eight bodies over eight hundred frames would want hundreds of megabytes and the page
has sixteen. ``tools/dump_swarm.py`` solves it with a pose dictionary, so this tool's whole job is
to wrap the container that tool wrote in a page that can draw it, to attach the audio that tool
captured, and to say plainly how much of the budget the result costs.

The page is one HTML file with no external script of any kind. The container goes in as base64 and
is parsed in the browser, so ``tools/dump_swarm.py`` stays the single source of truth for the
layout and this tool never rewrites a byte of it. Everything the viewer needs already lives in the
container header: the checkpoints and how each one did, the warp zone, the goal height, the pose
dictionary and the staircase's own collision triangles.

Audio is attached the same way. ``tools/dump_swarm.py --audio_dir DIR`` writes one
``swarm_<name>.mp3`` per checkpoint, and this tool looks in that same directory for a track whose
name matches each checkpoint. A missing track is not an error; the page disables its sound toggle
for that checkpoint and says so.

Two ways to run it. By default it reads a container that ``tools/dump_swarm.py`` already wrote,
which is fast and needs neither the ROM nor a GPU. With ``--capture`` it runs the capture first,
using ``tools/dump_swarm.py``'s own flags, and writes the container to ``--swarm`` on the way
through.

Sizes, measured on the six checkpoint container at population 48 over 800 frames: 2.97 MB of
container becomes 3.96 MB of base64, six mp3s of 321 KB each become 2.57 MB more, and the page
lands at 6.68 MB, which is 42 percent of the budget. The pose dictionary is 82 percent of the
container and it is shared by every checkpoint, so a seventh checkpoint costs about 170 KB rather
than another megabyte.
"""

import base64
import dataclasses
import json
import os
import struct
import sys
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from absl import app, flags, logging

from tools.dump_swarm import (
    FORMAT_VERSION,
    MAGIC,
    PAGE_BUDGET,
    config_from_flags,
    pack,
    run,
)

PLACEHOLDER = "__REPLAY_DATA__"

GAME_FPS = 30.0

FLAGS = flags.FLAGS


def _define(definer: Any, name: str, default: Any, help_text: str, **kwargs: Any) -> None:
    """Defines a flag unless importing tools.dump_swarm already defined it.

    tools.dump_swarm is imported for its capture and its container reader, and registers its own
    command line as a side effect, so the names this tool shares with it are already present.
    Redefining one raises DuplicateFlagError, and silently inheriting the other tool's default
    would be worse, so the shared names keep this tool's default via set_default.

    Args:
        definer: One of the absl DEFINE_ functions.
        name: Flag name.
        default: This tool's default.
        help_text: Flag help.
        **kwargs: Passed through to the definer.
    """
    if name in FLAGS:
        FLAGS.set_default(name, default)
        return
    definer(name, default, help_text, **kwargs)


_define(flags.DEFINE_string, "swarm", os.path.join(_ROOT, "results", "swarm.bin"),
        "Container that tools/dump_swarm.py wrote, and where --capture writes one.")
_define(flags.DEFINE_string, "template", os.path.join(_ROOT, "tools",
                                                      "viewer_swarm_template.html"),
        "Template holding the __REPLAY_DATA__ placeholder.")
_define(flags.DEFINE_string, "page", os.path.join(_ROOT, "results", "viewer_swarm.html"),
        "Page to write.")
_define(flags.DEFINE_boolean, "capture", False,
        "Run the capture first with tools/dump_swarm.py's flags, writing --swarm on the way. "
        "Off by default, which reads the container that is already there.")
_define(flags.DEFINE_string, "audio_dir", None,
        "Directory tools/dump_swarm.py wrote its per checkpoint mp3s into, and which --capture "
        "writes them to. The tracks are inlined as base64 so the page stays one file. None "
        "leaves the page silent and disables its sound toggle.")


@dataclasses.dataclass(frozen=True)
class ViewerConfig:
    """Everything the page build needs, resolved from flags.

    Attributes:
        container_path: Container to read, and to write when capturing.
        template_path: Template carrying the __REPLAY_DATA__ placeholder.
        page_path: Page to write.
        capture: Whether to run the capture before building the page.
        audio_dir: Directory holding one mp3 per checkpoint, or None for a silent page.
    """

    container_path: str
    template_path: str
    page_path: str
    capture: bool
    audio_dir: str | None


def capture_container(container_path: str) -> bytes:
    """Runs tools.dump_swarm's capture and returns the container bytes.

    Args:
        container_path: Where the container is written.

    Returns:
        The packed container.
    """
    config = dataclasses.replace(config_from_flags(), out_path=container_path)
    logging.info("capturing %d checkpoints at population %d over %d frames",
                 len(config.checkpoints), config.population, config.frames)
    blob = pack(config, run(config))
    os.makedirs(os.path.dirname(os.path.abspath(container_path)), exist_ok=True)
    with open(container_path, "wb") as handle:
        handle.write(blob)
    return blob


def describe(header: dict[str, Any]) -> list[str]:
    """Summarises a container header as the lines this tool prints.

    Args:
        header: Parsed container header.

    Returns:
        One line per checkpoint, in container order.
    """
    lines = []
    for spec in header["checkpoints"]:
        outcome = spec["outcome"]
        lines.append(f"  {spec['name']:<18s} {spec['steps_label']:>6s} steps  "
                     f"{spec['frames']:>5d} frames  "
                     f"{outcome['successes']:>2d}/{outcome['episodes']:<3d} reached the top  "
                     f"best y {outcome['best_height']:>7.0f}  "
                     f"peak forwardVel {outcome['best_peak_backward']:>10.2f}  "
                     f"{spec['warps']:>4d} warps")
    return lines


AUDIO_MIME = "audio/mpeg"

AUDIO_PREFIX = "swarm_"


def audio_path(audio_dir: str, name: str) -> str:
    """Names the mp3 tools/dump_swarm.py writes for one checkpoint.

    Args:
        audio_dir: Directory the capture wrote its tracks into.
        name: Checkpoint name as the container header spells it.

    Returns:
        The path the track would be at, whether or not it exists.
    """
    return os.path.join(audio_dir, f"{AUDIO_PREFIX}{name}.mp3")


def collect_audio(header: dict[str, Any], audio_dir: str | None) -> list[dict[str, Any] | None]:
    """Reads one audio track per checkpoint, in container order.

    The capture is the only thing that knows how to render the population's sound, so this does
    not synthesise anything. It looks for the file that capture would have written and inlines it
    when it is there. A checkpoint with no track gets None, which the page reads as silence for
    that checkpoint rather than as an error.

    Args:
        header: Parsed container header.
        audio_dir: Directory the capture wrote its tracks into, or None to stay silent.

    Returns:
        One entry per checkpoint, each either None or a dict carrying the base64 track.
    """
    if not audio_dir:
        return [None for _ in header["checkpoints"]]
    seconds = header["frames"] * max(1, header["stride"]) / GAME_FPS
    tracks: list[dict[str, Any] | None] = []
    for spec in header["checkpoints"]:
        path = audio_path(audio_dir, spec["name"])
        if not os.path.exists(path):
            logging.warning("no audio for checkpoint %s at %s", spec["name"], path)
            tracks.append(None)
            continue
        with open(path, "rb") as handle:
            raw = handle.read()
        tracks.append({
            "name": spec["name"],
            "mime": AUDIO_MIME,
            "bytes": len(raw),
            "seconds": round(seconds, 3),
            "data": base64.b64encode(raw).decode("ascii"),
        })
    return tracks


def build_payload(blob: bytes, source: str,
                  audio_dir: str | None = None) -> dict[str, Any]:
    """Wraps the container in the object the page's script reads.

    The container goes in untouched. The page parses its header itself, so the payload carries
    only the things that are true of this build rather than of the capture, plus whatever audio
    the capture left on disk.

    Args:
        blob: The container bytes.
        source: Where the container came from, for the page to name.
        audio_dir: Directory holding one mp3 per checkpoint, or None for a silent page.

    Returns:
        The payload dict, with the container and every audio track base64 encoded.

    Raises:
        ValueError: If the blob is not a container this page can read.
    """
    header, _ = open_container(blob)
    if header["format"] != FORMAT_VERSION:
        raise ValueError(f"container is {header['format']!r}, the page reads {FORMAT_VERSION!r}")
    return {
        "format": header["format"],
        "source": os.path.basename(source),
        "bytes": len(blob),
        "checkpoint_count": len(header["checkpoints"]),
        "container": base64.b64encode(blob).decode("ascii"),
        "audio": collect_audio(header, audio_dir),
    }


def open_container(blob: bytes) -> tuple[dict[str, Any], bytes]:
    """Parses a container that is already in memory.

    tools.dump_swarm.read_container does the same job for a path. The capture path already holds
    the bytes, and reading them back off disk to describe them would be silly, so this works on
    the blob.

    Args:
        blob: The whole container.

    Returns:
        A pair of the parsed header and the payload bytes.

    Raises:
        ValueError: If the magic does not match.
    """
    if blob[:len(MAGIC)] != MAGIC:
        raise ValueError(f"not a {MAGIC.decode()} container")
    length = struct.unpack("<I", blob[len(MAGIC):len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    return json.loads(blob[start:start + length]), blob[start + length:]


def build_page(template: str, payload: dict[str, Any]) -> str:
    """Injects the payload into the template.

    Args:
        template: Template text containing exactly one __REPLAY_DATA__ placeholder.
        payload: The object the page's script reads.

    Returns:
        The finished page.

    Raises:
        ValueError: If the template does not carry exactly one placeholder.
    """
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"template needs exactly one {PLACEHOLDER}, "
                         f"found {template.count(PLACEHOLDER)}")
    return template.replace(PLACEHOLDER, json.dumps(payload, separators=(",", ":")))


def config_from_viewer_flags() -> ViewerConfig:
    """Resolves the command line flags into a ViewerConfig.

    Returns:
        The resolved configuration.
    """
    return ViewerConfig(container_path=FLAGS.swarm, template_path=FLAGS.template,
                        page_path=FLAGS.page, capture=FLAGS.capture, audio_dir=FLAGS.audio_dir)


def main(argv: list[str]) -> None:
    """Builds the swarm page and reports its real size.

    Args:
        argv: Unparsed command line arguments.

    Raises:
        app.UsageError: If positional arguments were given, or the container is missing and the
            run was not asked to capture one.
    """
    if len(argv) > 1:
        raise app.UsageError(f"unexpected arguments: {argv[1:]}")
    config = config_from_viewer_flags()
    if config.capture:
        blob = capture_container(config.container_path)
    else:
        if not os.path.exists(config.container_path):
            raise app.UsageError(
                f"no container at {config.container_path}: run tools/dump_swarm.py or pass "
                "--capture with its flags")
        with open(config.container_path, "rb") as handle:
            blob = handle.read()

    payload = build_payload(blob, config.container_path, config.audio_dir)
    header, _ = open_container(blob)
    with open(config.template_path, encoding="utf-8") as handle:
        template = handle.read()
    page = build_page(template, payload)
    os.makedirs(os.path.dirname(os.path.abspath(config.page_path)), exist_ok=True)
    with open(config.page_path, "w", encoding="utf-8") as handle:
        handle.write(page)

    tracks = [track for track in payload["audio"] if track]
    audio_raw = sum(track["bytes"] for track in tracks)
    audio_encoded = sum(len(track["data"]) for track in tracks)
    size = os.path.getsize(config.page_path)
    print(f"wrote {config.page_path}")
    print(f"  {header['scene']}, {header['population']} Marios, {header['frames']} frames, "
          f"{header['pose_count']} poses, {len(header['checkpoints'])} checkpoints")
    for line in describe(header):
        print(line)
    print(f"  container      {len(blob):>10d} B  {len(blob) / 1e6:.3f} MB")
    print(f"  base64         {len(payload['container']):>10d} B  "
          f"{len(payload['container']) / 1e6:.3f} MB")
    print(f"  audio          {audio_raw:>10d} B  {audio_raw / 1e6:.3f} MB  "
          f"{len(tracks)} of {len(payload['audio'])} checkpoints")
    print(f"  audio base64   {audio_encoded:>10d} B  {audio_encoded / 1e6:.3f} MB")
    print(f"  page           {size:>10d} B  {size / 1e6:.3f} MB  "
          f"({size / PAGE_BUDGET * 100:.1f}% of the {PAGE_BUDGET / 1e6:.0f} MB budget)")
    if size > PAGE_BUDGET:
        logging.warning("page is over the %d byte budget by %d bytes; raise dump_swarm's "
                        "--stride, drop --audio_dir or drop a checkpoint", PAGE_BUDGET,
                        size - PAGE_BUDGET)


if __name__ == "__main__":
    app.run(main)
