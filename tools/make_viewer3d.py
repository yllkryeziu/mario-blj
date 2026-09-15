"""Builds the single file 3D replay viewer that shows Mario's real mesh on the real staircase.

The 2D viewer draws a dot on a side elevation. This one draws the actual model libsm64 hands back
every tick, walking the decompilation's display lists with Mario's own bone matrices baked in, on
the actual collision triangles of the castle's endless staircase. Everything the page needs is
inlined: the geometry container as base64, the 704 by 64 texture atlas as a PNG inside it, and a
hand written WebGL renderer with no external script of any kind.

Two things this tool does that ``tools/dump_mesh.py`` cannot. First, it captures the staircase run
with ``check_instant_warp`` in the loop, which libsm64 does not implement and which the runaway
depends on: without it Mario coasts off the top of the area after a couple of cycles and the chain
stalls near the air attractor at a peak forwardVel of about -32, while with it he is thrown back
down into the treads 37 times and the chain reaches -1543.88 before he finally crosses the whole
trigger zone inside one frame. Second, it accepts containers that ``dump_mesh.py`` already wrote,
so any scene that tool can dump can be dropped into the same page as an extra tab.

The container bytes are never rewritten. Each scene is injected as a base64 copy of the exact
``mblj-mesh/1`` blob plus a small JSON meta block for the things that live outside the format,
such as the warp zone, the goal and which frames the warp fired on. The page parses the container
itself, so ``tools/dump_mesh.py`` stays the single source of truth for the layout.

Sizes, measured on the real staircase at stride 1: about 6.1 KB of base64 per kept frame, so the
1120 frame run lands near 7 MB and the whole page near 7.3 MB of the 16 MB budget.
"""

import base64
import dataclasses
import json
import os
import struct
import sys
from array import array
from typing import TYPE_CHECKING, Any, cast

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from absl import app, flags, logging

from src.agent.scripted import ScriptedBlj, calibrate_stick, stick_toward
from src.env.endless_stairs import apply_instant_warp, load_scene, minimum_escape_speed
from src.env.geometry import ground_plane
from src.env.native import TEXTURE_HEIGHT, TEXTURE_WIDTH, Sm64
from tools.dump_mesh import (
    MAGIC,
    DumpConfig,
    FrameRecord,
    SceneSpec,
    encode_png,
    geometry_view,
    pack,
    quantise_color,
    quantise_normals,
    quantise_positions,
    triangle_colors,
)

if TYPE_CHECKING:
    from src.agent.drivers import Driver

_AREA_2_COLLISION = os.path.join(_ROOT, "third_party", "sm64-port", "levels", "castle_inside",
                                 "areas", "2", "collision.inc.c")

PLACEHOLDER = "__REPLAY_DATA__"
AUDIO_PLACEHOLDER = "__AUDIO_DATA__"
PAGE_BUDGET = 16_000_000

SCENE_LABELS = {
    "endless_stairs": ("The endless staircase", "the real castle, rise 25.6 run 51.2, warp live"),
    "stairs": ("Synthetic stairs, 36.9°", "rise 75, run 100, no warp"),
    "ramp": ("Ramp, 30°", "uniform slope"),
    "flat": ("Flat ground", "no rise at all"),
}

FLAGS = flags.FLAGS


def _define(definer: Any, name: str, default: Any, help_text: str, **kwargs: Any) -> None:
    """Defines a flag unless importing tools.dump_mesh already defined it.

    tools.dump_mesh is imported for its container encoder and registers its own command line as a
    side effect, so the names this tool shares with it are already present. Redefining one raises
    DuplicateFlagError, and silently inheriting dump_mesh's default would be worse, so the shared
    names keep this tool's default via set_default.

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


_define(flags.DEFINE_string, "rom", os.path.join(_ROOT, "roms", "baserom.us.z64"),
        "Super Mario 64 US ROM that libsm64 reads animation and texture data from.")
_define(flags.DEFINE_string, "template", os.path.join(_ROOT, "tools", "viewer3d_template.html"),
        "Template holding the __REPLAY_DATA__ placeholder.")
_define(flags.DEFINE_string, "out", os.path.join(_ROOT, "results", "viewer3d.html"),
        "Page to write.")
_define(flags.DEFINE_integer, "frames", 1120, "Frames to simulate on the endless staircase.",
        lower_bound=1)
_define(flags.DEFINE_integer, "stride", 1, "Keep every Nth simulated frame.", lower_bound=1)
_define(flags.DEFINE_integer, "max_frames", None, "Cap on kept frames. Unlimited when unset.",
        lower_bound=1)
_define(flags.DEFINE_float, "position_scale", 4.0,
        "Quantisation steps per world unit for Mario's vertices.", lower_bound=0.0625)
_define(flags.DEFINE_string, "approach", None,
        "Stick direction to walk downhill in. Calibrated toward +z when unset.")
_define(flags.DEFINE_list, "spawn", None,
        "Spawn as x,y,z. Defaults to the endless staircase's own bottom landing spawn.")
_define(flags.DEFINE_string, "collision_path", None,
        "castle_inside/areas/2/collision.inc.c. Defaults to the module's own path.")
_define(flags.DEFINE_string, "surface_header", None,
        "surface_terrains.h supplying the SURFACE_* constants.")
_define(flags.DEFINE_string, "audio", None,
        "Audio file recorded by scripts/record_episode.py --audio, inlined as a data URI and "
        "synced to the frame scrubber.")
_define(flags.DEFINE_string, "replay", None,
        "Replay JSON from scripts/record_episode.py. Its recorded per frame controller state is "
        "fed back through the environment, which reproduces that episode exactly and keeps the "
        "page independent of any policy sampling order.")
_define(flags.DEFINE_string, "model", None,
        "stable-baselines3 model zip to drive Mario with. The scripted expert drives when unset.")
_define(flags.DEFINE_boolean, "stochastic", False,
        "Sample from the policy rather than taking its argmax action.")
_define(flags.DEFINE_integer, "policy_seed", 0, "Seeds the policy's sampling.")
_define(flags.DEFINE_boolean, "capture", True,
        "Capture the endless staircase run. Turn off to build a page out of --container scenes "
        "alone.")
_define(flags.DEFINE_multi_string, "container", [],
        "Extra scene as name=path, where path is a container that tools/dump_mesh.py wrote in "
        "mesh mode.")


@dataclasses.dataclass(frozen=True)
class ViewerConfig:
    """Everything the page build needs, resolved from flags.

    Attributes:
        rom_path: Path to the Super Mario 64 US ROM.
        template_path: Template carrying the __REPLAY_DATA__ placeholder.
        out_path: Page to write.
        frames: Frames to simulate on the endless staircase.
        stride: Keep every Nth simulated frame.
        max_frames: Cap on kept frames, or None for no cap.
        position_scale: Quantisation steps per world unit for Mario's vertices.
        approach: Stick direction to walk downhill in, or None to calibrate toward +z.
        spawn: Spawn override, or None for the staircase's own bottom landing spawn.
        collision_path: Override for the area's collision.inc.c, or None for the default.
        surface_header: Override for surface_terrains.h, or None for the default.
        capture: Whether to capture the endless staircase run.
        containers: Extra scenes as (name, container path) pairs.
        model_path: Trained policy to drive Mario with, or None for the scripted expert.
        stochastic: Sample from the policy rather than taking its argmax action.
        policy_seed: Seeds the policy's sampling.
    """

    rom_path: str
    template_path: str
    out_path: str
    frames: int
    stride: int
    max_frames: int | None
    position_scale: float
    approach: str | None
    spawn: tuple[float, float, float] | None
    collision_path: str | None
    surface_header: str | None
    capture: bool
    containers: tuple[tuple[str, str], ...]
    replay_path: str | None = None
    model_path: str | None = None
    stochastic: bool = False
    policy_seed: int = 0


@dataclasses.dataclass(frozen=True)
class Capture:
    """One captured scene, ready to pack.

    Attributes:
        config: The DumpConfig that pack needs to describe the run.
        collected: The same dict shape tools.dump_mesh.run returns.
        meta: Scene facts that live outside the container format, such as the warp zone and
            which kept frames the warp fired on.
    """

    config: DumpConfig
    collected: dict[str, Any]
    meta: dict[str, Any]


def resolve_scene_kwargs(config: ViewerConfig) -> dict[str, Any]:
    """Builds the keyword arguments for src.env.endless_stairs.load_scene.

    Args:
        config: Resolved viewer configuration.

    Returns:
        Only the overrides the user actually asked for, so the module's own defaults stand.
    """
    kwargs: dict[str, Any] = {}
    if config.collision_path is not None:
        kwargs["collision_path"] = config.collision_path
    if config.surface_header is not None:
        kwargs["header_path"] = config.surface_header
    if config.spawn is not None:
        kwargs["spawn"] = config.spawn
    return kwargs


def capture_endless_stairs(config: ViewerConfig) -> Capture:
    """Runs the scripted BLJ on the real endless staircase with the instant warp in the loop.

    The geometry buffers libsm64 writes during a tick belong to Mario's pre-warp position, so the
    warp check runs after the frame has been recorded. That keeps every frame's vertices and its
    recorded origin in the same space and makes the displacement visible as the jump it is.

    Args:
        config: Resolved viewer configuration.

    Returns:
        The capture, with a DumpConfig that tools.dump_mesh.pack accepts.

    Raises:
        RuntimeError: If libsm64 changes its triangle count mid run, which the page's single
            static vertex buffer would not survive.
    """
    scene = load_scene(**resolve_scene_kwargs(config))
    environment = None
    observation = None
    info: dict = {"warps": 0}
    drive: Driver | None = None
    if config.replay_path:
        from src.agent.drivers import action_index
        from src.env.blj_env import BljConfig, BljEnv, RewardConfig

        with open(config.replay_path, encoding="utf-8") as handle:
            recorded = json.load(handle)["frames"]
        queue = [action_index(row["inputs"]["stick_x"], row["inputs"]["stick_y"],
                              bool(row["inputs"]["a"]), bool(row["inputs"]["z"]))
                 for row in recorded]
        logging.info("replaying %d recorded frames from %s", len(queue), config.replay_path)

        environment = BljEnv(BljConfig(
            rom_path=config.rom_path,
            collision_path=config.collision_path or _AREA_2_COLLISION,
            reward=RewardConfig(terminal=1.0),
            max_frames=max(len(queue), config.frames)))
        step_counter = {"i": 0}

        def replay_drive(observation: Any, info: dict[str, Any]) -> int:
            """Returns the next recorded action, holding the last one if the queue runs dry.

            Args:
                observation: Ignored, the queue is fixed before the run starts.
                info: Ignored.

            Returns:
                The action index recorded for this step.
            """
            del observation, info
            index = min(step_counter["i"], len(queue) - 1)
            step_counter["i"] += 1
            return queue[index]

        drive = replay_drive
        game = environment.game
    elif config.model_path:
        from src.agent.drivers import model_driver
        from src.env.blj_env import BljConfig, BljEnv, RewardConfig

        environment = BljEnv(BljConfig(
            rom_path=config.rom_path,
            collision_path=config.collision_path or _AREA_2_COLLISION,
            reward=RewardConfig(terminal=1.0),
            max_frames=config.frames))
        drive = model_driver(config.model_path, deterministic=not config.stochastic,
                             seed=config.policy_seed)
        game = environment.game
    else:
        game = Sm64(config.rom_path)
    try:
        atlas = encode_png(TEXTURE_WIDTH, TEXTURE_HEIGHT, bytes(game._texture))
        if config.approach is None:
            table = calibrate_stick(game, ground_plane(8000.0), (0.0, 100.0, 0.0))
            approach = stick_toward(table, 0.0, 1.0)
        else:
            approach = config.approach
        logging.info("approach stick %s, spawn %s, %d collision triangles",
                     approach, scene.spawn, len(scene.surfaces))

        view = geometry_view(game)
        policy = ScriptedBlj(approach)
        # Calibration above loads a flat plane to walk on, and when the environment owns this
        # libsm64 handle that plane replaces the staircase, so the surfaces go back first.
        if environment is None:
            game.load_surfaces(scene.surfaces)
            game.create_mario(*scene.spawn)
        else:
            game.load_surfaces(environment.scene.surfaces)
            observation, info = environment.reset(seed=0)
        finished = False

        palette: dict[tuple[int, int, int], int] = {}
        color_variants: list[array] = []
        color_keys: dict[bytes, int] = {}
        uv_variants: list[array] = []
        uv_keys: dict[bytes, int] = {}
        records: list[FrameRecord] = []
        warped: list[int] = []
        counts: set[int] = set()
        worst_error = 0.0
        warp_count = 0
        top_frame: int | None = None

        for index in range(config.frames):
            if environment is None:
                state = game.tick(policy.inputs())
                policy.observe(state.action, state.forwardVelocity)
                fired_now = None
            else:
                before = info["warps"]
                observation, _, terminated, truncated, info = environment.step(
                    cast("Driver", drive)(observation, info))
                state = game.state
                fired_now = info["warps"] > before
                finished = terminated or truncated
            origin = (state.position[0], state.position[1], state.position[2])
            keep = index % config.stride == 0 and (
                config.max_frames is None or len(records) < config.max_frames)

            if keep:
                count = view.triangle_count
                counts.add(count)
                if len(counts) > 1:
                    raise RuntimeError(f"triangle count changed mid run: {sorted(counts)}")

                indices = array("B", bytes(count))
                for triangle, rgb in enumerate(triangle_colors(view, count)):
                    key = quantise_color(rgb)
                    if key not in palette:
                        palette[key] = len(palette)
                    indices[triangle] = palette[key]
                color_variant = color_keys.setdefault(indices.tobytes(), len(color_variants))
                if color_variant == len(color_variants):
                    color_variants.append(indices)

                uv = array("f", (view.uv[i] for i in range(6 * count)))
                uv_variant = uv_keys.setdefault(uv.tobytes(), len(uv_variants))
                if uv_variant == len(uv_variants):
                    uv_variants.append(uv)

                positions, error = quantise_positions(view, count, origin, config.position_scale)
                worst_error = max(worst_error, error)
                records.append(FrameRecord(
                    index=index, triangle_count=count, position=origin,
                    face_angle=state.faceAngle, forward_velocity=state.forwardVelocity,
                    action=state.action, anim_id=state.animID, anim_frame=state.animFrame,
                    color_variant=color_variant, uv_variant=uv_variant,
                    positions=positions, normals=quantise_normals(view, count)))

            if fired_now is None:
                fired = apply_instant_warp(game, game.extra_state().floorType, origin, scene.warp)
            else:
                fired = fired_now
            warp_count += int(fired)
            if keep:
                warped.append(int(fired))
            if top_frame is None and origin[1] >= scene.goal_y and origin[2] <= scene.goal_z:
                top_frame = max(0, len(records) - 1)
            if config.max_frames is not None and len(records) >= config.max_frames:
                break
            if finished:
                break
    finally:
        if environment is not None:
            environment.close()
        else:
            game.close()

    dump_config = DumpConfig(
        rom_path=config.rom_path,
        scene=SceneSpec("endless_stairs", scene.surfaces, scene.spawn),
        frames=config.frames, stride=config.stride, max_frames=config.max_frames, mode="mesh",
        position_scale=config.position_scale, approach=approach, include_surfaces=True,
        out_path=config.out_path)
    collected = {
        "records": records,
        "palette": [list(rgb) for rgb, _ in sorted(palette.items(), key=lambda kv: kv[1])],
        "color_variants": color_variants,
        "uv_variants": uv_variants,
        "reference": None,
        "atlas": atlas,
        "worst_error": worst_error,
        "triangle_counts": sorted(counts),
    }
    meta = {
        "warped": warped,
        "warp_count": warp_count,
        "warp": {
            "surface_type": scene.warp.surface_type,
            "displacement": list(scene.warp.displacement),
            "x_range": list(scene.warp.x_range),
            "y_range": list(scene.warp.y_range),
            "z_range": list(scene.warp.z_range),
            "depth": scene.warp.depth,
            "minimum_escape_speed": minimum_escape_speed(scene.warp),
        },
        "goal_y": scene.goal_y,
        "goal_z": scene.goal_z,
        "ascends_toward": list(scene.ascends_toward),
        "top_frame": top_frame,
        "success": top_frame is not None,
        "approach_stick": approach,
    }
    return Capture(config=dump_config, collected=collected, meta=meta)


def open_container(blob: bytes) -> dict[str, Any]:
    """Reads the JSON header out of a mblj-mesh/1 container.

    Args:
        blob: The whole container as written by tools.dump_mesh.pack.

    Returns:
        The decoded header.

    Raises:
        ValueError: If the magic is wrong or the container is not in mesh mode, which the page's
            renderer requires because it uploads a fresh vertex buffer every frame.
    """
    if blob[:len(MAGIC)] != MAGIC:
        raise ValueError(f"not a {MAGIC.decode()} container")
    length = struct.unpack("<I", blob[len(MAGIC):len(MAGIC) + 4])[0]
    header = json.loads(blob[len(MAGIC) + 4:len(MAGIC) + 4 + length])
    if header["mode"] != "mesh":
        raise ValueError(f"container is {header['mode']!r}; the 3D page needs mesh mode")
    missing = [name for name in ("positions", "normals", "transforms", "uv", "color_index",
                                 "atlas_png", "surface_vertices", "surface_types")
               if name not in header["streams"]]
    if missing:
        raise ValueError(f"container is missing the streams {missing}; dump it with --surfaces")
    return header


def scene_payload(name: str, blob: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Wraps one container plus its out-of-format meta into the page's per-scene payload.

    Args:
        name: Scene key the page switches on.
        blob: The container bytes, passed through untouched.
        meta: Facts that live outside the container format.

    Returns:
        The payload dict, with the container base64 encoded.
    """
    header = open_container(blob)
    velocities = [row[header["frame_fields"].index("forward_velocity")]
                  for row in header["frames"]]
    label, note = SCENE_LABELS.get(name, (name, ""))
    return {
        "name": name,
        "label": label,
        "note": note,
        "frame_count": header["frame_count"],
        "triangle_count": header["triangle_counts"][0],
        "surface_count": header["streams"]["surface_types"]["shape"][0],
        "stride": header["stride"],
        "peak_velocity": min(velocities) if velocities else 0.0,
        "bytes": len(blob),
        "meta": meta,
        "container": base64.b64encode(blob).decode("ascii"),
    }


def build_page(template: str, payload: dict[str, Any]) -> str:
    """Injects the replay payload into the template.

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
    filled = template.replace(PLACEHOLDER, json.dumps(payload, separators=(",", ":")))
    return filled.replace(AUDIO_PLACEHOLDER, json.dumps(audio_payload(FLAGS.audio),
                                                        separators=(",", ":")))


def audio_payload(path: str | None) -> dict | None:
    """Reads an audio file into a data URI payload the page can play.

    Args:
        path: Audio file written by scripts/record_episode.py, or None for a silent page.

    Returns:
        A dict with the data URI and the duration in seconds, or None.

    Raises:
        ValueError: If the file extension is not one a browser will decode.
    """
    if not path:
        return None
    media = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg"}
    extension = os.path.splitext(path)[1].lower()
    if extension not in media:
        raise ValueError(f"unsupported audio extension {extension!r}")
    with open(path, "rb") as handle:
        raw = handle.read()
    seconds = None
    if extension == ".wav":
        import wave
        with wave.open(path, "rb") as handle:
            seconds = handle.getnframes() / handle.getframerate()
    logging.info("audio %s, %d bytes", path, len(raw))
    return {
        "src": f"data:{media[extension]};base64,{base64.b64encode(raw).decode('ascii')}",
        "bytes": len(raw),
        "seconds": seconds if seconds is not None else 0.0,
    }


def config_from_flags() -> ViewerConfig:
    """Resolves the command line flags into a ViewerConfig.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If --spawn is not three numbers, or a --container entry is not name=path.
    """
    if FLAGS.spawn is None:
        spawn = None
    else:
        if len(FLAGS.spawn) != 3:
            raise ValueError("--spawn needs exactly three numbers")
        values = [float(value) for value in FLAGS.spawn]
        spawn = (values[0], values[1], values[2])
    containers = []
    for entry in FLAGS.container:
        if "=" not in entry:
            raise ValueError(f"--container wants name=path, got {entry!r}")
        name, path = entry.split("=", 1)
        containers.append((name, path))
    return ViewerConfig(
        rom_path=FLAGS.rom, template_path=FLAGS.template, out_path=FLAGS.out,
        frames=FLAGS.frames, stride=FLAGS.stride, max_frames=FLAGS.max_frames,
        position_scale=FLAGS.position_scale, approach=FLAGS.approach, spawn=spawn,
        collision_path=FLAGS.collision_path, surface_header=FLAGS.surface_header,
        capture=FLAGS.capture, containers=tuple(containers), replay_path=FLAGS.replay,
        model_path=FLAGS.model, stochastic=FLAGS.stochastic, policy_seed=FLAGS.policy_seed)


def main(argv: list[str]) -> None:
    """Captures the staircase run, builds the page and reports its real size.

    Args:
        argv: Unparsed command line arguments.

    Raises:
        app.UsageError: If positional arguments were given, or no scene was asked for.
    """
    if len(argv) > 1:
        raise app.UsageError(f"unexpected arguments: {argv[1:]}")
    config = config_from_flags()
    if not config.capture and not config.containers:
        raise app.UsageError("nothing to build: pass --container or leave --capture on")

    scenes: list[dict[str, Any]] = []
    if config.capture:
        capture = capture_endless_stairs(config)
        blob = pack(capture.config, capture.collected)
        scenes.append(scene_payload("endless_stairs", blob, capture.meta))
        records = capture.collected["records"]
        peak = min(record.forward_velocity for record in records)
        logging.info("endless staircase: %d frames kept, %d warps, peak forwardVel %.2f, "
                     "top reached %s", len(records), capture.meta["warp_count"], peak,
                     capture.meta["top_frame"])
        logging.info("worst position quantisation error %.4f units",
                     capture.collected["worst_error"])

    for name, path in config.containers:
        with open(path, "rb") as handle:
            scenes.append(scene_payload(name, handle.read(), {}))

    payload = {
        "order": [scene["name"] for scene in scenes],
        "scenes": {scene["name"]: scene for scene in scenes},
    }
    with open(config.template_path, encoding="utf-8") as handle:
        template = handle.read()
    page = build_page(template, payload)
    os.makedirs(os.path.dirname(os.path.abspath(config.out_path)), exist_ok=True)
    with open(config.out_path, "w", encoding="utf-8") as handle:
        handle.write(page)

    size = os.path.getsize(config.out_path)
    print(f"wrote {config.out_path}")
    for scene in scenes:
        print(f"  {scene['name']:<16s} {scene['frame_count']:>5d} frames  "
              f"{scene['triangle_count']:>4d} mario tris  "
              f"{scene['surface_count']:>5d} level tris  "
              f"peak {scene['peak_velocity']:>9.2f}  "
              f"{len(scene['container']) / 1e6:.3f} MB base64")
    print(f"  page           {size:>10d} B  {size / 1e6:.3f} MB  "
          f"({size / PAGE_BUDGET * 100:.1f}% of the {PAGE_BUDGET / 1e6:.0f} MB budget)")
    if size > PAGE_BUDGET:
        logging.warning("page is over the %d byte budget by %d bytes; raise --stride",
                        PAGE_BUDGET, size - PAGE_BUDGET)


if __name__ == "__main__":
    app.run(main)
