"""Dumps Mario's animated geometry out of libsm64 into a compact container for a 3D viewer.

libsm64 hands back a fresh triangle soup every tick. There is no skeleton and no index buffer on
the other side of the API, only the flattened result of walking the decompilation's display lists
with Mario's current bone matrices already baked in. That makes the geometry trivial to consume
and expensive to store, so this tool measures what is actually redundant and strips it.

What the substrate gives us, measured on a 700 frame scripted BLJ on the synthetic staircase:

  * 752 triangles every single frame, across 32 distinct animation ids. The count never moves.
  * Positions are 9 floats per triangle, laid out as three vertices of xyz, in the same world
    space as the surfaces loaded through sm64_static_surfaces_load.
  * Normals are per vertex, unit length, 9 floats per triangle.
  * Colors are flat per triangle and the whole 6768 float array is byte identical on every frame,
    so it collapses to a six entry palette plus one index per triangle.
  * UVs only take three distinct values over a whole run, one per eye state, so they collapse to
    a small variant table plus one index per frame.

Two encodings come out of that. Mesh mode keeps every frame's deformed geometry, quantised to
int16 relative to Mario's own position, delta coded against the previous frame and deflated.
Transform-only mode keeps a single reference pose in Mario local space plus a position and a
face angle per frame, which is two orders of magnitude smaller but renders a rigid Mario sliding
through the scene rather than an animated one.

Container layout, little endian throughout:

    magic           9 bytes, b"MBLJMESH1"
    header_length   uint32
    header          header_length bytes of UTF-8 JSON
    payload         the streams named in header["streams"], back to back

Every stream entry carries an offset relative to the start of the payload, a byte length, a dtype
and a codec. The codecs are:

    raw               the bytes as they are
    deflate           zlib.compress of the bytes
    delta16+deflate   int16 samples, frame 0 absolute and every later frame stored as the
                      wrapping difference against the frame before it, then zlib.compress
    delta8+deflate    the same scheme on int8 samples

To rebuild frame i in mesh mode, inflate the positions stream, undo the deltas, then for every
component take ``world = frame_position[axis] + quantised / header["position_scale"]``.

To rebuild frame i in transform-only mode, take the reference pose, which is stored in Mario
local space, and map it with the face angle yaw of that frame:

    world_x = local_x * cos(yaw) + local_z * sin(yaw) + position_x
    world_y = local_y                                 + position_y
    world_z = -local_x * sin(yaw) + local_z * cos(yaw) + position_z

Triangles are wound counter-clockwise when seen from the front face, in the same right handed
sense the collision loader uses for its upward floor normals, so positions can go straight into a
WebGL front-face-CCW pipeline with no winding or axis flip.
"""

import ctypes
import dataclasses
import json
import math
import os
import struct
import sys
import zlib
from array import array
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from absl import app, flags, logging

from src.agent.scripted import ScriptedBlj, calibrate_stick, stick_toward
from src.env.collision import parse_collision, surface_constants
from src.env.geometry import flat_area, ground_plane, ramp, staircase
from src.env.native import GEO_MAX_TRIANGLES, TEXTURE_HEIGHT, TEXTURE_WIDTH, Sm64, action_name

MAGIC = b"MBLJMESH1"
FORMAT_VERSION = "mblj-mesh/1"

ATLAS_SLOT_WIDTH = 64
ATLAS_SLOT_COUNT = 11
ATLAS_SLOT_NAMES = ("metal", "yellow_button", "m_logo", "hair_sideburn", "mustache", "eyes_front",
                    "eyes_half_closed", "eyes_closed", "eyes_dead", "wings_half_1",
                    "wings_half_2")
ATLAS_SLOT_WIDTHS = (64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32)
ATLAS_SLOT_HEIGHTS = (32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64)
ATLAS_SLOT_WRAP = ("repeat",) + ("clamp",) * 10

UNTEXTURED_SENTINEL = 1.0

APRON = flat_area(-3000.0, 3000.0, -12000.0, 0.0)
DEFAULT_SPAWN = (0.0, 100.0, -2000.0)

FLAGS = flags.FLAGS

flags.DEFINE_string("rom", os.path.join(_ROOT, "roms", "baserom.us.z64"),
                    "Super Mario 64 US ROM that libsm64 reads animation and texture data from.")
flags.DEFINE_enum("scene", "stairs", ["stairs", "flat", "ramp", "collision"],
                  "Scene to run the scripted BLJ on. 'collision' parses --collision_path.")
flags.DEFINE_string("collision_path", None,
                    "collision.inc.c to parse when --scene=collision.")
flags.DEFINE_string("surface_header", os.path.join(_ROOT, "third_party", "sm64-port", "include",
                                                   "surface_terrains.h"),
                    "Header that supplies the SURFACE_* constants for --scene=collision.")
flags.DEFINE_list("spawn", None,
                  "Spawn as x,y,z. Defaults to the synthetic scene spawn (0, 100, -2000).")
flags.DEFINE_integer("frames", 700, "Frames to simulate.", lower_bound=1)
flags.DEFINE_integer("stride", 1, "Keep every Nth simulated frame.", lower_bound=1)
flags.DEFINE_integer("max_frames", None,
                     "Stop once this many frames have been kept. Unlimited when unset.",
                     lower_bound=1)
flags.DEFINE_string("replay", None,
                    "Replay JSON from scripts/record_episode.py. With --driver=replay its "
                    "recorded per frame controller state is fed back through the environment, "
                    "which reproduces that episode exactly.")
flags.DEFINE_enum("driver", "chain", ["chain", "env", "replay"],
                  "chain drives the raw scripted policy with no level logic. env drives "
                  "src.env.blj_env.BljEnv, which reimplements the staircase's instant warp, so "
                  "the loop throws Mario back and the speed chain has to actually run away.")
flags.DEFINE_enum("mode", "mesh", ["mesh", "transform-only"],
                  "'mesh' keeps every kept frame's deformed geometry. 'transform-only' keeps one "
                  "reference pose plus a position and face angle per frame.")
flags.DEFINE_float("position_scale", 4.0,
                   "Quantisation steps per world unit for positions. 4 gives a quarter unit grid "
                   "and a worst case error of an eighth of a unit.", lower_bound=0.0625)
flags.DEFINE_string("approach", None,
                    "Stick direction to walk in. Calibrated toward -z when unset.")
flags.DEFINE_boolean("surfaces", True, "Embed the scene's collision triangles in the container.")
flags.DEFINE_string("out", None,
                    "Output path. Defaults to results/mesh_<scene>_<mode>.bin.")


@dataclasses.dataclass(frozen=True)
class SceneSpec:
    """A loaded scene plus the spawn the scripted policy should start from.

    Attributes:
        name: Human readable scene name recorded in the container header.
        surfaces: Collision triangles to hand to sm64_static_surfaces_load.
        spawn: World space spawn position as (x, y, z).
        collision_path: Source of the collision data when the scene came from a level file.
    """

    name: str
    surfaces: list[Any]
    spawn: tuple[float, float, float]
    collision_path: str | None = None


@dataclasses.dataclass(frozen=True)
class DumpConfig:
    """Everything the dump needs, resolved from flags.

    Attributes:
        rom_path: Path to the Super Mario 64 US ROM.
        scene: Scene to simulate on.
        frames: Number of frames to simulate.
        stride: Keep every Nth simulated frame.
        max_frames: Cap on kept frames, or None for no cap.
        mode: Either "mesh" or "transform-only".
        position_scale: Quantisation steps per world unit for positions.
        approach: Stick direction to walk in, or None to calibrate toward -z.
        include_surfaces: Whether to embed the scene's collision triangles.
        out_path: Where to write the container.
        driver: Either "chain" for the raw scripted policy or "env" for the environment, which
            adds the staircase's instant warp loop.
    """

    rom_path: str
    scene: SceneSpec
    frames: int
    stride: int
    max_frames: int | None
    mode: str
    position_scale: float
    approach: str | None
    include_surfaces: bool
    out_path: str
    driver: str = "chain"
    replay_path: str | None = None


@dataclasses.dataclass(frozen=True)
class GeometryView:
    """Direct views onto the geometry buffers that src.env.native allocates for libsm64.

    The Sm64 binding keeps its SM64MarioGeometryBuffers private and returns only the mario state
    from tick, so this tool reaches through the struct to read the four arrays that libsm64 wrote
    during the same tick. The views are stable for the lifetime of the Sm64 instance.

    Attributes:
        buffers: The SM64MarioGeometryBuffers struct the binding passes to sm64_mario_tick.
        position: Nine floats per triangle, three vertices of xyz in world space.
        normal: Nine floats per triangle, one unit normal per vertex.
        color: Nine floats per triangle, the part's light color repeated on all three vertices.
        uv: Six floats per triangle, one atlas uv per vertex.
    """

    buffers: Any
    position: Any
    normal: Any
    color: Any
    uv: Any

    @property
    def triangle_count(self) -> int:
        """Returns the triangle count libsm64 wrote on the most recent tick."""
        return int(self.buffers.numTrianglesUsed)


@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """One kept frame of the run.

    Attributes:
        index: Index of the frame within the simulated run.
        triangle_count: Triangles libsm64 emitted on this frame.
        position: Mario's world position as (x, y, z).
        face_angle: Mario's yaw in radians as libsm64 reports it.
        forward_velocity: Mario's forwardVel.
        action: Raw action bitfield.
        anim_id: Animation id.
        anim_frame: Frame within the animation.
        color_variant: Index into the container's color index variant table.
        uv_variant: Index into the container's uv variant table.
        positions: Quantised mario-relative int16 positions, or None in transform-only mode.
        normals: Quantised int8 normals, or None in transform-only mode.
    """

    index: int
    triangle_count: int
    position: tuple[float, float, float]
    face_angle: float
    forward_velocity: float
    action: int
    anim_id: int
    anim_frame: int
    color_variant: int
    uv_variant: int
    positions: array | None
    normals: array | None


def build_scene(scene: str, collision_path: str | None, surface_header: str,
                spawn: tuple[float, float, float]) -> SceneSpec:
    """Builds the requested scene.

    Args:
        scene: One of "stairs", "flat", "ramp" or "collision".
        collision_path: collision.inc.c to parse when scene is "collision".
        surface_header: Header supplying the SURFACE_* constants for the collision parser.
        spawn: World space spawn position.

    Returns:
        The loaded SceneSpec.

    Raises:
        ValueError: If scene is "collision" and collision_path is missing.
    """
    if scene == "flat":
        return SceneSpec("flat", list(APRON), spawn)
    if scene == "stairs":
        return SceneSpec("stairs", APRON + staircase(400, 75.0, 100.0, 6000.0), spawn)
    if scene == "ramp":
        return SceneSpec("ramp", APRON + ramp(24000.0, 6000.0, 30.0), spawn)
    if collision_path is None:
        raise ValueError("--scene=collision needs --collision_path")
    constants = surface_constants(surface_header)
    return SceneSpec(os.path.basename(collision_path),
                     parse_collision(collision_path, constants), spawn,
                     collision_path=collision_path)


def geometry_view(game: Sm64) -> GeometryView:
    """Opens views onto a running Sm64 instance's geometry buffers.

    Args:
        game: An initialised Sm64 binding.

    Returns:
        A GeometryView over the binding's position, normal, color and uv arrays.
    """
    buffers = game._geometry
    floats = ctypes.c_float * (9 * GEO_MAX_TRIANGLES)
    uv_floats = ctypes.c_float * (6 * GEO_MAX_TRIANGLES)
    return GeometryView(
        buffers=buffers,
        position=ctypes.cast(buffers.position, ctypes.POINTER(floats)).contents,
        normal=ctypes.cast(buffers.normal, ctypes.POINTER(floats)).contents,
        color=ctypes.cast(buffers.color, ctypes.POINTER(floats)).contents,
        uv=ctypes.cast(buffers.uv, ctypes.POINTER(uv_floats)).contents)


def encode_png(width: int, height: int, rgba: bytes) -> bytes:
    """Encodes an RGBA8 image as a PNG using only zlib.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        rgba: width * height * 4 bytes of top-down RGBA8 samples.

    Returns:
        The PNG file bytes.

    Raises:
        ValueError: If rgba is not width * height * 4 bytes long.
    """
    if len(rgba) != width * height * 4:
        raise ValueError(f"expected {width * height * 4} bytes of RGBA, got {len(rgba)}")
    stride = width * 4
    raw = bytearray()
    for y in range(height):
        raw += b"\x00"
        raw += rgba[y * stride:(y + 1) * stride]

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def atlas_slots() -> list[dict[str, Any]]:
    """Describes how the 704 by 64 Mario texture atlas is carved up.

    sm64_global_init blits eleven Mario textures into a row of 64 pixel wide slots and leaves the
    unused part of each slot fully transparent. gfx_adapter's convert_uv_to_atlas maps a texture
    local uv into slot ``i`` as ``u_atlas = u_local * width / 64 / 11 + i / 11`` and
    ``v_atlas = v_local * height / 64``, so each slot's valid sub-rectangle is smaller than the
    slot itself for every texture that is not 64 by 64.

    Returns:
        One dict per slot with its name, pixel size, atlas sub-rectangle and wrap mode.
    """
    slots = []
    for index in range(ATLAS_SLOT_COUNT):
        width = ATLAS_SLOT_WIDTHS[index]
        height = ATLAS_SLOT_HEIGHTS[index]
        slots.append({
            "index": index,
            "name": ATLAS_SLOT_NAMES[index],
            "pixel_width": width,
            "pixel_height": height,
            "u0": index / ATLAS_SLOT_COUNT,
            "u1": index / ATLAS_SLOT_COUNT + width / ATLAS_SLOT_WIDTH / ATLAS_SLOT_COUNT,
            "v0": 0.0,
            "v1": height / ATLAS_SLOT_WIDTH,
            "wrap": ATLAS_SLOT_WRAP[index],
        })
    return slots


def triangle_colors(view: GeometryView, count: int) -> list[tuple[float, float, float]]:
    """Reads one flat color per triangle.

    Args:
        view: Views onto the geometry buffers.
        count: Triangles written on the current frame.

    Returns:
        One (r, g, b) triple per triangle, each component in [0, 1].
    """
    color = view.color
    return [(color[9 * t], color[9 * t + 1], color[9 * t + 2]) for t in range(count)]


def quantise_positions(view: GeometryView, count: int, origin: tuple[float, float, float],
                       scale: float) -> tuple[array, float]:
    """Quantises world space positions to int16 relative to Mario's own position.

    Args:
        view: Views onto the geometry buffers.
        count: Triangles written on the current frame.
        origin: Mario's world position on this frame.
        scale: Quantisation steps per world unit.

    Returns:
        A pair of the int16 samples and the worst absolute quantisation error in world units.
    """
    source = view.position
    out = array("h", bytes(2 * 9 * count))
    worst = 0.0
    for component in range(9 * count):
        value = source[component] - origin[component % 3]
        step = int(round(value * scale))
        worst = max(worst, abs(step / scale - value))
        out[component] = max(-32768, min(32767, step))
    return out, worst


def quantise_normals(view: GeometryView, count: int) -> array:
    """Quantises unit normals to int8.

    Args:
        view: Views onto the geometry buffers.
        count: Triangles written on the current frame.

    Returns:
        The int8 samples, 9 per triangle, to be divided by 127 on the other side.
    """
    source = view.normal
    out = array("b", bytes(9 * count))
    for component in range(9 * count):
        out[component] = max(-127, min(127, int(round(source[component] * 127.0))))
    return out


def to_local_space(view: GeometryView, count: int, origin: tuple[float, float, float],
                   yaw: float) -> tuple[array, array]:
    """Removes Mario's position and yaw from one frame to make a reference pose.

    Args:
        view: Views onto the geometry buffers.
        count: Triangles written on the current frame.
        origin: Mario's world position on this frame.
        yaw: Mario's face angle in radians on this frame.

    Returns:
        A pair of float32 local space positions and float32 local space normals.
    """
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    positions = array("f", bytes(4 * 9 * count))
    normals = array("f", bytes(4 * 9 * count))
    for vertex in range(3 * count):
        base = 3 * vertex
        x = view.position[base] - origin[0]
        y = view.position[base + 1] - origin[1]
        z = view.position[base + 2] - origin[2]
        positions[base] = x * cos_yaw - z * sin_yaw
        positions[base + 1] = y
        positions[base + 2] = x * sin_yaw + z * cos_yaw
        nx, ny, nz = view.normal[base], view.normal[base + 1], view.normal[base + 2]
        normals[base] = nx * cos_yaw - nz * sin_yaw
        normals[base + 1] = ny
        normals[base + 2] = nx * sin_yaw + nz * cos_yaw
    return positions, normals


def quantise_color(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    """Turns one triangle's float color into the palette key.

    The palette is a dict keyed on the byte triple, so the rounding has to happen before the
    lookup and has to be exact: two triangles that quantise to the same bytes must land on the
    same entry, which is what collapses Mario's 752 triangles onto a six colour palette.

    Args:
        rgb: One triangle's color, each component nominally in [0, 1].

    Returns:
        The three components scaled to [0, 255], clamped rather than wrapped.
    """
    red, green, blue = (max(0, min(255, int(round(channel * 255.0)))) for channel in rgb)
    return red, green, blue


def mesh_frames(frames: list[array | None], stream: str) -> list[array]:
    """Asserts that every kept frame carries its geometry, and hands the frames back.

    FrameRecord holds positions and normals as optional because transform-only mode keeps a single
    reference pose instead of per frame geometry. Mesh mode fills both on every kept frame, so
    this states that invariant where the streams are built rather than letting a missing frame
    reach delta_encode as an AttributeError.

    Args:
        frames: One array per kept frame, possibly unset.
        stream: Name of the stream being built, for the error message.

    Returns:
        The same frames in the same order.

    Raises:
        ValueError: If any frame is missing its array, which would mean mesh mode skipped one.
    """
    if any(frame is None for frame in frames):
        raise ValueError(f"mesh mode left {stream} unset on at least one frame")
    return [frame for frame in frames if frame is not None]


def delta_encode(frames: list[array], typecode: str) -> bytes:
    """Delta codes a list of equal length integer frames against their predecessor.

    A frame whose length differs from the previous one is stored absolutely, which keeps the
    scheme correct if libsm64 ever changes its triangle count mid run.

    Args:
        frames: One integer array per frame.
        typecode: Either "h" for int16 or "b" for int8.

    Returns:
        The concatenated delta coded bytes.

    Raises:
        ValueError: If typecode is not "h" or "b".
    """
    if typecode not in ("h", "b"):
        raise ValueError(f"unsupported typecode {typecode!r}")
    modulus = 65536 if typecode == "h" else 256
    half = modulus // 2
    out = bytearray()
    previous: array | None = None
    for current in frames:
        if previous is None or len(previous) != len(current):
            out += current.tobytes()
        else:
            deltas = array(typecode, bytes(current.itemsize * len(current)))
            for i in range(len(current)):
                deltas[i] = ((current[i] - previous[i] + half) % modulus) - half
            out += deltas.tobytes()
        previous = current
    return bytes(out)


class StreamTable:
    """Accumulates named payload streams and the header entries that describe them."""

    def __init__(self) -> None:
        """Initialises an empty table."""
        self._payload = bytearray()
        self._entries: dict[str, dict[str, Any]] = {}

    def add(self, name: str, data: bytes, dtype: str, codec: str,
            shape: list[int] | None = None) -> None:
        """Appends one stream.

        Args:
            name: Stream name referenced from the header.
            data: The already encoded bytes.
            dtype: Element type on the decoded side, such as "int16" or "float32".
            codec: One of "raw", "deflate", "delta16+deflate" or "delta8+deflate".
            shape: Logical shape of the decoded data, for the reader's benefit.
        """
        self._entries[name] = {
            "offset": len(self._payload),
            "length": len(data),
            "dtype": dtype,
            "codec": codec,
            "shape": shape or [],
        }
        self._payload += data

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        """Returns the header entries describing every stream added so far."""
        return self._entries

    @property
    def payload(self) -> bytes:
        """Returns the concatenated payload bytes."""
        return bytes(self._payload)


def run(config: DumpConfig) -> dict[str, Any]:
    """Runs the scripted BLJ and collects every kept frame's geometry.

    Args:
        config: Resolved dump configuration.

    Returns:
        A dict with the frame records, the variant tables, the color palette, the texture atlas
        PNG and the worst position quantisation error observed.
    """
    environment = None
    queue: list[int] = []
    if config.driver in ("env", "replay"):
        from src.agent.drivers import action_index, scripted_driver
        from src.env.blj_env import BljConfig, BljEnv, RewardConfig

        if config.driver == "replay":
            if config.replay_path is None:
                raise ValueError("--driver replay needs --replay")
            with open(config.replay_path, encoding="utf-8") as handle:
                queue = [action_index(row["inputs"]["stick_x"], row["inputs"]["stick_y"],
                                      bool(row["inputs"]["a"]), bool(row["inputs"]["z"]))
                         for row in json.load(handle)["frames"]]
            logging.info("replaying %d recorded frames", len(queue))
        environment = BljEnv(BljConfig(
            rom_path=config.rom_path,
            collision_path=config.scene.collision_path or _DEFAULT_AREA_2,
            reward=RewardConfig(terminal=1.0),
            max_frames=max(len(queue), config.frames)))
        game = environment.game
    else:
        game = Sm64(config.rom_path)
    try:
        atlas = encode_png(TEXTURE_WIDTH, TEXTURE_HEIGHT, bytes(game._texture))
        if config.approach is None:
            table = calibrate_stick(game, ground_plane(8000.0), (0.0, 100.0, 0.0))
            approach = stick_toward(table, 0.0, -1.0)
        else:
            approach = config.approach
        logging.info("approach stick: %s", approach)

        view = geometry_view(game)
        if environment is None:
            policy = ScriptedBlj(approach)
            game.load_surfaces(config.scene.surfaces)
            game.create_mario(*config.scene.spawn)
        else:
            if queue:
                cursor = {"i": 0}

                def drive(observation, info):
                    """Returns the next recorded action, holding the last once exhausted."""
                    del observation, info
                    index = min(cursor["i"], len(queue) - 1)
                    cursor["i"] += 1
                    return queue[index]
            else:
                drive = scripted_driver(approach)
            game.load_surfaces(environment.scene.surfaces)
            observation, info = environment.reset(seed=0)

        palette: dict[tuple[int, int, int], int] = {}
        color_variants: list[array] = []
        color_keys: dict[bytes, int] = {}
        uv_variants: list[array] = []
        uv_keys: dict[bytes, int] = {}
        records: list[FrameRecord] = []
        reference: tuple[array, array] | None = None
        worst_error = 0.0
        counts: set[int] = set()

        for index in range(config.frames):
            if environment is None:
                inputs = policy.inputs()
                state = game.tick(inputs)
                policy.observe(state.action, state.forwardVelocity)
                finished = False
            else:
                action = drive(observation, info)
                observation, _, terminated, truncated, info = environment.step(action)
                state = game.state
                finished = terminated or truncated
            if index % config.stride:
                if finished:
                    break
                continue
            if config.max_frames is not None and len(records) >= config.max_frames:
                break

            count = view.triangle_count
            counts.add(count)
            origin = (state.position[0], state.position[1], state.position[2])

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

            positions: array | None = None
            normals: array | None = None
            if config.mode == "mesh":
                positions, error = quantise_positions(view, count, origin, config.position_scale)
                worst_error = max(worst_error, error)
                normals = quantise_normals(view, count)
            elif reference is None:
                reference = to_local_space(view, count, origin, state.faceAngle)

            records.append(FrameRecord(
                index=index, triangle_count=count, position=origin,
                face_angle=state.faceAngle, forward_velocity=state.forwardVelocity,
                action=state.action, anim_id=state.animID, anim_frame=state.animFrame,
                color_variant=color_variant, uv_variant=uv_variant,
                positions=positions, normals=normals))
            if finished:
                break
    finally:
        if environment is not None:
            environment.close()
        else:
            game.close()

    return {
        "records": records,
        "palette": [list(rgb) for rgb, _ in sorted(palette.items(), key=lambda kv: kv[1])],
        "color_variants": color_variants,
        "uv_variants": uv_variants,
        "reference": reference,
        "atlas": atlas,
        "worst_error": worst_error,
        "triangle_counts": sorted(counts),
    }


def pack(config: DumpConfig, collected: dict[str, Any]) -> bytes:
    """Packs a collected run into the container bytes.

    Args:
        config: Resolved dump configuration.
        collected: The dict returned by run.

    Returns:
        The full container, ready to write to disk.

    Raises:
        ValueError: If transform-only mode collected no reference pose.
    """
    records: list[FrameRecord] = collected["records"]
    streams = StreamTable()
    streams.add("atlas_png", collected["atlas"], "uint8", "raw",
                [TEXTURE_HEIGHT, TEXTURE_WIDTH, 4])
    streams.add("color_index", zlib.compress(
        b"".join(variant.tobytes() for variant in collected["color_variants"]), 9),
        "uint8", "deflate", [len(collected["color_variants"]), -1])
    streams.add("uv", zlib.compress(
        b"".join(variant.tobytes() for variant in collected["uv_variants"]), 9),
        "float32", "deflate", [len(collected["uv_variants"]), -1, 3, 2])
    streams.add("transforms", zlib.compress(b"".join(
        struct.pack("<4f", *record.position, record.face_angle) for record in records), 9),
        "float32", "deflate", [len(records), 4])

    if config.mode == "mesh":
        streams.add("positions",
                    zlib.compress(delta_encode(
                        mesh_frames([r.positions for r in records], "positions"), "h"), 9),
                    "int16", "delta16+deflate", [len(records), -1, 3, 3])
        streams.add("normals",
                    zlib.compress(delta_encode(
                        mesh_frames([r.normals for r in records], "normals"), "b"), 9),
                    "int8", "delta8+deflate", [len(records), -1, 3, 3])
    else:
        if collected["reference"] is None:
            raise ValueError("transform-only mode collected no frames to take a reference from")
        positions, normals = collected["reference"]
        streams.add("reference_positions", zlib.compress(positions.tobytes(), 9),
                    "float32", "deflate", [-1, 3, 3])
        streams.add("reference_normals", zlib.compress(normals.tobytes(), 9),
                    "float32", "deflate", [-1, 3, 3])

    if config.include_surfaces:
        vertices = array("i", (int(surface.vertices[i][axis])
                               for surface in config.scene.surfaces
                               for i in range(3) for axis in range(3)))
        types = array("h", (int(surface.type) for surface in config.scene.surfaces))
        streams.add("surface_vertices", zlib.compress(vertices.tobytes(), 9), "int32", "deflate",
                    [len(config.scene.surfaces), 3, 3])
        streams.add("surface_types", zlib.compress(types.tobytes(), 9), "int16", "deflate",
                    [len(config.scene.surfaces)])

    header: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "mode": config.mode,
        "scene": config.scene.name,
        "spawn": list(config.scene.spawn),
        "rom": os.path.basename(config.rom_path),
        "simulated_frames": config.frames,
        "stride": config.stride,
        "frame_count": len(records),
        "triangle_counts": collected["triangle_counts"],
        "position_scale": config.position_scale,
        "position_quantisation_error": collected["worst_error"],
        "normal_scale": 127.0,
        "coordinate_system": "sm64 world space, y up, right handed, one unit per sm64 unit, "
                             "same space as the loaded collision surfaces",
        "winding": "counter-clockwise front facing",
        "mario_origin": "state position sits at Mario's feet; the mesh spans about y+0 to y+161 "
                        "and about 75 units either side in x and z",
        "untextured_uv_sentinel": UNTEXTURED_SENTINEL,
        "untextured_note": "a triangle whose three uvs are all exactly 1.0 was emitted with "
                           "texturing off; shade it from the vertex color alone, since uv 1,1 "
                           "lands on a fully transparent atlas texel",
        "local_to_world": "wx = lx*cos(yaw) + lz*sin(yaw) + px; wy = ly + py; "
                          "wz = -lx*sin(yaw) + lz*cos(yaw) + pz",
        "atlas": {
            "width": TEXTURE_WIDTH,
            "height": TEXTURE_HEIGHT,
            "slot_width": ATLAS_SLOT_WIDTH,
            "slot_count": ATLAS_SLOT_COUNT,
            "slots": atlas_slots(),
            "sampling_note": "clamp each uv into its slot sub-rectangle; plain clamp-to-edge on "
                             "the whole atlas bleeds into the transparent remainder of the slot",
        },
        "color_palette": collected["palette"],
        "color_variant_count": len(collected["color_variants"]),
        "uv_variant_count": len(collected["uv_variants"]),
        "frame_fields": ["frame", "triangle_count", "action", "anim_id", "anim_frame",
                         "color_variant", "uv_variant", "forward_velocity"],
        "frames": [[r.index, r.triangle_count, r.action, r.anim_id, r.anim_frame,
                    r.color_variant, r.uv_variant, round(r.forward_velocity, 3)]
                   for r in records],
        "action_names": {str(action): action_name(action)
                         for action in sorted({r.action for r in records})},
        "streams": streams.entries,
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(blob)) + blob + streams.payload


_DEFAULT_AREA_2 = os.path.join(_ROOT, "third_party", "sm64-port", "levels", "castle_inside",
                               "areas", "2", "collision.inc.c")


def config_from_flags() -> DumpConfig:
    """Resolves the command line flags into a DumpConfig.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If --spawn is given but is not three numbers.
    """
    if FLAGS.spawn is None:
        spawn = DEFAULT_SPAWN
    else:
        if len(FLAGS.spawn) != 3:
            raise ValueError("--spawn needs exactly three numbers")
        values = [float(value) for value in FLAGS.spawn]
        spawn = (values[0], values[1], values[2])
    scene = build_scene(FLAGS.scene, FLAGS.collision_path, FLAGS.surface_header, spawn)
    out = FLAGS.out or os.path.join(_ROOT, "results", f"mesh_{scene.name}_{FLAGS.mode}.bin")
    return DumpConfig(
        rom_path=FLAGS.rom, scene=scene, frames=FLAGS.frames, stride=FLAGS.stride,
        max_frames=FLAGS.max_frames, mode=FLAGS.mode, position_scale=FLAGS.position_scale,
        approach=FLAGS.approach, include_surfaces=FLAGS.surfaces, out_path=out,
        driver=FLAGS.driver, replay_path=FLAGS.replay)


def main(argv: list[str]) -> None:
    """Dumps Mario's mesh for a scripted BLJ run and reports the on-disk size.

    Args:
        argv: Unparsed command line arguments.

    Raises:
        app.UsageError: If positional arguments were given.
    """
    if len(argv) > 1:
        raise app.UsageError(f"unexpected arguments: {argv[1:]}")
    config = config_from_flags()
    logging.info("scene %s: %d collision triangles, spawn %s",
                 config.scene.name, len(config.scene.surfaces), config.scene.spawn)
    collected = run(config)
    blob = pack(config, collected)
    os.makedirs(os.path.dirname(config.out_path), exist_ok=True)
    with open(config.out_path, "wb") as handle:
        handle.write(blob)

    records = collected["records"]
    size = os.path.getsize(config.out_path)
    logging.info("mode %s, %d frames kept of %d simulated at stride %d",
                 config.mode, len(records), config.frames, config.stride)
    logging.info("triangle counts seen: %s", collected["triangle_counts"])
    logging.info("color palette %d entries, %d color variants, %d uv variants",
                 len(collected["palette"]), len(collected["color_variants"]),
                 len(collected["uv_variants"]))
    if config.mode == "mesh":
        logging.info("worst position quantisation error %.4f units", collected["worst_error"])
    header_length = struct.unpack("<I", blob[len(MAGIC):len(MAGIC) + 4])[0]
    print(f"wrote {config.out_path}")
    print(f"  on disk        {size:>10d} B  {size / 1e6:.3f} MB")
    print(f"  header json    {header_length:>10d} B")
    print(f"  base64 inflated{int(size * 4 / 3):>10d} B  {size * 4 / 3 / 1e6:.3f} MB "
          f"({size * 4 / 3 / 16e6 * 100:.1f}% of a 16 MB artifact page)")
    entries = json.loads(blob[len(MAGIC) + 4:len(MAGIC) + 4 + header_length])["streams"]
    for name, entry in sorted(entries.items(), key=lambda item: -item[1]["length"]):
        print(f"  {name:<20s} {entry['length']:>10d} B  {entry['codec']}")
    if records:
        peak = min(record.forward_velocity for record in records)
        print(f"  peak forwardVel {peak:.2f}, final position "
              f"({records[-1].position[0]:.1f}, {records[-1].position[1]:.1f}, "
              f"{records[-1].position[2]:.1f})")


if __name__ == "__main__":
    app.run(main)
