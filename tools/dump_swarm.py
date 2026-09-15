"""Captures a room full of trained Marios into one container a viewer can replay.

``tools/dump_mesh.py`` keeps every frame of one Mario's deformed geometry, which costs about
6 KB of base64 per frame. That is affordable for a single replay and hopeless for a swarm: thirty
two Marios over nine hundred frames would want close to two hundred megabytes before compression,
and the artifact page has sixteen.

The way out is that Mario is not thirty two independent animations, he is one animation played by
thirty two bodies at different offsets. libsm64 bakes the bone matrices into a flat 752 triangle
soup at ``m->pos`` with yaw only, so subtracting the reported position and un-rotating by the
reported face angle recovers a pose that repeats exactly whenever ``(animID, animFrame)`` repeats.
Measured on an 1100 frame backwards long jump that collapses to 69 distinct poses, with a median
and a p99 disagreement between repeats of 0.000 units and a worst case of 17.6 units on about one
percent of frames, against a Mario who is 160 units tall. That tail is a limb wobble nobody can
see at swarm scale.

So this tool writes a pose dictionary once and then spends nine bytes per Mario per frame:

    pose index      uint16, into the shared dictionary
    position        3 x int16, quantised at ``position_scale`` steps per world unit
    yaw             int16, one step per 2 pi / 65536 of a turn, which wraps for free
    status          uint8, the bits a viewer needs to not lie about episode boundaries

Everything else is written once for the whole file: the pose positions and normals, the flat per
triangle color palette, the handful of distinct uv layouts, the staircase's own collision
triangles and the Mario texture atlas.

Several checkpoints share one file. Each one is captured as its own run over the same population,
the same frame budget and the same shared pose dictionary, and the header carries what the
checkpoint was and how it did, so a viewer can put an early policy and a late one side by side and
the improvement is visible rather than asserted.

Container layout, little endian, the same shape ``tools/dump_mesh.py`` writes:

    magic           9 bytes, b"MBLJSWRM1"
    header_length   uint32
    header          header_length bytes of UTF-8 JSON
    payload         the streams named in header["streams"], back to back

The codecs are ``dump_mesh``'s: raw, deflate, delta16+deflate and delta8+deflate. Per frame
streams are delta coded down the frame axis, so a Mario standing still costs almost nothing and a
Mario mid chain costs a few bytes.

To place slot ``s`` on kept frame ``f``, take ``p = pose_index[f, s]`` and
``yaw = yaw[f, s] * pi / 32768`` and then for every vertex of pose ``p``:

    lx, ly, lz = pose_positions[p] / pose_scale
    wx = lx * cos(yaw) + lz * sin(yaw) + position[f, s, 0] / position_scale
    wy = ly                           + position[f, s, 1] / position_scale
    wz = -lx * sin(yaw) + lz * cos(yaw) + position[f, s, 2] / position_scale

which is exactly the inverse of the transform the capture applied, and the same formula
``dump_mesh``'s transform-only mode documents.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
import re
import struct
import sys
import time
import zlib
from array import array
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from absl import app, flags, logging

from src.env.audio import AudioRecorder
from src.env.blj_env import ACT_FLAG_AIR, RewardConfig
from src.env.endless_stairs import minimum_escape_speed
from src.env.native import TEXTURE_HEIGHT, TEXTURE_WIDTH, MarioState
from src.env.swarm import GOAL_TOLERANCE, MeshView, Swarm, SwarmConfig
from tools.dump_mesh import (
    ATLAS_SLOT_COUNT,
    ATLAS_SLOT_WIDTH,
    UNTEXTURED_SENTINEL,
    StreamTable,
    atlas_slots,
    delta_encode,
    encode_png,
)

MAGIC = b"MBLJSWRM1"
FORMAT_VERSION = "mblj-swarm/1"
PAGE_BUDGET = 16_000_000

STATUS_RESET = 0x01
STATUS_SUCCESS = 0x02
STATUS_WARPED = 0x04
STATUS_AIRBORNE = 0x08
STATUS_RECORD_SPEED = 0x10

STATUS_BITS = {
    "reset": STATUS_RESET,
    "success": STATUS_SUCCESS,
    "warped": STATUS_WARPED,
    "airborne": STATUS_AIRBORNE,
    "record_speed": STATUS_RECORD_SPEED,
}

YAW_STEPS_PER_TURN = 65536.0
_YAW_SCALE = YAW_STEPS_PER_TURN / (2.0 * math.pi)
NORMAL_SCALE = 127.0

_AREA_2_COLLISION = os.path.join(_ROOT, "third_party", "sm64-port", "levels", "castle_inside",
                                 "areas", "2", "collision.inc.c")
_SURFACE_HEADER = os.path.join(_ROOT, "third_party", "libsm64", "src", "decomp", "include",
                               "surface_terrains.h")

_DTYPES = {
    "int8": "i1",
    "uint8": "u1",
    "int16": "<i2",
    "uint16": "<u2",
    "int32": "<i4",
    "float32": "<f4",
}

_STEP_SUFFIXES = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}

FLAGS = flags.FLAGS


def _define(definer: Any, name: str, default: Any, help_text: str, **kwargs: Any) -> None:
    """Defines a flag unless importing tools.dump_mesh already defined it.

    tools.dump_mesh is imported for its container encoder and registers its own command line as a
    side effect, so every name this tool shares with it is already present and redefining one
    raises DuplicateFlagError. Inheriting dump_mesh's default would be the wrong fix, since the
    two tools disagree about frames and stride, so a shared name keeps this tool's default.

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


_define(flags.DEFINE_multi_string, "checkpoints", [],
        "Repeatable. Either name=path to a stable-baselines3 zip, a bare path whose file name "
        "becomes the name, or auto:<directory> to glob a ladder output tree of the shape "
        "results/ladder/<rung>/seed_N/checkpoints/ppo_<steps>_steps.zip and pick a spread of "
        "step counts.")
_define(flags.DEFINE_string, "audio_dir", None,
        "Directory to write one mp3 per checkpoint into, capturing the whole population's sound. "
        "One audio tick per frame renders every Mario, because libsm64 keeps the audio engine's "
        "state outside the per Mario GlobalState and its own mixer combines the requests.")
_define(flags.DEFINE_float, "audio_volume", 0.4,
        "Master volume for the captured audio. The game's mixer sums every Mario's voices, so a "
        "population of 48 clips at full scale under a policy that keeps them all busy.")
_define(flags.DEFINE_integer, "population", 32, "Marios sharing the room.", lower_bound=1)
_define(flags.DEFINE_integer, "frames", 900, "Frames to simulate per checkpoint.", lower_bound=1)
_define(flags.DEFINE_integer, "stride", 1, "Keep every Nth simulated frame.", lower_bound=1)
_define(flags.DEFINE_boolean, "stochastic", True,
        "Sample from the policy rather than taking its argmax action. Sampling is the default "
        "here because an argmax policy gives every Mario in the room the same action for the "
        "same observation, which collapses the swarm onto one trajectory per spawn offset.")
_define(flags.DEFINE_integer, "policy_seed", 0,
        "Seeds the policy's sampling and the spawn jitter. Each checkpoint is offset from it so "
        "two checkpoints do not share one noise sequence.")
_define(flags.DEFINE_string, "rom", os.path.join(_ROOT, "roms", "baserom.us.z64"),
        "Super Mario 64 US ROM that libsm64 reads animation and texture data from.")
_define(flags.DEFINE_string, "collision_path", _AREA_2_COLLISION,
        "castle_inside/areas/2/collision.inc.c, the endless staircase.")
_define(flags.DEFINE_string, "surface_header", _SURFACE_HEADER,
        "surface_terrains.h supplying the SURFACE_* constants.")
_define(flags.DEFINE_string, "out", os.path.join(_ROOT, "results", "swarm.bin"),
        "Container to write.")
_define(flags.DEFINE_float, "pose_scale", 8.0,
        "Quantisation steps per world unit inside the pose dictionary. 8 gives an eighth of a "
        "unit grid on a body 160 units tall.", lower_bound=0.0625)
_define(flags.DEFINE_float, "position_scale", 4.0,
        "Quantisation steps per world unit for each Mario's world position. 4 gives a quarter "
        "unit grid and keeps the whole staircase inside int16.", lower_bound=0.0625)
_define(flags.DEFINE_integer, "episode_frames", 400,
        "Frames an episode may last before the slot truncates and respawns. Shorter than the "
        "training cap on purpose, so a capture window holds several attempts per Mario.",
        lower_bound=1)
_define(flags.DEFINE_float, "spawn_spread", 140.0,
        "Lateral spawn jitter in world units, across the corridor. Zero stacks every Mario on "
        "one tile, which looks like a single Mario until the policies diverge.", lower_bound=0.0)
_define(flags.DEFINE_integer, "auto_limit", 4,
        "Checkpoints to keep per rung when --checkpoints auto:<dir> globs a ladder tree.",
        lower_bound=1)
_define(flags.DEFINE_boolean, "surfaces", True,
        "Embed the staircase's collision triangles in the container.")
_define(flags.DEFINE_integer, "mesh_samples", 8,
        "Frames per checkpoint whose full world space mesh is kept in memory as ground truth for "
        "the decoder check. Zero skips the mesh half of the check.", lower_bound=0)
_define(flags.DEFINE_boolean, "verify", True,
        "Read the container back after writing it and report the round trip error.")


@dataclasses.dataclass(frozen=True)
class CheckpointSpec:
    """One trained policy to capture, with whatever its path says about its provenance.

    Attributes:
        name: Short label the viewer shows, unique within the container.
        path: Path to the stable-baselines3 zip.
        rung: Reward shaping rung the checkpoint was trained on, parsed from the path.
        seed: Training seed, or None when the path does not say.
        steps: Training steps the checkpoint was saved at, or None when the path does not say.
    """

    name: str
    path: str
    rung: str
    seed: int | None
    steps: int | None


@dataclasses.dataclass(frozen=True)
class DumpSwarmConfig:
    """Everything the capture needs, resolved from flags.

    Attributes:
        rom_path: Path to the Super Mario 64 US ROM.
        collision_path: Path to the staircase's collision.inc.c.
        header_path: Path to surface_terrains.h.
        checkpoints: Policies to capture, in the order the viewer should show them.
        population: Marios sharing the room.
        frames: Frames simulated per checkpoint.
        stride: Keep every Nth simulated frame.
        stochastic: Sample from the policy rather than taking its argmax action.
        policy_seed: Seeds the policy sampling and the spawn jitter.
        pose_scale: Quantisation steps per world unit inside the pose dictionary.
        position_scale: Quantisation steps per world unit for Mario's world position.
        episode_frames: Frames an episode may last before it truncates and respawns.
        spawn_spread: Lateral spawn jitter in world units.
        include_surfaces: Whether to embed the collision triangles.
        mesh_samples: Frames per checkpoint kept as ground truth for the decoder check.
        out_path: Container to write.
    """

    rom_path: str
    collision_path: str
    header_path: str
    checkpoints: tuple[CheckpointSpec, ...]
    population: int
    frames: int
    stride: int
    stochastic: bool
    policy_seed: int
    pose_scale: float
    position_scale: float
    episode_frames: int
    spawn_spread: float
    include_surfaces: bool
    mesh_samples: int
    out_path: str
    audio_dir: str | None = None
    audio_volume: float = 0.4



@dataclasses.dataclass(frozen=True)
class Outcome:
    """Aggregate result over a set of episodes.

    Attributes:
        episodes: Episodes the aggregate covers.
        successes: Episodes that reached the top landing.
        mean_best_height: Mean over episodes of the record height reached while grounded.
        best_height: Best record height any of the episodes reached.
        mean_peak_backward: Mean over episodes of the record backward forwardVel, so negative is
            fast and zero is a Mario who never reversed.
        best_peak_backward: Most negative forwardVel any of the episodes reached.
        mean_return: Mean episode return under the reward weights the capture ran with.
    """

    episodes: int
    successes: int
    mean_best_height: float
    best_height: float
    mean_peak_backward: float
    best_peak_backward: float
    mean_return: float


@dataclasses.dataclass(frozen=True)
class RunCapture:
    """One checkpoint's captured window.

    Attributes:
        spec: The checkpoint this run came from.
        kept_frames: Frames kept after striding.
        pose_index: One uint16 array per kept frame, one entry per slot, indexing the pose
            dictionary in insertion order. :func:`pack` remaps them into the order the container
            stores.
        position: One int16 array per kept frame, three entries per slot.
        yaw: One int16 array per kept frame, one entry per slot.
        status: One uint8-in-int8 array per kept frame, one entry per slot.
        outcome: Aggregate over the episodes that finished inside the window.
        in_flight: Aggregate over the episodes still running when the window closed.
        warps: Instant warps the whole population took inside the window.
        seconds: Wall clock the capture took.
        policy_seconds: Wall clock spent inside model.predict.
        truth_position: Float32 world positions as libsm64 reported them, shape
            ``(kept_frames, population, 3)``, kept for the decoder check.
        truth_yaw: Float32 face angles in radians, shape ``(kept_frames, population)``.
        truth_meshes: Sampled ``(kept frame, slot)`` pairs mapped to a copy of that slot's world
            space mesh, kept for the decoder check.
    """

    spec: CheckpointSpec
    kept_frames: int
    pose_index: list[np.ndarray]
    position: list[array]
    yaw: list[array]
    status: list[array]
    outcome: Outcome
    in_flight: Outcome
    warps: int
    seconds: float
    policy_seconds: float
    truth_position: np.ndarray
    truth_yaw: np.ndarray
    truth_meshes: dict[tuple[int, int], np.ndarray]


def format_steps(steps: int | None) -> str:
    """Renders a training step count the way a viewer tab should read.

    Args:
        steps: Step count, or None when unknown.

    Returns:
        A short human readable string such as "2.0M", "500k" or "?".
    """
    if steps is None:
        return "?"
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M"
    if steps >= 1_000:
        return f"{steps / 1_000:.0f}k"
    return str(steps)


def parse_step_count(text: str) -> int | None:
    """Pulls a training step count out of a checkpoint file name.

    Two spellings show up in this project. The cluster's PPO callback writes
    ``ppo_<steps>_steps.zip``, and the hand copied checkpoints carry a magnitude suffix, as in
    ``curriculum_seed_1_3p5M.zip`` where ``p`` stands in for the decimal point.

    Args:
        text: File name or full path.

    Returns:
        The step count, or None when the name does not carry one.
    """
    match = re.search(r"ppo_(\d+)_steps", text)
    if match:
        return int(match.group(1))
    match = re.search(r"[_-](\d+(?:p\d+)?)([kKmMgG])(?=[_.-]|$)", os.path.basename(text))
    if match:
        return int(float(match.group(1).replace("p", ".")) * _STEP_SUFFIXES[match.group(2).lower()])
    return None


def parse_checkpoint(name: str | None, path: str) -> CheckpointSpec:
    """Reads a checkpoint's rung, seed and step count off its path.

    A ladder path has the shape ``.../<rung>/seed_N/checkpoints/ppo_<steps>_steps.zip`` and says
    everything. A loose file has to say it in its own name, as ``<rung>_seed<N>[_<steps>].zip``.

    Args:
        name: Label the caller gave, or None to build one from the path.
        path: Path to the stable-baselines3 zip.

    Returns:
        The resolved spec.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    stem = os.path.splitext(os.path.basename(path))[0]
    steps = parse_step_count(path)
    rung = stem
    seed: int | None = None
    if "checkpoints" in parts:
        anchor = len(parts) - 1 - parts[::-1].index("checkpoints")
        if anchor >= 2:
            rung = parts[anchor - 2]
            match = re.fullmatch(r"seed[_-]?(\d+)", parts[anchor - 1])
            seed = int(match.group(1)) if match else None
    else:
        match = re.match(r"(.+?)[_-]seed", stem)
        if match:
            rung = match.group(1)
        found = re.search(r"seed[_-]?(\d+)", stem)
        seed = int(found.group(1)) if found else None
    if name is None:
        name = f"{rung}-s{seed}-{format_steps(steps)}" if steps is not None else stem
    return CheckpointSpec(name=name, path=path, rung=rung, seed=seed, steps=steps)


def spread(values: list[int], limit: int) -> list[int]:
    """Picks a limited spread across sorted values, keeping the first and the last.

    Args:
        values: Sorted values to choose from.
        limit: How many to keep.

    Returns:
        A sorted subset of at most ``limit`` values, always including the extremes.
    """
    if limit >= len(values):
        return list(values)
    if limit == 1:
        return [values[-1]]
    picked = {values[round(i * (len(values) - 1) / (limit - 1))] for i in range(limit)}
    return sorted(picked)


def discover_checkpoints(directory: str, limit: int) -> list[CheckpointSpec]:
    """Globs a ladder output tree and picks a spread of step counts per rung.

    The tree the cluster writes is ``<root>/<rung>/seed_N/checkpoints/ppo_<steps>_steps.zip``, and
    this also accepts being pointed straight at a rung directory or at one seed's directory. When
    a rung has several seeds the one with the most checkpoints wins, because a viewer tab per
    rung per seed per step count is more tabs than anybody wants, and a partially trained seed
    would make the improvement look worse than it is.

    Args:
        directory: Directory to glob.
        limit: Checkpoints to keep per rung.

    Returns:
        Specs sorted by rung and then by step count, so the viewer shows training order.

    Raises:
        ValueError: If the directory holds no checkpoints of the expected shape.
    """
    patterns = (
        os.path.join(directory, "*", "seed_*", "checkpoints", "ppo_*_steps.zip"),
        os.path.join(directory, "seed_*", "checkpoints", "ppo_*_steps.zip"),
        os.path.join(directory, "checkpoints", "ppo_*_steps.zip"),
        os.path.join(directory, "ppo_*_steps.zip"),
    )
    found = sorted({match for pattern in patterns for match in glob.glob(pattern)})
    if not found:
        raise ValueError(f"no ppo_*_steps.zip checkpoints under {directory}")

    groups: dict[tuple[str, int | None], dict[int, str]] = {}
    for path in found:
        spec = parse_checkpoint(None, path)
        if spec.steps is None:
            continue
        groups.setdefault((spec.rung, spec.seed), {})[spec.steps] = path

    chosen: list[CheckpointSpec] = []
    for rung in sorted({key[0] for key in groups}):
        seeds = [key for key in groups if key[0] == rung]
        best = max(seeds, key=lambda key: (len(groups[key]), -(key[1] or 0)))
        steps = sorted(groups[best])
        for step in spread(steps, limit):
            chosen.append(parse_checkpoint(None, groups[best][step]))
    return chosen


def resolve_checkpoints(entries: list[str], limit: int) -> tuple[CheckpointSpec, ...]:
    """Turns the --checkpoints flag into specs.

    Args:
        entries: Raw flag values, each ``name=path``, a bare path, or ``auto:<directory>``.
        limit: Checkpoints to keep per rung for an auto entry.

    Returns:
        The resolved specs, in flag order, with duplicate names disambiguated by a suffix.

    Raises:
        ValueError: If no checkpoints were named, or one of them is missing on disk.
    """
    specs: list[CheckpointSpec] = []
    for entry in entries:
        if entry.startswith("auto:"):
            specs.extend(discover_checkpoints(entry[len("auto:"):], limit))
            continue
        name, separator, path = entry.partition("=")
        specs.append(parse_checkpoint(name if separator else None, path if separator else name))
    if not specs:
        raise ValueError("--checkpoints needs at least one name=path, path or auto:<directory>")

    seen: dict[str, int] = {}
    resolved: list[CheckpointSpec] = []
    for spec in specs:
        if not os.path.isfile(spec.path):
            raise ValueError(f"checkpoint {spec.path} does not exist")
        count = seen.get(spec.name, 0)
        seen[spec.name] = count + 1
        resolved.append(spec if not count else dataclasses.replace(
            spec, name=f"{spec.name}#{count + 1}"))
    return tuple(resolved)


def quantise_yaw(yaw: float) -> int:
    """Quantises a face angle to a wrapping int16.

    Angles are the one quantity where wrapping is a feature. One step is 2 pi / 65536 of a turn,
    which is also the precision the game's own s16 angles carry, and the delta coder's modular
    arithmetic makes a spin across the branch cut cost the same as any other frame.

    Args:
        yaw: Face angle in radians.

    Returns:
        The angle in int16 steps, in ``[-32768, 32767]``.
    """
    steps = int(round(yaw * _YAW_SCALE)) % 65536
    return steps - 65536 if steps >= 32768 else steps


def pack_unsigned16(values: np.ndarray) -> array:
    """Reinterprets uint16 samples as the int16 array the delta coder wants.

    Args:
        values: Integer samples in ``[0, 65535]``.

    Returns:
        An ``array("h")`` holding the same bytes, so ``delta_encode`` produces differences that
        wrap mod 65536 and a reader can accumulate them back into uint16.
    """
    return array("h", values.astype("<u2").tobytes())


def pack_unsigned8(values: np.ndarray) -> array:
    """Reinterprets uint8 samples as the int8 array the delta coder wants.

    Args:
        values: Integer samples in ``[0, 255]``.

    Returns:
        An ``array("b")`` holding the same bytes.
    """
    return array("b", values.astype("u1").tobytes())


class PoseTable:
    """The shared pose dictionary, keyed on ``(animID, animFrame)``.

    A pose is one frame of Mario's mesh with his position and his yaw removed, which is what makes
    it shareable between Marios and between checkpoints. The first Mario to show up in a given
    animation frame donates the geometry, and everybody who reaches that frame later spends two
    bytes pointing at it.

    Attributes:
        entries: One ``(anim_id, anim_frame, triangles, color_variant, uv_variant)`` tuple per
            pose, in insertion order.
        positions: Quantised local space positions, one int16 array per pose.
        normals: Quantised local space normals, one int8 array per pose.
        palette: Flat triangle colors seen so far, mapped to their index.
        color_variants: Distinct per triangle palette index layouts.
        uv_variants: Distinct per vertex uv layouts.
        worst_error: Worst absolute position quantisation error in world units.
        lower: Componentwise minimum of every stored local position.
        upper: Componentwise maximum of every stored local position.
    """

    def __init__(self, scale: float) -> None:
        """Starts an empty dictionary.

        Args:
            scale: Quantisation steps per world unit for the stored local positions.
        """
        self._scale = scale
        self._index: dict[tuple[int, int], int] = {}
        self._color_keys: dict[bytes, int] = {}
        self._uv_keys: dict[bytes, int] = {}
        self.entries: list[tuple[int, int, int, int, int]] = []
        self.positions: list[array] = []
        self.normals: list[array] = []
        self.palette: dict[tuple[int, int, int], int] = {}
        self.color_variants: list[array] = []
        self.uv_variants: list[array] = []
        self.worst_error = 0.0
        self.lower = np.full(3, np.inf)
        self.upper = np.full(3, -np.inf)

    def __len__(self) -> int:
        """Returns the number of distinct poses stored so far."""
        return len(self.entries)

    def sorted_order(self) -> tuple[list[int], np.ndarray]:
        """Orders the dictionary by animation, which is what makes the delta coder earn its keep.

        Poses arrive in the order the population happened to reach them, so neighbours in
        insertion order are usually unrelated animations and their difference is larger than
        either pose. Sorting by ``(animID, animFrame)`` puts consecutive frames of one animation
        next to each other, where the difference is a limb moving a few units.

        Returns:
            A pair of the insertion indices in stored order and a lookup that maps an insertion
            index to its stored index.
        """
        order = sorted(range(len(self.entries)), key=lambda index: self.entries[index][:2])
        lookup = np.zeros(len(order), dtype=np.uint16)
        for stored, original in enumerate(order):
            lookup[original] = stored
        return order, lookup

    def add(self, anim_id: int, anim_frame: int, mesh: MeshView,
            origin: tuple[float, float, float], yaw: float) -> int:
        """Returns the index of this animation frame, storing it on first sight.

        Args:
            anim_id: Animation id from the mario state.
            anim_frame: Frame within that animation.
            mesh: Live views onto the Mario who is currently in this pose.
            origin: That Mario's world position.
            yaw: That Mario's face angle in radians.

        Returns:
            The pose index, in ``[0, 65536)``.

        Raises:
            ValueError: If the dictionary would grow past what a uint16 index can address.
        """
        key = (anim_id, anim_frame)
        found = self._index.get(key)
        if found is not None:
            return found
        if len(self.entries) >= 65536:
            raise ValueError("pose dictionary outgrew a uint16 index")

        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        delta = mesh.position.astype(np.float64) - np.asarray(origin, dtype=np.float64)
        local = np.empty_like(delta)
        local[..., 0] = delta[..., 0] * cos_yaw - delta[..., 2] * sin_yaw
        local[..., 1] = delta[..., 1]
        local[..., 2] = delta[..., 0] * sin_yaw + delta[..., 2] * cos_yaw
        steps = np.clip(np.rint(local * self._scale), -32768.0, 32767.0)
        self.worst_error = max(self.worst_error,
                               float(np.max(np.abs(steps / self._scale - local))))
        flattened = local.reshape(-1, 3)
        self.lower = np.minimum(self.lower, flattened.min(axis=0))
        self.upper = np.maximum(self.upper, flattened.max(axis=0))

        normal = mesh.normal.astype(np.float64)
        rotated = np.empty_like(normal)
        rotated[..., 0] = normal[..., 0] * cos_yaw - normal[..., 2] * sin_yaw
        rotated[..., 1] = normal[..., 1]
        rotated[..., 2] = normal[..., 0] * sin_yaw + normal[..., 2] * cos_yaw
        octets = np.clip(np.rint(rotated * NORMAL_SCALE), -127.0, 127.0)

        flat = np.clip(np.rint(mesh.color[:, 0, :] * 255.0), 0.0, 255.0).astype(np.uint8)
        indices = array("B", bytes(mesh.num_triangles))
        for triangle in range(mesh.num_triangles):
            rgb = (int(flat[triangle, 0]), int(flat[triangle, 1]), int(flat[triangle, 2]))
            if rgb not in self.palette:
                self.palette[rgb] = len(self.palette)
            indices[triangle] = self.palette[rgb]
        color_variant = self._color_keys.setdefault(indices.tobytes(), len(self.color_variants))
        if color_variant == len(self.color_variants):
            self.color_variants.append(indices)

        uv = np.ascontiguousarray(mesh.uv, dtype=np.float32).tobytes()
        uv_variant = self._uv_keys.setdefault(uv, len(self.uv_variants))
        if uv_variant == len(self.uv_variants):
            self.uv_variants.append(array("f", uv))

        index = len(self.entries)
        self._index[key] = index
        self.positions.append(array("h", steps.astype("<i2").tobytes()))
        self.normals.append(array("b", octets.astype("i1").tobytes()))
        self.entries.append((anim_id, anim_frame, mesh.num_triangles, color_variant, uv_variant))
        return index


def slot_states(swarm: Swarm) -> list[MarioState]:
    """Reaches for the per slot mario state the swarm ticks into.

    The swarm hands out observations, bookkeeping and live mesh views, but a pose dictionary needs
    the animation key, the world position and the face angle, which only the raw state carries.
    This is the same reach-through ``tools/dump_mesh.py`` already makes for the geometry buffers,
    and it is read only.

    Args:
        swarm: A live swarm.

    Returns:
        One state per slot, in slot order, aliasing the structs libsm64 writes into.
    """
    return [slot.state for slot in swarm._slots]


def summarise(episodes: list[tuple[bool, float, float, float]]) -> Outcome:
    """Aggregates a list of episode results.

    Args:
        episodes: One ``(success, best_height, peak_backward, episode_return)`` tuple per episode.

    Returns:
        The aggregate, all zeros when the list is empty.
    """
    if not episodes:
        return Outcome(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    heights = [episode[1] for episode in episodes]
    speeds = [episode[2] for episode in episodes]
    returns = [episode[3] for episode in episodes]
    return Outcome(
        episodes=len(episodes),
        successes=sum(1 for episode in episodes if episode[0]),
        mean_best_height=float(np.mean(heights)),
        best_height=float(np.max(heights)),
        mean_peak_backward=float(np.mean(speeds)),
        best_peak_backward=float(np.min(speeds)),
        mean_return=float(np.mean(returns)),
    )


def capture_run(config: DumpSwarmConfig, order: int, spec: CheckpointSpec,
                poses: PoseTable, audio_path: str | None = None) -> tuple[RunCapture, bytes, Any]:
    """Runs one checkpoint's population and collects its per frame streams.

    When an audio path is given, the whole population's sound is recorded with one tick per
    frame. That works because libsm64 keeps the audio engine's state as a file scope global
    rather than in the per Mario GlobalState, so every slot's play_sound calls land in the same
    request queue and the game's own mixer combines them. The mix saturates at around eight
    simultaneous Marios because the engine has a fixed voice limit and drops the surplus itself,
    which is the console's behaviour rather than a shortcoming here.

    Args:
        config: Resolved capture configuration.
        order: Index of this checkpoint within the container, used to offset the seeds.
        spec: The checkpoint to load.
        poses: Shared pose dictionary, extended in place.
        audio_path: Where to write this checkpoint's audio, or None to stay silent.

    Returns:
        A triple of the capture, the Mario texture atlas as PNG bytes and the loaded scene.
    """
    from stable_baselines3 import PPO

    model = PPO.load(spec.path, device="cpu")
    model.set_random_seed(config.policy_seed + order)
    swarm = Swarm(SwarmConfig(
        rom_path=config.rom_path,
        collision_path=config.collision_path,
        header_path=config.header_path,
        population=config.population,
        max_frames=config.episode_frames,
        reward=RewardConfig(terminal=1.0),
        spawn_spread=config.spawn_spread,
        seed=config.policy_seed + order))
    recorder = (AudioRecorder(swarm.game, volume=config.audio_volume)
                if audio_path else None)
    try:
        atlas = encode_png(TEXTURE_WIDTH, TEXTURE_HEIGHT, bytes(swarm._game._texture))
        scene = swarm.scene
        states = slot_states(swarm)
        population = config.population
        kept = len(range(0, config.frames, config.stride))
        sample_every = max(1, kept // config.mesh_samples) if config.mesh_samples else 0

        pose_index: list[np.ndarray] = []
        position: list[array] = []
        yaw: list[array] = []
        status: list[array] = []
        truth_position = np.zeros((kept, population, 3), dtype=np.float32)
        truth_yaw = np.zeros((kept, population), dtype=np.float32)
        truth_meshes: dict[tuple[int, int], np.ndarray] = {}
        finished: list[tuple[bool, float, float, float]] = []
        pending = np.zeros(population, dtype=np.uint8)
        warps = 0
        policy_seconds = 0.0
        started = time.perf_counter()
        previous = swarm.members()
        row = 0

        for frame in range(config.frames):
            observation = swarm.observations()
            before = time.perf_counter()
            actions, _ = model.predict(observation, deterministic=not config.stochastic)
            policy_seconds += time.perf_counter() - before
            members = swarm.step(actions)
            if recorder is not None:
                recorder.capture()

            for index, (member, earlier) in enumerate(zip(members, previous, strict=True)):
                if member.warps > earlier.warps:
                    pending[index] |= STATUS_WARPED
                if member.peak_backward < earlier.peak_backward:
                    pending[index] |= STATUS_RECORD_SPEED
                if member.episodes > earlier.episodes:
                    pending[index] |= STATUS_RESET
                    if member.success:
                        pending[index] |= STATUS_SUCCESS
                    finished.append((member.success, member.best_height, member.peak_backward,
                                     member.episode_return))
                    warps += member.warps
            previous = members

            if frame % config.stride:
                continue

            pose_row = np.zeros(population, dtype=np.uint16)
            position_row = np.zeros(3 * population, dtype=np.int16)
            yaw_row = np.zeros(population, dtype=np.int16)
            status_row = np.zeros(population, dtype=np.uint8)
            for index, state in enumerate(states):
                origin = (float(state.position[0]), float(state.position[1]),
                          float(state.position[2]))
                angle = float(state.faceAngle)
                mesh = swarm.mesh(index)
                pose_row[index] = poses.add(int(state.animID), int(state.animFrame), mesh, origin,
                                            angle)
                for axis in range(3):
                    steps = int(round(origin[axis] * config.position_scale))
                    position_row[3 * index + axis] = max(-32768, min(32767, steps))
                yaw_row[index] = quantise_yaw(angle)
                status_row[index] = pending[index] | (
                    STATUS_AIRBORNE if state.action & ACT_FLAG_AIR else 0)
                truth_position[row, index] = origin
                truth_yaw[row, index] = angle
                if sample_every and row % sample_every == 0 and index == row % population:
                    truth_meshes[(row, index)] = mesh.position.copy()
            pending[:] = 0

            pose_index.append(pose_row)
            position.append(array("h", position_row.tobytes()))
            yaw.append(array("h", yaw_row.tobytes()))
            status.append(pack_unsigned8(status_row))
            row += 1

        elapsed = time.perf_counter() - started
        running = swarm.members()
        in_flight = summarise([(member.success, member.best_height, member.peak_backward,
                                member.episode_return) for member in running])
        warps += sum(member.warps for member in running)
        return RunCapture(
            spec=spec, kept_frames=row, pose_index=pose_index, position=position, yaw=yaw,
            status=status, outcome=summarise(finished), in_flight=in_flight, warps=warps,
            seconds=elapsed, policy_seconds=policy_seconds, truth_position=truth_position[:row],
            truth_yaw=truth_yaw[:row], truth_meshes=truth_meshes), atlas, scene
    finally:
        if recorder is not None and recorder.frames:
            written = recorder.write(audio_path)
            logging.info("audio %s: %.2f s, peak %d, %d bytes", written, recorder.seconds,
                         recorder.peak, os.path.getsize(written))
        swarm.close()


def run(config: DumpSwarmConfig) -> dict[str, Any]:
    """Captures every checkpoint into one shared pose dictionary.

    Args:
        config: Resolved capture configuration.

    Returns:
        A dict with the captures, the pose table, the texture atlas and the loaded scene.
    """
    poses = PoseTable(config.pose_scale)
    captures: list[RunCapture] = []
    atlas = b""
    scene: Any = None
    for order, spec in enumerate(config.checkpoints):
        logging.info("capturing %s: %s", spec.name, spec.path)
        audio_path = (os.path.join(config.audio_dir, f"swarm_{spec.name}.mp3")
                      if config.audio_dir else None)
        if audio_path:
            os.makedirs(config.audio_dir, exist_ok=True)
        capture, atlas, scene = capture_run(config, order, spec, poses, audio_path=audio_path)
        logging.info("%s: %d frames, %d episodes, %d successes, best height %.0f, peak %.1f, "
                     "%d poses so far, %.2f s (%.2f s policy)",
                     spec.name, capture.kept_frames, capture.outcome.episodes,
                     capture.outcome.successes, capture.outcome.best_height,
                     capture.outcome.best_peak_backward, len(poses), capture.seconds,
                     capture.policy_seconds)
        captures.append(capture)
    return {"captures": captures, "poses": poses, "atlas": atlas, "scene": scene}


def scene_name(collision_path: str) -> str:
    """Names the level area a collision path points at.

    Args:
        collision_path: Path to a ``collision.inc.c`` inside the decompilation's level tree.

    Returns:
        A short label such as "castle_inside/areas/2", or the file's own directory name when the
        path does not have the decompilation's shape.
    """
    parts = os.path.normpath(os.path.abspath(collision_path)).split(os.sep)
    if len(parts) >= 4:
        return "/".join(parts[-4:-1])
    return os.path.basename(os.path.dirname(collision_path))


def pack(config: DumpSwarmConfig, collected: dict[str, Any]) -> bytes:
    """Packs the captured runs into the container bytes.

    Args:
        config: Resolved capture configuration.
        collected: The dict returned by :func:`run`.

    Returns:
        The full container, ready to write to disk.
    """
    poses: PoseTable = collected["poses"]
    captures: list[RunCapture] = collected["captures"]
    scene = collected["scene"]
    streams = StreamTable()

    pose_order, lookup = poses.sorted_order()
    streams.add("atlas_png", collected["atlas"], "uint8", "raw",
                [TEXTURE_HEIGHT, TEXTURE_WIDTH, 4])
    streams.add("pose_positions",
                zlib.compress(delta_encode([poses.positions[pose] for pose in pose_order], "h"), 9),
                "int16", "delta16+deflate", [len(poses), -1, 3, 3])
    streams.add("pose_normals",
                zlib.compress(delta_encode([poses.normals[pose] for pose in pose_order], "b"), 9),
                "int8", "delta8+deflate", [len(poses), -1, 3, 3])
    streams.add("color_index", zlib.compress(
        b"".join(variant.tobytes() for variant in poses.color_variants), 9),
        "uint8", "deflate", [len(poses.color_variants), -1])
    streams.add("uv", zlib.compress(
        b"".join(variant.tobytes() for variant in poses.uv_variants), 9),
        "float32", "deflate", [len(poses.uv_variants), -1, 3, 2])

    for order, capture in enumerate(captures):
        prefix = f"run{order}"
        streams.add(f"{prefix}/pose_index",
                    zlib.compress(delta_encode(
                        [pack_unsigned16(lookup[row]) for row in capture.pose_index], "h"), 9),
                    "uint16", "delta16+deflate", [capture.kept_frames, config.population])
        streams.add(f"{prefix}/position",
                    zlib.compress(delta_encode(capture.position, "h"), 9),
                    "int16", "delta16+deflate", [capture.kept_frames, config.population, 3])
        streams.add(f"{prefix}/yaw", zlib.compress(delta_encode(capture.yaw, "h"), 9),
                    "int16", "delta16+deflate", [capture.kept_frames, config.population])
        streams.add(f"{prefix}/status", zlib.compress(delta_encode(capture.status, "b"), 9),
                    "uint8", "delta8+deflate", [capture.kept_frames, config.population])

    if config.include_surfaces:
        vertices = array("i", (int(surface.vertices[corner][axis])
                               for surface in scene.surfaces
                               for corner in range(3) for axis in range(3)))
        types = array("h", (int(surface.type) for surface in scene.surfaces))
        streams.add("surface_vertices", zlib.compress(vertices.tobytes(), 9), "int32", "deflate",
                    [len(scene.surfaces), 3, 3])
        streams.add("surface_types", zlib.compress(types.tobytes(), 9), "int16", "deflate",
                    [len(scene.surfaces)])

    header: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "rom": os.path.basename(config.rom_path),
        "scene": scene_name(config.collision_path),
        "collision_path": config.collision_path,
        "population": config.population,
        "frames": captures[0].kept_frames if captures else 0,
        "simulated_frames": config.frames,
        "stride": config.stride,
        "episode_frames": config.episode_frames,
        "stochastic": config.stochastic,
        "policy_seed": config.policy_seed,
        "spawn_spread": config.spawn_spread,
        "spawn": list(scene.spawn),
        "pose_count": len(poses),
        "pose_fields": ["anim_id", "anim_frame", "triangles", "color_variant", "uv_variant"],
        "poses": [list(poses.entries[pose]) for pose in pose_order],
        "pose_scale": config.pose_scale,
        "pose_local_bounds": [poses.lower.round(3).tolist(), poses.upper.round(3).tolist()],
        "pose_quantisation_error": poses.worst_error,
        "position_scale": config.position_scale,
        "normal_scale": NORMAL_SCALE,
        "yaw_radians_per_step": 2.0 * math.pi / YAW_STEPS_PER_TURN,
        "yaw_note": "yaw = sample * 2 * pi / 65536, so the int16 wraps exactly once per turn",
        "status_bits": STATUS_BITS,
        "status_note": "reset marks the frame a slot respawned on, and its geometry already "
                       "belongs to the new episode, so do not interpolate across it. success "
                       "rides along with reset, because reaching the landing ends the episode. "
                       "warped and record_speed are ORed over the frames skipped by the stride.",
        "warp": {
            "surface_type": scene.warp.surface_type,
            "displacement": list(scene.warp.displacement),
            "x_range": list(scene.warp.x_range),
            "y_range": list(scene.warp.y_range),
            "z_range": list(scene.warp.z_range),
            "depth": scene.warp.depth,
            "minimum_escape_speed": minimum_escape_speed(scene.warp),
        },
        "goal": {
            "goal_y": scene.goal_y,
            "goal_z": scene.goal_z,
            "tolerance": GOAL_TOLERANCE,
            "ascends_toward": list(scene.ascends_toward),
        },
        "checkpoints": [{
            "name": capture.spec.name,
            "path": capture.spec.path,
            "rung": capture.spec.rung,
            "seed": capture.spec.seed,
            "steps": capture.spec.steps,
            "steps_label": format_steps(capture.spec.steps),
            "frames": capture.kept_frames,
            "warps": capture.warps,
            "seconds": round(capture.seconds, 3),
            "policy_seconds": round(capture.policy_seconds, 3),
            "outcome": dataclasses.asdict(capture.outcome),
            "in_flight": dataclasses.asdict(capture.in_flight),
        } for capture in captures],
        "coordinate_system": "sm64 world space, y up, right handed, one unit per sm64 unit, "
                             "same space as the loaded collision surfaces",
        "winding": "counter-clockwise front facing",
        "mario_origin": "the stored position is the one libsm64 reports, which sits at Mario's "
                        "feet for an upright action; pose_local_bounds carries the measured "
                        "extent of every stored pose around it, and some actions reach below it",
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
        "color_palette": [list(rgb) for rgb, _ in sorted(poses.palette.items(),
                                                         key=lambda item: item[1])],
        "color_variant_count": len(poses.color_variants),
        "uv_variant_count": len(poses.uv_variants),
        "streams": streams.entries,
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(blob)) + blob + streams.payload


def read_container(path: str) -> tuple[dict[str, Any], bytes]:
    """Reads a container back off disk.

    Args:
        path: Path to a container this tool wrote.

    Returns:
        A pair of the parsed header and the payload bytes.

    Raises:
        ValueError: If the magic does not match.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:len(MAGIC)] != MAGIC:
        raise ValueError(f"{path} is not a {MAGIC.decode()} container")
    length = struct.unpack("<I", blob[len(MAGIC):len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    header = json.loads(blob[start:start + length])
    return header, blob[start + length:]


def decode_stream(entry: dict[str, Any], payload: bytes) -> np.ndarray:
    """Decodes one payload stream back into its samples.

    Args:
        entry: The stream's header entry, carrying offset, length, dtype, codec and shape.
        payload: The container's payload bytes.

    Returns:
        The decoded samples, shaped ``(rows, -1)`` for a delta coded stream and flat otherwise.

    Raises:
        ValueError: If the codec is not one this format defines.
    """
    raw = payload[entry["offset"]:entry["offset"] + entry["length"]]
    codec = entry["codec"]
    if codec.endswith("deflate") and codec != "raw":
        raw = zlib.decompress(raw)
    if codec in ("raw", "deflate"):
        return np.frombuffer(raw, dtype=_DTYPES[entry["dtype"]])
    if codec == "delta16+deflate":
        modulus, signed = 65536, "<i2"
    elif codec == "delta8+deflate":
        modulus, signed = 256, "i1"
    else:
        raise ValueError(f"unknown codec {codec!r}")
    rows = entry["shape"][0]
    deltas = np.frombuffer(raw, dtype=signed).reshape(rows, -1).astype(np.int64)
    wrapped = np.cumsum(deltas, axis=0) % modulus
    if entry["dtype"].startswith("int"):
        wrapped = np.where(wrapped >= modulus // 2, wrapped - modulus, wrapped)
    return wrapped


def rebuild_pose(header: dict[str, Any], payload: bytes, pose: int) -> np.ndarray:
    """Rebuilds one pose's local space geometry from the container.

    Args:
        header: Parsed container header.
        payload: Container payload bytes.
        pose: Pose index.

    Returns:
        A ``(triangles, 3, 3)`` float64 array of local space positions in world units.
    """
    samples = decode_stream(header["streams"]["pose_positions"], payload)
    triangles = header["poses"][pose][2]
    return samples[pose, :9 * triangles].reshape(triangles, 3, 3) / header["pose_scale"]


def verify(path: str, captures: list[RunCapture]) -> dict[str, Any]:
    """Reads a container back and checks it against what the capture actually saw.

    The positions and the face angles have to round trip inside their own quantisation step, which
    is a hard bound and is asserted. The mesh is a softer question: a pose dictionary answers with
    the first mesh that ever carried a given ``(animID, animFrame)``, and repeats of an animation
    frame are not always bit identical, so the reconstruction error is reported as a distribution
    rather than checked against a threshold.

    Args:
        path: Container to read.
        captures: The captures that produced it, as ground truth.

    Returns:
        A dict of the measured errors and the bounds they were checked against.

    Raises:
        ValueError: If a position or a face angle failed to round trip.
    """
    header, payload = read_container(path)
    position_step = 1.0 / header["position_scale"]
    yaw_step = header["yaw_radians_per_step"]
    worst_position = 0.0
    worst_yaw = 0.0
    mesh_errors: list[float] = []
    checked_meshes = 0

    for order, capture in enumerate(captures):
        streams = header["streams"]
        positions = decode_stream(streams[f"run{order}/position"], payload).reshape(
            capture.kept_frames, header["population"], 3) / header["position_scale"]
        yaws = decode_stream(streams[f"run{order}/yaw"], payload).reshape(
            capture.kept_frames, header["population"]) * yaw_step
        poses = decode_stream(streams[f"run{order}/pose_index"], payload).reshape(
            capture.kept_frames, header["population"])

        worst_position = max(worst_position,
                             float(np.max(np.abs(positions - capture.truth_position))))
        difference = (yaws - capture.truth_yaw + math.pi) % (2.0 * math.pi) - math.pi
        worst_yaw = max(worst_yaw, float(np.max(np.abs(difference))))

        for (row, slot), truth in capture.truth_meshes.items():
            local = rebuild_pose(header, payload, int(poses[row, slot]))
            yaw = float(yaws[row, slot])
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            world = np.empty_like(local)
            world[..., 0] = local[..., 0] * cos_yaw + local[..., 2] * sin_yaw
            world[..., 1] = local[..., 1]
            world[..., 2] = -local[..., 0] * sin_yaw + local[..., 2] * cos_yaw
            world += positions[row, slot]
            mesh_errors.append(float(np.max(np.abs(world - truth))))
            checked_meshes += 1

    bound = 0.5 * position_step + 1e-6
    if worst_position > bound:
        raise ValueError(f"position round trip error {worst_position} exceeds half a step {bound}")
    if worst_yaw > 0.5 * yaw_step + 1e-9:
        raise ValueError(f"yaw round trip error {worst_yaw} exceeds half a step")

    return {
        "position_step": position_step,
        "worst_position_error": worst_position,
        "yaw_step": yaw_step,
        "worst_yaw_error": worst_yaw,
        "meshes_checked": checked_meshes,
        "mesh_error_median": float(np.median(mesh_errors)) if mesh_errors else 0.0,
        "mesh_error_p99": float(np.percentile(mesh_errors, 99)) if mesh_errors else 0.0,
        "mesh_error_max": float(np.max(mesh_errors)) if mesh_errors else 0.0,
    }


def config_from_flags() -> DumpSwarmConfig:
    """Resolves the command line flags into a DumpSwarmConfig.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If no checkpoint was named or one of them is missing.
    """
    return DumpSwarmConfig(
        rom_path=FLAGS.rom,
        collision_path=FLAGS.collision_path,
        header_path=FLAGS.surface_header,
        checkpoints=resolve_checkpoints(list(FLAGS.checkpoints), FLAGS.auto_limit),
        population=FLAGS.population,
        frames=FLAGS.frames,
        stride=FLAGS.stride,
        stochastic=FLAGS.stochastic,
        policy_seed=FLAGS.policy_seed,
        pose_scale=FLAGS.pose_scale,
        position_scale=FLAGS.position_scale,
        episode_frames=FLAGS.episode_frames,
        spawn_spread=FLAGS.spawn_spread,
        include_surfaces=FLAGS.surfaces,
        mesh_samples=FLAGS.mesh_samples,
        out_path=FLAGS.out,
        audio_dir=FLAGS.audio_dir,
        audio_volume=FLAGS.audio_volume)


def main(argv: list[str]) -> None:
    """Captures the swarm, writes the container and reports what it cost.

    Args:
        argv: Unparsed command line arguments.

    Raises:
        app.UsageError: If positional arguments were given.
    """
    if len(argv) > 1:
        raise app.UsageError(f"unexpected arguments: {argv[1:]}")
    config = config_from_flags()
    for spec in config.checkpoints:
        logging.info("checkpoint %s: rung %s, seed %s, steps %s", spec.name, spec.rung, spec.seed,
                     format_steps(spec.steps))
    collected = run(config)
    blob = pack(config, collected)
    os.makedirs(os.path.dirname(config.out_path), exist_ok=True)
    with open(config.out_path, "wb") as handle:
        handle.write(blob)

    poses: PoseTable = collected["poses"]
    captures: list[RunCapture] = collected["captures"]
    size = len(blob)
    header_length = struct.unpack("<I", blob[len(MAGIC):len(MAGIC) + 4])[0]
    encoded = int(size * 4 / 3)
    frames = captures[0].kept_frames if captures else 0
    mario_frames = frames * config.population * len(captures)

    print(f"wrote {config.out_path}")
    print(f"  {len(captures)} checkpoints x {config.population} Marios x {frames} kept frames "
          f"= {mario_frames} mario-frames")
    print(f"  pose dictionary {len(poses)} poses, worst local quantisation error "
          f"{poses.worst_error:.4f} units")
    print(f"  on disk        {size:>10d} B  {size / 1e6:.3f} MB")
    print(f"  header json    {header_length:>10d} B")
    print(f"  base64 inflated{encoded:>10d} B  {encoded / 1e6:.3f} MB "
          f"({encoded / PAGE_BUDGET * 100:.1f}% of a 16 MB artifact page)")
    if mario_frames:
        print(f"  per mario-frame {size / mario_frames:.2f} B on disk")
    entries = json.loads(blob[len(MAGIC) + 4:len(MAGIC) + 4 + header_length])["streams"]
    for name, entry in sorted(entries.items(), key=lambda item: -item[1]["length"]):
        print(f"  {name:<22s} {entry['length']:>10d} B  {entry['codec']}")
    for capture in captures:
        outcome = capture.outcome
        print(f"  {capture.spec.name:<24s} episodes {outcome.episodes:>3d} successes "
              f"{outcome.successes:>3d} height mean {outcome.mean_best_height:>7.0f} best "
              f"{outcome.best_height:>7.0f} peak mean {outcome.mean_peak_backward:>9.1f} best "
              f"{outcome.best_peak_backward:>9.1f} warps {capture.warps:>4d}")

    if FLAGS.verify:
        report = verify(config.out_path, captures)
        print("decoder check")
        print(f"  position step {report['position_step']:.4f} units, worst round trip error "
              f"{report['worst_position_error']:.6f} units")
        print(f"  yaw step {report['yaw_step']:.6f} rad, worst round trip error "
              f"{report['worst_yaw_error']:.8f} rad")
        print(f"  mesh reconstruction over {report['meshes_checked']} sampled frames: median "
              f"{report['mesh_error_median']:.3f}, p99 {report['mesh_error_p99']:.3f}, max "
              f"{report['mesh_error_max']:.3f} units")


if __name__ == "__main__":
    app.run(main)
