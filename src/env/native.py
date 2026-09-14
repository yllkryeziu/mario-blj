"""ctypes binding for the vendored libsm64, the physics substrate this project measures on.

libsm64 is the Super Mario 64 decompilation built as a shared library: Mario's own movement code
and his collision against a static surface list, and nothing else. There is no level script, no
camera and no object behaviour behind the API, which is why the instant warp that turns the
endless staircase into a loop is reimplemented in ``src.env.endless_stairs`` rather than found
here.

Two properties of this boundary are worth knowing before changing anything in this module.

Every struct below was checked field by field against ``offsetof`` on the C side rather than read
off the header by eye, because ctypes derives its layout from the field list alone and a mismatch
does not fail to load. It silently reads the wrong bytes. ``MarioExtraState`` is 124 bytes on both
sides; adding a field to one side without the other turns ``sm64_mario_extra_state`` into a buffer
overflow whose symptom is plausible looking numbers.

The stick fields are normalized to [-1, 1]. libsm64 scales them by 64 on the way in and negates
stickX, so the +-64 range an N64 controller reports is already applied and passing it through a
second time yields a stick magnitude of 4096 instead of 64. That is a different physics regime
rather than a clipped input: the ground actions clamp it away, while one air frame of it takes
forwardVel from 0 to -6142 and back in a single tick, which reads as an enormous chain that
carried nothing.
"""

import ctypes
import os
import re
import sys

TEXTURE_WIDTH = 64 * 11
TEXTURE_HEIGHT = 64
GEO_MAX_TRIANGLES = 1024

ACT_IDLE = 0x0C400201
ACT_CROUCHING = 0x0C008220
ACT_START_CROUCHING = 0x0C008221
ACT_LONG_JUMP_LAND_STOP = 0x0800023B
ACT_WALKING = 0x04000440
ACT_BRAKING = 0x04000445
ACT_CROUCH_SLIDE = 0x04808459
ACT_LONG_JUMP_LAND = 0x00000479
ACT_LONG_JUMP = 0x03000888
ACT_DIVE = 0x0188088A
ACT_FREEFALL = 0x0100088C
ACT_GROUND_POUND = 0x008008A9

_ACTION_DEFINE = re.compile(r"^#define\s+(ACT_[A-Z0-9_]+)\s+(0x[0-9A-Fa-f]+)")
_ACTION_EXCLUDED = ("FLAG", "MASK", "GROUP")

FALLBACK_ACTION_NAMES = {
    ACT_IDLE: "idle",
    ACT_CROUCHING: "crouching",
    ACT_START_CROUCHING: "start_crouching",
    ACT_LONG_JUMP_LAND_STOP: "long_jump_land_stop",
    ACT_WALKING: "walking",
    ACT_BRAKING: "braking",
    ACT_CROUCH_SLIDE: "crouch_slide",
    ACT_LONG_JUMP_LAND: "long_jump_land",
    ACT_LONG_JUMP: "long_jump",
    ACT_DIVE: "dive",
    ACT_FREEFALL: "freefall",
    ACT_GROUND_POUND: "ground_pound",
}


def _load_action_names() -> dict[int, str]:
    """Reads every ACT_ constant from the vendored decompilation header.

    The twelve constants this module declares by hand cover the backwards long jump chain, but a
    trace is far easier to read when every action Mario can enter has a name. Falls back to the
    hand written table when third_party is absent, so the module imports on a fresh clone.

    Returns:
        A mapping from action value to a lower case name with the ACT_ prefix removed.
    """
    header = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "third_party", "libsm64", "src", "decomp", "include", "sm64.h")
    if not os.path.exists(header):
        return dict(FALLBACK_ACTION_NAMES)

    names: dict[int, str] = {}
    with open(header, encoding="utf-8") as handle:
        for line in handle:
            match = _ACTION_DEFINE.match(line)
            if match is None:
                continue
            name = match.group(1)
            if any(tag in name for tag in _ACTION_EXCLUDED):
                continue
            names.setdefault(int(match.group(2), 16), name[len("ACT_"):].lower())
    return names or dict(FALLBACK_ACTION_NAMES)


ACTION_NAMES = _load_action_names()


def action_name(action: int) -> str:
    """Names an action value for a trace, a replay or a log line.

    Every consumer of an action goes through here rather than through ACTION_NAMES, so that the
    unknown case is a readable hex value instead of a KeyError. Unknown values do turn up: the
    table is built from the header the checkout happens to have, and Mario carries action bits
    that are not in it.

    Args:
        action: Raw action bitfield, as ``MarioState.action`` reports it.

    Returns:
        The lower case name, or the value formatted as ``0xXXXXXXXX`` if it is not in the table.
    """
    return ACTION_NAMES.get(action, f"0x{action:08X}")

SURFACE_DEFAULT = 0x0000
SURFACE_VERY_SLIPPERY = 0x0013
SURFACE_SLIPPERY = 0x0014
SURFACE_NOT_SLIPPERY = 0x0015
SURFACE_ICE = 0x002E

TERRAIN_GRASS = 0x0000
TERRAIN_STONE = 0x0001
TERRAIN_SLIDE = 0x0006


class MarioInputs(ctypes.Structure):
    """One frame of controller state, mirroring ``struct SM64MarioInputs``.

    The camera fields are a look direction rather than a position, and the decompilation resolves
    the stick against the camera's yaw, so a scene that pins ``camLookX, camLookZ`` to (0, 1) hands
    the agent a stick frame that never rotates under it. Every caller in this project does that.

    Attributes:
        camLookX: x of the camera look direction.
        camLookZ: z of the camera look direction.
        stickX: Stick x in [-1, 1], positive right. libsm64 multiplies by -64 on the way in.
        stickY: Stick y in [-1, 1], positive forward. libsm64 multiplies by 64.
        buttonA: Nonzero while A is held.
        buttonB: Nonzero while B is held.
        buttonZ: Nonzero while Z is held.
    """

    _fields_ = [
        ("camLookX", ctypes.c_float),
        ("camLookZ", ctypes.c_float),
        ("stickX", ctypes.c_float),
        ("stickY", ctypes.c_float),
        ("buttonA", ctypes.c_uint8),
        ("buttonB", ctypes.c_uint8),
        ("buttonZ", ctypes.c_uint8),
    ]


class MarioState(ctypes.Structure):
    """What ``sm64_mario_tick`` writes back, mirroring ``struct SM64MarioState``.

    ``faceAngle`` is the one field libsm64 converts on the way out, from the decompilation's s16
    angle units into radians. Everything else is Mario's own field verbatim, ``forwardVelocity``
    included, which is why the exploit can be measured on it directly instead of inferred from the
    velocity vector.

    Attributes:
        position: World position, three floats.
        velocity: World velocity, three floats.
        faceAngle: Yaw in radians, converted from s16 units.
        forwardVelocity: Signed speed along Mario's facing. The backwards long jump grows this in
            the negative direction with no bound the game enforces, so it is the quantity every
            reward term, replay and validation number in this project reports.
        health: Health in the game's units, 0x880 when full. Zero ends an episode.
        action: Raw action bitfield. Compare against the ACT_ constants or pass to action_name.
        animID: Animation the graphics node is playing.
        animFrame: Frame within that animation.
        flags: Mario's flag bitfield, cap and metal state included.
        particleFlags: Particles this frame asked for.
        invincTimer: Frames of invincibility left.
    """

    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("velocity", ctypes.c_float * 3),
        ("faceAngle", ctypes.c_float),
        ("forwardVelocity", ctypes.c_float),
        ("health", ctypes.c_int16),
        ("action", ctypes.c_uint32),
        ("animID", ctypes.c_int32),
        ("animFrame", ctypes.c_int16),
        ("flags", ctypes.c_uint32),
        ("particleFlags", ctypes.c_uint32),
        ("invincTimer", ctypes.c_int16),
    ]


class MarioExtraState(ctypes.Structure):
    """Mario's internal state that ``MarioState`` leaves out, 124 bytes on both sides.

    ``sm64_mario_extra_state`` exists because the tick struct carries what a renderer needs while
    this project needs what the physics decided: the floor under Mario, the action bookkeeping and
    the direction the stick was resolved into. These fields are copied out of ``gMarioState`` with
    no conversion at all, so the angles here are still s16 units while ``MarioState.faceAngle`` is
    radians.

    The layout is the load bearing part of this class. ctypes trusts the field list, so a field
    added to ``libsm64.h`` and not to this list makes the library write past the end of the
    buffer. Compare ``ctypes.sizeof`` against the C struct after any vendored update.

    Attributes:
        floorType: Surface type of the floor under Mario. The instant warp check reads this one.
        floorTerrain: Terrain type of that floor.
        floorHeight: Height of that floor, so ``position[1] - floorHeight`` is Mario's clearance.
        floorNormalY: y of its normal, 1.0 on flat ground and lower the steeper the slope.
        ceilHeight: Height of the ceiling above Mario.
        peakHeight: Highest point of the current jump.
        slideVelX: x of the sliding velocity the slide and jump actions integrate.
        slideVelZ: z of that sliding velocity.
        intendedMag: Stick magnitude the decompilation resolved, quadratic in the scaled stick:
            32.0 at full normalized deflection, 131072.0 if the raw +-64 is passed through. The
            moving actions clamp it to 8.0, which is why over deflection is invisible on the
            ground and violent in the air.
        intendedYaw: Direction the stick resolved to, s16 units, camera relative.
        input: Mario's own per frame input bitfield, not the controller's.
        action: Current action bitfield, the same value ``MarioState.action`` carries.
        prevAction: Action of the previous frame.
        actionState: The current action's state counter.
        actionTimer: Frames the current action has run.
        actionArg: Argument the action was entered with.
        hurtCounter: Damage still to be applied.
        squishTimer: Frames left of being squished.
        hasWall: Nonzero when collision found a wall this frame.
        hasFloor: Nonzero when collision found a floor this frame.
        gfxPosition: Position of the graphics node, which trails the physics position.
        gfxAngle: Angles of the graphics node, s16 units.
        faceAnglePitch: Pitch of Mario's facing, s16 units.
        faceAngleRoll: Roll of Mario's facing, s16 units.
        torsoAngle: Torso angles, s16 units.
    """

    _fields_ = [
        ("floorType", ctypes.c_int32),
        ("floorTerrain", ctypes.c_int32),
        ("floorHeight", ctypes.c_float),
        ("floorNormalY", ctypes.c_float),
        ("ceilHeight", ctypes.c_float),
        ("peakHeight", ctypes.c_float),
        ("slideVelX", ctypes.c_float),
        ("slideVelZ", ctypes.c_float),
        ("intendedMag", ctypes.c_float),
        ("intendedYaw", ctypes.c_int32),
        ("input", ctypes.c_int32),
        ("action", ctypes.c_int32),
        ("prevAction", ctypes.c_int32),
        ("actionState", ctypes.c_int32),
        ("actionTimer", ctypes.c_int32),
        ("actionArg", ctypes.c_int32),
        ("hurtCounter", ctypes.c_int32),
        ("squishTimer", ctypes.c_int32),
        ("hasWall", ctypes.c_int32),
        ("hasFloor", ctypes.c_int32),
        ("gfxPosition", ctypes.c_float * 3),
        ("gfxAngle", ctypes.c_int32 * 3),
        ("faceAnglePitch", ctypes.c_int32),
        ("faceAngleRoll", ctypes.c_int32),
        ("torsoAngle", ctypes.c_int32 * 3),
    ]


class Surface(ctypes.Structure):
    """One static collision triangle, mirroring ``struct SM64Surface``.

    The vertices are int32 here where the original game used int16, because libsm64 widened them.
    Winding is what decides whether a triangle is a floor or a ceiling, so both producers in this
    project, ``src.env.collision`` and ``src.env.geometry``, order the corners so that the y
    component of the edge cross product comes out positive.

    Attributes:
        type: Surface type, one of the SURFACE_ constants. Slipperiness and the instant warp that
            makes the staircase endless are both carried here rather than in the geometry.
        force: Movement force, used only by the surface types that push Mario. Zero throughout
            this project.
        terrain: Terrain type, which selects footstep sounds and the terrain dependent branches of
            the movement code.
        vertices: Three corners of three int32 world coordinates each.
    """

    _fields_ = [
        ("type", ctypes.c_int16),
        ("force", ctypes.c_int16),
        ("terrain", ctypes.c_uint16),
        ("vertices", (ctypes.c_int32 * 3) * 3),
    ]


class MarioGeometryBuffers(ctypes.Structure):
    """Destination for one tick of Mario's mesh, mirroring ``struct SM64MarioGeometryBuffers``.

    The caller owns the four arrays and ``sm64_mario_tick`` fills whichever ones it is handed,
    keeping no mesh state of its own between calls. That is what makes a population of Marios in
    one process possible, and ``src.env.swarm`` relies on it: every Mario keeps its own id and its
    own buffers, and the library writes into the pair it was given.

    The arrays are sized for GEO_MAX_TRIANGLES, so ``numTrianglesUsed`` is the only thing that
    says how much of each one the last tick actually wrote. Mario has come out at 752 triangles on
    every frame measured so far, but that is an observation about the model rather than a promise
    from the API, so readers should use the count.

    Attributes:
        position: Nine floats per triangle, three world space vertices of xyz.
        normal: Nine floats per triangle, one unit normal per vertex.
        color: Nine floats per triangle, the part's flat light color repeated on all three.
        uv: Six floats per triangle, one texture atlas coordinate per vertex.
        numTrianglesUsed: Triangles written by the most recent tick.
    """

    _fields_ = [
        ("position", ctypes.POINTER(ctypes.c_float)),
        ("normal", ctypes.POINTER(ctypes.c_float)),
        ("color", ctypes.POINTER(ctypes.c_float)),
        ("uv", ctypes.POINTER(ctypes.c_float)),
        ("numTrianglesUsed", ctypes.c_uint16),
    ]


def _library_name() -> str:
    return "libsm64.dylib" if sys.platform == "darwin" else "libsm64.so"


def _default_library_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "third_party", "libsm64", "dist", _library_name())


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.sm64_global_init.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
    lib.sm64_global_init.restype = None
    lib.sm64_global_terminate.argtypes = []
    lib.sm64_global_terminate.restype = None
    lib.sm64_static_surfaces_load.argtypes = [ctypes.POINTER(Surface), ctypes.c_uint32]
    lib.sm64_static_surfaces_load.restype = None
    lib.sm64_mario_create.argtypes = [ctypes.c_float] * 3
    lib.sm64_mario_create.restype = ctypes.c_int32
    lib.sm64_mario_tick.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(MarioInputs),
        ctypes.POINTER(MarioState),
        ctypes.POINTER(MarioGeometryBuffers),
    ]
    lib.sm64_mario_tick.restype = None
    lib.sm64_mario_delete.argtypes = [ctypes.c_int32]
    lib.sm64_mario_delete.restype = None
    lib.sm64_set_mario_position.argtypes = [ctypes.c_int32] + [ctypes.c_float] * 3
    lib.sm64_set_mario_position.restype = None
    lib.sm64_set_mario_faceangle.argtypes = [ctypes.c_int32, ctypes.c_float]
    lib.sm64_set_mario_faceangle.restype = None
    lib.sm64_set_mario_velocity.argtypes = [ctypes.c_int32] + [ctypes.c_float] * 3
    lib.sm64_set_mario_velocity.restype = None
    lib.sm64_set_mario_forward_velocity.argtypes = [ctypes.c_int32, ctypes.c_float]
    lib.sm64_set_mario_forward_velocity.restype = None
    lib.sm64_set_mario_action.argtypes = [ctypes.c_int32, ctypes.c_uint32]
    lib.sm64_set_mario_action.restype = None
    lib.sm64_set_mario_health.argtypes = [ctypes.c_int32, ctypes.c_uint16]
    lib.sm64_set_mario_health.restype = None
    lib.sm64_mario_extra_state.argtypes = [ctypes.c_int32, ctypes.POINTER(MarioExtraState)]
    lib.sm64_mario_extra_state.restype = None
    return lib


class Sm64:
    """A live libsm64: one ROM, one static surface set, one Mario.

    The library keeps its state in C statics, so there is one of everything per process however
    many of these objects exist, and ``close`` terminates the library for all of them. That is the
    reason ``src.env.blj_env`` documents itself as one environment per process and sends
    vectorized training through subprocesses.

    Holding a single Mario is this class's restriction rather than the library's.
    ``sm64_mario_create`` returns an id and every other entry point takes one, which is what
    ``src.env.swarm`` uses to drive a whole population; this class keeps one id and one reusable
    set of buffers because that is all one environment needs.

    Those buffers are allocated once and handed back by reference, so ``tick`` and ``extra_state``
    return the same two objects on every call. A caller that wants to keep a frame has to copy the
    fields out rather than store the struct.
    """

    def __init__(self, rom_path: str, library_path: str | None = None):
        """Loads the ROM, initializes the library and allocates the per frame buffers.

        The ROM is an asset source and not code: libsm64 reads Mario's animations, his model and
        the 704 by 64 texture atlas out of it during ``sm64_global_init`` and runs the decompiled
        logic from the shared library. Nothing on this side validates the image, so a wrong or
        byte swapped ROM is the library's problem rather than this binding's.

        Args:
            rom_path: Path to the Super Mario 64 US ROM.
            library_path: Path to the built shared library, or None for the one in
                ``third_party/libsm64/dist`` beside this checkout.

        Raises:
            OSError: If the ROM cannot be read or the shared library cannot be loaded.
        """
        with open(rom_path, "rb") as handle:
            rom = handle.read()
        self._lib = _bind(ctypes.CDLL(library_path or _default_library_path()))
        rom_buffer = (ctypes.c_uint8 * len(rom)).from_buffer_copy(rom)
        self._texture = (ctypes.c_uint8 * (TEXTURE_WIDTH * TEXTURE_HEIGHT * 4))()
        self._lib.sm64_global_init(rom_buffer, self._texture)
        self._geometry = MarioGeometryBuffers(
            position=(ctypes.c_float * (9 * GEO_MAX_TRIANGLES))(),
            normal=(ctypes.c_float * (9 * GEO_MAX_TRIANGLES))(),
            color=(ctypes.c_float * (9 * GEO_MAX_TRIANGLES))(),
            uv=(ctypes.c_float * (6 * GEO_MAX_TRIANGLES))(),
            numTrianglesUsed=0,
        )
        self._state = MarioState()
        self._extra = MarioExtraState()
        self._mario_id = -1

    def load_surfaces(self, surfaces: list[Surface]) -> None:
        """Replaces the process wide static collision set.

        The library holds one static surface list, freeing the previous one, so this is a
        replacement rather than an addition and it applies to every Mario in the process. Anything
        that loads a calibration plane before the real scene has to load the real scene again
        afterwards, which is the trap the viewer tools call out.

        Args:
            surfaces: Every triangle of the scene. The call copies them into C storage, so the
                list does not have to outlive it.
        """
        array = (Surface * len(surfaces))(*surfaces)
        self._lib.sm64_static_surfaces_load(array, len(surfaces))

    def create_mario(self, x: float, y: float, z: float) -> int:
        """Spawns Mario, replacing the one this handle already owns.

        Mario arrives in ACT_SPAWN_SPIN_AIRBORNE and does not reach ACT_IDLE until roughly frame
        23, so a caller that starts pressing buttons immediately is driving the spawn animation
        rather than Mario. The settle frames in ``src.agent.scripted`` are there for this reason.

        Args:
            x: Spawn x.
            y: Spawn y, a little above the floor.
            z: Spawn z.

        Returns:
            The new Mario's id.

        Raises:
            RuntimeError: If libsm64 could not initialize a Mario at that point. Raising here
                matters: ``sm64_mario_tick`` only logs and returns for an id that does not exist,
                so an unchecked -1 would look like a Mario frozen in place.
        """
        if self._mario_id >= 0:
            self._lib.sm64_mario_delete(self._mario_id)
        self._mario_id = self._lib.sm64_mario_create(x, y, z)
        if self._mario_id < 0:
            raise RuntimeError(f"sm64_mario_create failed at ({x}, {y}, {z})")
        return self._mario_id

    def tick(self, inputs: MarioInputs) -> MarioState:
        """Advances Mario by one frame at the game's 30 Hz.

        The geometry buffers are filled during the same call, so a renderer reading them after
        this returns sees the mesh for the position this state reports.

        Args:
            inputs: Controller state to hold for this frame, sticks normalized to [-1, 1].

        Returns:
            The state buffer this object owns, overwritten in place every call.
        """
        self._lib.sm64_mario_tick(
            self._mario_id, ctypes.byref(inputs), ctypes.byref(self._state),
            ctypes.byref(self._geometry))
        return self._state

    @property
    def state(self) -> MarioState:
        """Returns the buffer the most recent tick wrote, without ticking again.

        ``tick`` already returns this object, so this is for a caller that ticked indirectly and
        still needs the raw frame: the recorders and the mesh dumps step ``BljEnv``, which keeps
        the return value to build its observation from, and they want Mario's own numbers rather
        than the normalized vector.

        Returns:
            The state buffer this object owns, which the next tick overwrites in place.
        """
        return self._state

    def extra_state(self) -> MarioExtraState:
        """Reads the internal state the tick struct leaves out.

        This is a second call into the library rather than part of ``tick``, so it reports
        whatever Mario holds now. Call it after the tick whose floor and action are wanted.

        Returns:
            The extra state buffer this object owns, overwritten in place every call.
        """
        self._lib.sm64_mario_extra_state(self._mario_id, ctypes.byref(self._extra))
        return self._extra

    def set_position(self, x: float, y: float, z: float) -> None:
        """Teleports Mario, leaving his action and velocity alone.

        This is the primitive the instant warp is built from, which is why it does not touch
        anything else: the staircase's loop displaces Mario mid jump and the jump has to survive.

        Args:
            x: New x.
            y: New y.
            z: New z.
        """
        self._lib.sm64_set_mario_position(self._mario_id, x, y, z)

    def set_face_angle(self, yaw: float) -> None:
        """Sets Mario's facing.

        Args:
            yaw: Yaw in radians, in the same convention ``MarioState.faceAngle`` reports.
        """
        self._lib.sm64_set_mario_faceangle(self._mario_id, yaw)

    def set_forward_velocity(self, velocity: float) -> None:
        """Sets the signed speed along Mario's facing.

        Args:
            velocity: New forwardVel. Negative is backwards, which is the sign the exploit grows.
        """
        self._lib.sm64_set_mario_forward_velocity(self._mario_id, velocity)

    def set_velocity(self, x: float, y: float, z: float) -> None:
        """Sets Mario's world velocity.

        The world velocity and forwardVel are separate fields that the actions keep in step, so
        restoring a recorded frame has to write both.

        Args:
            x: New x velocity.
            y: New y velocity.
            z: New z velocity.
        """
        self._lib.sm64_set_mario_velocity(self._mario_id, x, y, z)

    def set_action(self, action: int) -> None:
        """Forces Mario into an action.

        This runs the decompilation's own ``set_mario_action``, so it is a transition and not an
        assignment: entering an airborne action scales forwardVel by 1.5 and rewrites the vertical
        velocity. ``BljEnv._restore`` depends on that ordering.

        Args:
            action: Action bitfield, one of the ACT_ constants.
        """
        self._lib.sm64_set_mario_action(self._mario_id, action)

    def set_health(self, health: int) -> None:
        """Sets Mario's health.

        Args:
            health: Health in the game's units, 0x880 being full.
        """
        self._lib.sm64_set_mario_health(self._mario_id, health)

    def close(self) -> None:
        """Deletes this Mario and terminates the library.

        The termination is process wide, so this ends every Mario in the process and not only the
        one this handle created. Idempotent for the Mario, which lets an environment close inside
        a finally block after a failed reset.
        """
        if self._mario_id >= 0:
            self._lib.sm64_mario_delete(self._mario_id)
            self._mario_id = -1
        self._lib.sm64_global_terminate()
