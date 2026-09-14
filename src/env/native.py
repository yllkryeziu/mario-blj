import ctypes
import os
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

ACTION_NAMES = {
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


def action_name(action: int) -> str:
    return ACTION_NAMES.get(action, f"0x{action:08X}")

SURFACE_DEFAULT = 0x0000
TERRAIN_STONE = 0x0001


class MarioInputs(ctypes.Structure):
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


class Surface(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int16),
        ("force", ctypes.c_int16),
        ("terrain", ctypes.c_uint16),
        ("vertices", (ctypes.c_int32 * 3) * 3),
    ]


class MarioGeometryBuffers(ctypes.Structure):
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
    return lib


class Sm64:
    def __init__(self, rom_path: str, library_path: str | None = None):
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
        self._mario_id = -1

    def load_surfaces(self, surfaces: list[Surface]) -> None:
        array = (Surface * len(surfaces))(*surfaces)
        self._lib.sm64_static_surfaces_load(array, len(surfaces))

    def create_mario(self, x: float, y: float, z: float) -> int:
        if self._mario_id >= 0:
            self._lib.sm64_mario_delete(self._mario_id)
        self._mario_id = self._lib.sm64_mario_create(x, y, z)
        if self._mario_id < 0:
            raise RuntimeError(f"sm64_mario_create failed at ({x}, {y}, {z})")
        return self._mario_id

    def tick(self, inputs: MarioInputs) -> MarioState:
        self._lib.sm64_mario_tick(
            self._mario_id, ctypes.byref(inputs), ctypes.byref(self._state),
            ctypes.byref(self._geometry))
        return self._state

    def set_position(self, x: float, y: float, z: float) -> None:
        self._lib.sm64_set_mario_position(self._mario_id, x, y, z)

    def set_face_angle(self, yaw: float) -> None:
        self._lib.sm64_set_mario_faceangle(self._mario_id, yaw)

    def set_forward_velocity(self, velocity: float) -> None:
        self._lib.sm64_set_mario_forward_velocity(self._mario_id, velocity)

    def set_velocity(self, x: float, y: float, z: float) -> None:
        self._lib.sm64_set_mario_velocity(self._mario_id, x, y, z)

    def set_action(self, action: int) -> None:
        self._lib.sm64_set_mario_action(self._mario_id, action)

    def set_health(self, health: int) -> None:
        self._lib.sm64_set_mario_health(self._mario_id, health)

    def close(self) -> None:
        if self._mario_id >= 0:
            self._lib.sm64_mario_delete(self._mario_id)
            self._mario_id = -1
        self._lib.sm64_global_terminate()
