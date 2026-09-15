"""Many Marios on one staircase, ticked together so a renderer can watch them all improve.

libsm64 keeps one static surface list for the whole process, which is why ``BljEnv`` documents
itself as one environment per process. It does not keep one Mario. ``sm64_mario_create``
returns an id and every other entry point takes that id, so a single process can hold a whole
population of Marios standing on the same collision geometry, each with its own action, its own
episode and its own mesh. That is what this module is: one ROM, one surface load, N Marios, stepped
in lockstep.

The observation this module hands out is bit for bit the one the trained policies were fit on. The
arithmetic is duplicated from ``BljEnv._observe`` rather than imported, because that method is
private to a module this file does not own, and ``swarm_test`` pins the duplicate against the
original on a scripted run. A drifting observation would not crash anything, it would quietly turn
every rendered swarm into a lie about what the policy sees, so the equality test is the load bearing
part of this file.

Episodes auto reset. A slot that reaches the top landing, dies or runs out of frames records its
return, bumps its episode counter and respawns on the bottom landing within the same call, so the
swarm never stalls and the view stays full.
"""

from __future__ import annotations

import ctypes
import dataclasses
import math
from typing import cast

import numpy as np

from src.env import endless_stairs
from src.env.blj_env import (
    _BACKWARDS_LAUNCH_SPEED,
    _CROUCH_ACTIONS,
    _POSITION_SCALE,
    _SPEED_SCALE,
    _VELOCITY_SCALE,
    ACT_FLAG_AIR,
    ACT_GROUP_MASK,
    ACT_GROUP_MOVING,
    ACT_GROUP_STATIONARY,
    OBSERVATION_SIZE,
    STAGE_BACKWARDS,
    STAGE_CHAINED,
    STAGE_CROUCHED,
    STAGE_GROUNDED,
    STAGE_LONG_JUMP,
    RewardConfig,
    decode_action,
)
from src.env.native import (
    ACT_LONG_JUMP,
    ACT_LONG_JUMP_LAND,
    GEO_MAX_TRIANGLES,
    MarioExtraState,
    MarioGeometryBuffers,
    MarioInputs,
    MarioState,
    Sm64,
)

GROUNDED_TOLERANCE = 30.0
FALL_MARGIN = 2500.0
GOAL_TOLERANCE = 120.0
CORRIDOR_X_MIN = -372.0
CORRIDOR_X_MAX = -37.0


@dataclasses.dataclass(frozen=True)
class SwarmConfig:
    """Everything needed to build one populated room.

    Attributes:
        rom_path: Path to the Super Mario 64 US ROM, read once for animation and texture data.
        collision_path: Path to the level's ``collision.inc.c``. Defaults to the castle's area 2,
            the endless staircase.
        population: Number of Marios sharing the room. Each one owns a slot for the whole life of
            the swarm, so a renderer can key persistent state on the slot index.
        max_frames: Frames an episode may last before the slot truncates and respawns.
        reward: Reward weights, applied per slot with exactly the terms ``BljEnv`` uses.
        action_repeat: Frames each chosen action is held for, as in ``BljEnv``.
        spawn_spread: Lateral jitter in world units applied to the spawn x, across the corridor.
            Zero puts every Mario on the same tile, which looks like one Mario until they diverge.
            The jittered x is clamped into the corridor so nobody spawns inside a wall.
        header_path: Path to ``surface_terrains.h`` for the surface type constants.
        library_path: Optional explicit path to the built libsm64 shared library.
        seed: Seeds the spawn jitter. None draws from the OS entropy pool.
    """

    rom_path: str
    collision_path: str = endless_stairs._DEFAULT_COLLISION
    population: int = 32
    max_frames: int = 1200
    reward: RewardConfig = RewardConfig()
    action_repeat: int = 1
    spawn_spread: float = 0.0
    header_path: str = endless_stairs._DEFAULT_HEADER
    library_path: str | None = None
    seed: int | None = None


@dataclasses.dataclass
class Member:
    """One slot's public bookkeeping, snapshotted on the way out of :meth:`Swarm.step`.

    Attributes:
        index: Stable slot number in ``[0, population)``. Never changes.
        mario_id: The libsm64 id currently backing this slot. Changes on every auto reset, because
            a reset deletes the Mario and creates a new one.
        frames: Frames elapsed in the current episode.
        episodes: Episodes this slot has finished.
        best_height: Record height reached while grounded this episode, the quantity the height
            term pays on.
        peak_backward: Record (most negative) forward velocity this episode, the quantity the
            speed term pays on.
        warps: Times the instant warp has fired this episode.
        success: Whether this episode has reached the top landing.
        last_return: Return of the most recently finished episode, zero until the first one ends.
        episode_return: Return accumulated so far in the current episode.
    """

    index: int
    mario_id: int
    frames: int
    episodes: int
    best_height: float
    peak_backward: float
    warps: int
    success: bool
    last_return: float
    episode_return: float


@dataclasses.dataclass(frozen=True)
class MeshView:
    """Views onto one slot's live mesh buffers.

    The arrays alias the memory ``sm64_mario_tick`` writes into, so they cost nothing to hand out
    and everything to hold on to: the next :meth:`Swarm.step` overwrites them in place. Copy what
    you need before stepping again.

    Attributes:
        index: The slot these buffers belong to.
        num_triangles: Triangles libsm64 filled on the last tick, about 752 for a normal frame.
        position: World space vertex positions, shape ``(num_triangles, 3, 3)``.
        normal: Per vertex normals, shape ``(num_triangles, 3, 3)``.
        color: Per vertex colors, shape ``(num_triangles, 3, 3)``.
        uv: Per vertex texture coordinates, shape ``(num_triangles, 3, 2)``.
    """

    index: int
    num_triangles: int
    position: np.ndarray
    normal: np.ndarray
    color: np.ndarray
    uv: np.ndarray


class _MarioHandle:
    """The one method :func:`src.env.endless_stairs.apply_instant_warp` needs, bound to one id.

    ``apply_instant_warp`` takes a live game handle and moves Mario through it. The swarm has one
    library and many Marios, so it hands the warp check a handle that knows which Mario it is
    talking about instead of the process wide :class:`Sm64`.
    """

    def __init__(self, lib: ctypes.CDLL, mario_id: int) -> None:
        """Binds the handle.

        Args:
            lib: The bound libsm64 library.
            mario_id: Id returned by ``sm64_mario_create``.
        """
        self._lib = lib
        self._mario_id = mario_id

    def set_position(self, x: float, y: float, z: float) -> None:
        """Teleports this slot's Mario.

        Args:
            x: World x.
            y: World y.
            z: World z.
        """
        self._lib.sm64_set_mario_position(self._mario_id, x, y, z)


@dataclasses.dataclass
class _Slot:
    """Everything one Mario owns, public bookkeeping included.

    Attributes:
        member: The public record for this slot.
        inputs: Controller state held for this slot, also read back by the observation.
        state: Destination for ``sm64_mario_tick``.
        extra: Destination for ``sm64_mario_extra_state``.
        geometry: Buffer descriptor handed to ``sm64_mario_tick``.
        position: Numpy view of the position buffer, shape ``(GEO_MAX_TRIANGLES, 3, 3)``.
        normal: Numpy view of the normal buffer.
        color: Numpy view of the color buffer.
        uv: Numpy view of the uv buffer, shape ``(GEO_MAX_TRIANGLES, 3, 2)``.
        handle: Per Mario handle for the instant warp check.
        stage: Highest curriculum stage this episode has demonstrated.
        air_frames: Consecutive frames spent in the long jump action.
        previous_mario_action: Mario action value on the previous frame.
        previous_launch: Forward velocity at the previous long jump launch.
        terminated: Whether the episode ended on its own terms during the last step.
        truncated: Whether the episode ran out of frames during the last step.
    """

    member: Member
    inputs: MarioInputs
    state: MarioState
    extra: MarioExtraState
    geometry: MarioGeometryBuffers
    position: np.ndarray
    normal: np.ndarray
    color: np.ndarray
    uv: np.ndarray
    handle: _MarioHandle
    stage: int
    air_frames: int
    previous_mario_action: int
    previous_launch: float
    terminated: bool
    truncated: bool


def build_observation(state: MarioState, extra: MarioExtraState, scene: endless_stairs.Scene,
                      air_frames: int, button_a: int, button_z: int) -> np.ndarray:
    """Packs one frame into the exact vector ``BljEnv`` feeds its policies.

    This is a deliberate duplicate of ``BljEnv._observe``. The twenty four entries, their order,
    their scales and the clip are all load bearing, because a policy trained on that layout reads
    garbage from any other one. ``swarm_test`` asserts the duplicate agrees with the original on a
    scripted run, which is the only thing keeping the two honest.

    Args:
        state: Mario state from the last tick.
        extra: Extra state from the last tick, from the project's libsm64 patch.
        scene: The loaded staircase, for the goal and the warp surface type.
        air_frames: Consecutive frames spent in the long jump action.
        button_a: Whether A was held on the frame that produced this state.
        button_z: Whether Z was held on the frame that produced this state.

    Returns:
        A ``(24,)`` float32 vector, every entry clipped to ``[-10, 10]``.
    """
    position = state.position
    velocity = state.velocity
    mario_action = state.action
    group = mario_action & ACT_GROUP_MASK
    observation = np.array([
        position[0] / _POSITION_SCALE,
        (scene.goal_y - position[1]) / _POSITION_SCALE,
        (position[2] - scene.goal_z) / _POSITION_SCALE,
        velocity[0] / _VELOCITY_SCALE,
        velocity[1] / _VELOCITY_SCALE,
        velocity[2] / _VELOCITY_SCALE,
        state.forwardVelocity / _SPEED_SCALE,
        math.sin(state.faceAngle),
        math.cos(state.faceAngle),
        extra.floorNormalY,
        (position[1] - extra.floorHeight) / 200.0,
        float(extra.hasFloor),
        float(extra.hasWall),
        float(extra.floorType == scene.warp.surface_type),
        (extra.peakHeight - position[1]) / 500.0,
        float(bool(mario_action & ACT_FLAG_AIR)),
        float(group == ACT_GROUP_MOVING),
        float(group == ACT_GROUP_STATIONARY),
        float(mario_action == ACT_LONG_JUMP),
        float(mario_action == ACT_LONG_JUMP_LAND),
        min(extra.actionTimer, 30) / 30.0,
        min(air_frames, 30) / 30.0,
        float(button_a),
        float(button_z),
    ], dtype=np.float32)
    return np.clip(observation, -10.0, 10.0)


class Swarm:
    """A population of Marios on one shared staircase, stepped together.

    One :class:`src.env.native.Sm64` handle performs the global init and the single surface load.
    Everything per Mario goes through the bound library directly, because ``Sm64.create_mario``
    tracks a single id and deletes the previous Mario when it makes a new one.
    """

    def __init__(self, config: SwarmConfig) -> None:
        """Loads the scene, starts libsm64 and spawns the whole population.

        Args:
            config: Swarm configuration.

        Raises:
            ValueError: If the population is not positive or the action repeat is below one.
            RuntimeError: If libsm64 refuses to create one of the Marios.
        """
        if config.population < 1:
            raise ValueError(f"population {config.population} must be positive")
        if config.action_repeat < 1:
            raise ValueError(f"action_repeat {config.action_repeat} must be at least one")

        self._config = config
        self._scene = endless_stairs.load_scene(config.collision_path, config.header_path)
        self._game = Sm64(config.rom_path, config.library_path)
        self._lib = self._game._lib
        self._game.load_surfaces(self._scene.surfaces)
        self._rng = np.random.default_rng(config.seed)
        self._climb = max(1.0, self._scene.goal_y - self._scene.spawn[1])
        self._escape = max(1.0, endless_stairs.minimum_escape_speed(self._scene.warp))
        self._closed = False
        self._observations = np.zeros((config.population, OBSERVATION_SIZE), dtype=np.float32)
        self._slots = [self._build_slot(index) for index in range(config.population)]
        for slot in self._slots:
            self._begin_episode(slot)

    @property
    def population(self) -> int:
        """Returns the number of slots."""
        return self._config.population

    @property
    def game(self) -> Sm64:
        """Returns the shared libsm64 handle.

        Intended for recording and rendering tools rather than for agents. One handle serves the
        whole population, which is also why one audio tick per frame renders every Mario: the
        audio engine's state is a file scope global in libsm64 rather than part of the per Mario
        GlobalState, so every slot's play_sound calls land in the same request queue and the
        game's own mixer combines them.
        """
        return self._game

    @property
    def scene(self) -> endless_stairs.Scene:
        """Returns the staircase every Mario is standing on."""
        return self._scene

    def observations(self) -> np.ndarray:
        """Returns the latest observation for every slot.

        Returns:
            A fresh ``(population, 24)`` float32 array, row ``i`` belonging to slot ``i``. Rows for
            slots that just auto reset hold the observation of their new episode's first frame,
            which is what ``BljEnv.reset`` would have returned.
        """
        return self._observations.copy()

    def step(self, actions: np.ndarray) -> list[Member]:
        """Holds one action per slot for ``action_repeat`` frames, then resets whoever finished.

        Args:
            actions: Integer action indices, one per slot, shape ``(population,)``.

        Returns:
            One snapshot per slot, in slot order. A slot that finished its episode during this
            call is described by the episode that finished: ``frames``, ``best_height``,
            ``peak_backward``, ``warps``, ``success`` and ``episode_return`` are its final values,
            with ``episodes`` already bumped and ``last_return`` already recorded. The slot itself
            has already respawned, so :meth:`members` and :meth:`observations` afterwards describe
            the new episode.

        Raises:
            ValueError: If the action array has the wrong shape or holds an out of range index.
            RuntimeError: If the swarm is closed, or libsm64 refuses to respawn a Mario.
        """
        if self._closed:
            raise RuntimeError("step on a closed swarm")
        chosen = np.asarray(actions).reshape(-1)
        if chosen.shape[0] != self.population:
            raise ValueError(f"expected {self.population} actions, got {chosen.shape[0]}")

        for slot, action in zip(self._slots, chosen.tolist(), strict=True):
            self._apply_action(slot, int(action))
        for _ in range(self._config.action_repeat):
            for slot in self._slots:
                if not (slot.terminated or slot.truncated):
                    self._tick(slot)

        snapshots: list[Member] = []
        for slot in self._slots:
            if slot.terminated or slot.truncated:
                slot.member.last_return = slot.member.episode_return
                slot.member.episodes += 1
                snapshots.append(dataclasses.replace(slot.member))
                self._begin_episode(slot)
            else:
                self._write_observation(slot)
                snapshots.append(dataclasses.replace(slot.member))
        return snapshots

    def mesh(self, index: int) -> MeshView:
        """Returns views onto one slot's mesh, as of its last tick.

        Args:
            index: Slot number in ``[0, population)``.

        Returns:
            A :class:`MeshView` aliasing that slot's live buffers.

        Raises:
            IndexError: If the slot number is out of range.
        """
        if not 0 <= index < self.population:
            raise IndexError(f"slot {index} outside [0, {self.population})")
        slot = self._slots[index]
        used = int(slot.geometry.numTrianglesUsed)
        return MeshView(
            index=index,
            num_triangles=used,
            position=slot.position[:used],
            normal=slot.normal[:used],
            color=slot.color[:used],
            uv=slot.uv[:used],
        )

    def members(self) -> list[Member]:
        """Returns a snapshot of every slot's current bookkeeping, in slot order.

        Returns:
            One copy per slot, safe to keep across a step.
        """
        return [dataclasses.replace(slot.member) for slot in self._slots]

    def close(self) -> None:
        """Deletes every Mario and releases libsm64. Idempotent."""
        if self._closed:
            return
        for slot in self._slots:
            if slot.member.mario_id >= 0:
                self._lib.sm64_mario_delete(slot.member.mario_id)
                slot.member.mario_id = -1
        self._closed = True
        self._game.close()

    def _build_slot(self, index: int) -> _Slot:
        """Allocates one slot's buffers and bookkeeping, without spawning a Mario yet.

        Args:
            index: Slot number.

        Returns:
            A slot whose Mario id is still negative.
        """
        vertex_floats = 9 * GEO_MAX_TRIANGLES
        position = (ctypes.c_float * vertex_floats)()
        normal = (ctypes.c_float * vertex_floats)()
        color = (ctypes.c_float * vertex_floats)()
        uv = (ctypes.c_float * (6 * GEO_MAX_TRIANGLES))()
        inputs = MarioInputs()
        inputs.camLookX, inputs.camLookZ = 0.0, 1.0
        return _Slot(
            member=Member(index=index, mario_id=-1, frames=0, episodes=0,
                          best_height=self._scene.spawn[1], peak_backward=0.0, warps=0,
                          success=False, last_return=0.0, episode_return=0.0),
            inputs=inputs,
            state=MarioState(),
            extra=MarioExtraState(),
            geometry=MarioGeometryBuffers(position=position, normal=normal, color=color, uv=uv,
                                          numTrianglesUsed=0),
            position=np.ctypeslib.as_array(position).reshape(GEO_MAX_TRIANGLES, 3, 3),
            normal=np.ctypeslib.as_array(normal).reshape(GEO_MAX_TRIANGLES, 3, 3),
            color=np.ctypeslib.as_array(color).reshape(GEO_MAX_TRIANGLES, 3, 3),
            uv=np.ctypeslib.as_array(uv).reshape(GEO_MAX_TRIANGLES, 3, 2),
            handle=_MarioHandle(self._lib, -1),
            stage=STAGE_GROUNDED,
            air_frames=0,
            previous_mario_action=0,
            previous_launch=0.0,
            terminated=False,
            truncated=False,
        )

    def _begin_episode(self, slot: _Slot) -> None:
        """Respawns one slot and takes its first frame, mirroring ``BljEnv.reset``.

        Args:
            slot: The slot to restart.

        Raises:
            RuntimeError: If libsm64 refuses to create the Mario.
        """
        if slot.member.mario_id >= 0:
            self._lib.sm64_mario_delete(slot.member.mario_id)
            slot.member.mario_id = -1

        spawn_x, spawn_y, spawn_z = self._scene.spawn
        spread = self._config.spawn_spread
        if spread:
            spawn_x = float(np.clip(spawn_x + self._rng.uniform(-spread, spread),
                                    CORRIDOR_X_MIN, CORRIDOR_X_MAX))
        mario_id = int(self._lib.sm64_mario_create(spawn_x, spawn_y, spawn_z))
        if mario_id < 0:
            raise RuntimeError(f"sm64_mario_create failed at ({spawn_x}, {spawn_y}, {spawn_z})")

        slot.member.mario_id = mario_id
        slot.member.frames = 0
        slot.member.best_height = self._scene.spawn[1]
        slot.member.peak_backward = 0.0
        slot.member.warps = 0
        slot.member.success = False
        slot.member.episode_return = 0.0
        slot.handle = _MarioHandle(self._lib, mario_id)
        slot.stage = STAGE_GROUNDED
        slot.air_frames = 0
        slot.previous_mario_action = 0
        slot.previous_launch = 0.0
        slot.terminated = False
        slot.truncated = False

        slot.inputs.stickX = slot.inputs.stickY = 0.0
        slot.inputs.buttonA = slot.inputs.buttonB = slot.inputs.buttonZ = 0
        self._raw_tick(slot)
        self._write_observation(slot)

    def _apply_action(self, slot: _Slot, action: int) -> None:
        """Writes one decoded action into a slot's controller state.

        Args:
            slot: The slot to drive.
            action: Index in ``[0, ACTION_SIZE)``.

        Raises:
            ValueError: If the index is out of range.
        """
        stick_x, stick_y, press_a, press_z = decode_action(action)
        slot.inputs.stickX, slot.inputs.stickY = stick_x, stick_y
        slot.inputs.buttonA = int(press_a)
        slot.inputs.buttonB = 0
        slot.inputs.buttonZ = int(press_z)
        slot.terminated = False
        slot.truncated = False

    def _raw_tick(self, slot: _Slot) -> None:
        """Advances one Mario by a frame and refreshes its state and extra state.

        Args:
            slot: The slot to tick.
        """
        self._lib.sm64_mario_tick(slot.member.mario_id, ctypes.byref(slot.inputs),
                                  ctypes.byref(slot.state), ctypes.byref(slot.geometry))
        self._lib.sm64_mario_extra_state(slot.member.mario_id, ctypes.byref(slot.extra))

    def _tick(self, slot: _Slot) -> None:
        """Advances one slot by a frame, accounting reward and episode end.

        Args:
            slot: The slot to tick.
        """
        self._raw_tick(slot)
        slot.member.frames += 1
        slot.member.episode_return += self._accumulate(slot)
        if self._is_terminal(slot):
            slot.terminated = True
        elif slot.member.frames >= self._config.max_frames:
            slot.truncated = True

    def _accumulate(self, slot: _Slot) -> float:
        """Advances one slot's bookkeeping for a frame and returns that frame's reward.

        Duplicates ``BljEnv._accumulate`` term for term: height pays on the record reached while
        grounded, speed pays on the record backward velocity, the curriculum pays once per stage
        newly reached, and the terminal pays once.

        Args:
            slot: The slot that has just ticked.

        Returns:
            The frame's reward.
        """
        weights = self._config.reward
        state, extra, member = slot.state, slot.extra, slot.member
        reward = -weights.time_penalty

        if endless_stairs.apply_instant_warp(
                cast(Sm64, slot.handle), extra.floorType,
                (state.position[0], state.position[1], state.position[2]), self._scene.warp):
            member.warps += 1

        height_above_floor = state.position[1] - extra.floorHeight
        grounded = bool(extra.hasFloor) and height_above_floor < GROUNDED_TOLERANCE
        if grounded and state.position[1] > member.best_height:
            before = self._progress(member.best_height)
            member.best_height = state.position[1]
            reward += weights.height_weight * (self._progress(member.best_height) - before)

        velocity = state.forwardVelocity
        if velocity < member.peak_backward:
            before = self._speed_share(member.peak_backward)
            member.peak_backward = velocity
            reward += weights.speed_weight * (self._speed_share(velocity) - before)

        if state.action == ACT_LONG_JUMP:
            slot.air_frames += 1
        else:
            slot.air_frames = 0

        stage = self._classify(slot)
        if stage > slot.stage:
            reward += weights.curriculum_weight * (stage - slot.stage) / STAGE_CHAINED
            slot.stage = stage

        if state.action == ACT_LONG_JUMP and slot.previous_mario_action != ACT_LONG_JUMP:
            slot.previous_launch = velocity
        slot.previous_mario_action = state.action

        if not member.success and self._reached_goal(slot):
            member.success = True
            reward += weights.terminal

        return reward

    def _progress(self, height: float) -> float:
        """Returns how far up the staircase a height is, clamped to [0, 1].

        Mirrors ``BljEnv._progress`` so a swarm slot earns exactly what a single environment
        would, which the parity test in swarm_test.py pins.

        Args:
            height: World y.

        Returns:
            The fraction of the climb from the spawn to the goal.
        """
        return min(1.0, max(0.0, (height - self._scene.spawn[1]) / self._climb))

    def _speed_share(self, velocity: float) -> float:
        """Returns backwards speed as a fraction of what defeats the loop, clamped to [0, 1].

        Args:
            velocity: Signed forward velocity.

        Returns:
            The fraction of the escape speed reached.
        """
        return min(1.0, max(0.0, -velocity / self._escape))

    def _classify(self, slot: _Slot) -> int:
        """Returns the highest curriculum stage this slot's episode has demonstrated.

        Args:
            slot: The slot to classify.

        Returns:
            One of the ``STAGE_`` constants from :mod:`src.env.blj_env`.
        """
        state = slot.state
        mario_action = state.action
        if mario_action == ACT_LONG_JUMP and state.forwardVelocity <= _BACKWARDS_LAUNCH_SPEED:
            launched = slot.previous_mario_action != ACT_LONG_JUMP
            grew = launched and slot.previous_launch < _BACKWARDS_LAUNCH_SPEED and (
                state.forwardVelocity < slot.previous_launch)
            return STAGE_CHAINED if grew else STAGE_BACKWARDS
        if mario_action in (ACT_LONG_JUMP, ACT_LONG_JUMP_LAND):
            return STAGE_LONG_JUMP
        if mario_action in _CROUCH_ACTIONS:
            return STAGE_CROUCHED
        return STAGE_GROUNDED

    def _reached_goal(self, slot: _Slot) -> bool:
        """Returns whether this slot's Mario is standing on the top landing.

        Args:
            slot: The slot to test.

        Returns:
            True if the top landing has been reached.
        """
        return (slot.state.position[2] <= self._scene.goal_z
                and slot.state.position[1] >= self._scene.goal_y - GOAL_TOLERANCE)

    def _is_terminal(self, slot: _Slot) -> bool:
        """Returns whether this slot's episode has ended on its own terms.

        Args:
            slot: The slot to test.

        Returns:
            True on success, death, or falling out of the level.
        """
        if slot.member.success:
            return True
        if slot.state.health <= 0:
            return True
        return slot.state.position[1] < self._scene.spawn[1] - FALL_MARGIN

    def _write_observation(self, slot: _Slot) -> None:
        """Rebuilds one slot's observation row from its latest state.

        Args:
            slot: The slot to observe.
        """
        self._observations[slot.member.index] = build_observation(
            slot.state, slot.extra, self._scene, slot.air_frames, int(slot.inputs.buttonA),
            int(slot.inputs.buttonZ))
