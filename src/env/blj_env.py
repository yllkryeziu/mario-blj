"""Gymnasium environment for the backwards long jump on the castle's endless staircase.

The task is to reach the top landing. Nothing but a backwards long jump can do it, because the
staircase loops Mario back down whenever his current floor is an instant warp triangle, and the
only way past that check is to cross the whole 154 unit zone inside one frame while Mario's
fastest ordinary movement is 48 units per frame. Success therefore certifies the exploit rather
than approximating it.

Reward is deliberately configurable, because the measurement this project is after is how much
shaping an agent needs before it finds the exploit. See ``src.train.ladder`` for the rungs.

libsm64 keeps one static surface set and one global audio and animation state per process, so a
process holds at most one environment. Vectorized training must therefore use subprocesses.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from src.env import endless_stairs
from src.env.endless_stairs import Scene
from src.env.native import (
    ACT_CROUCH_SLIDE,
    ACT_CROUCHING,
    ACT_LONG_JUMP,
    ACT_LONG_JUMP_LAND,
    ACT_START_CROUCHING,
    MarioInputs,
    Sm64,
    action_name,
)

OBSERVATION_SIZE = 24
STICK_DIRECTIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.0),
    (0.70710678, 0.70710678),
    (1.0, 0.0),
    (0.70710678, -0.70710678),
    (0.0, -1.0),
    (-0.70710678, -0.70710678),
    (-1.0, 0.0),
    (-0.70710678, 0.70710678),
)
ACTION_SIZE = len(STICK_DIRECTIONS) * 4

ACT_GROUP_MASK = 0x000001C0
ACT_GROUP_STATIONARY = 0x00000000
ACT_GROUP_MOVING = 0x00000040
ACT_GROUP_AIRBORNE = 0x00000080
ACT_FLAG_AIR = 0x00000800

_CROUCH_ACTIONS = frozenset({ACT_CROUCHING, ACT_START_CROUCHING, ACT_CROUCH_SLIDE})

STAGE_GROUNDED = 0
STAGE_CROUCHED = 1
STAGE_LONG_JUMP = 2
STAGE_BACKWARDS = 3
STAGE_CHAINED = 4

_BACKWARDS_LAUNCH_SPEED = -10.0
_POSITION_SCALE = 4000.0
_VELOCITY_SCALE = 50.0
_SPEED_SCALE = 200.0


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """Weights for each reward term.

    Attributes:
        terminal: Paid once when Mario reaches the top landing.
        speed_coefficient: Paid on every new record backward speed, proportional to the
            improvement. Shaping on the record rather than on the instantaneous value keeps it
            potential based, so an agent cannot farm it by hovering at one speed.
        curriculum_bonus: Paid once for each curriculum stage the episode reaches for the first
            time. This one names the method rather than the goal, so it is the most generous
            form of help the ladder offers and the least interesting rung to succeed at.
        height_coefficient: Paid on each new record height reached while standing on a floor,
            scaled so that climbing the whole staircase pays exactly this much. It states the
            goal, go up, and says nothing about how.

            Two details make it honest. It pays on the record rather than per frame, because the
            instant warp throws Mario back down and a per frame gain would pay him forever for
            re-climbing the same steps. And it only counts height while he has a floor under him,
            because a plain jump buys about 220 units of air that he could otherwise farm on the
            spot. The warp trigger sits at y 3960, so this term rises smoothly until Mario hits
            the barrier and then goes flat: every further unit of reward requires crossing the
            warp zone inside one frame, which requires the exploit.
        time_penalty: Subtracted every frame.
    """

    terminal: float = 1.0
    speed_coefficient: float = 0.0
    curriculum_bonus: float = 0.0
    height_coefficient: float = 0.0
    time_penalty: float = 0.0


@dataclasses.dataclass(frozen=True)
class InitialState:
    """A frame of Mario's state, enough to restart an episode from the middle of a chain.

    Attributes:
        position: World position.
        velocity: World velocity.
        forward_velocity: Signed speed along Mario's facing, the quantity the exploit grows.
        face_angle: Facing in radians.
        action: Mario action value.
    """

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    forward_velocity: float
    face_angle: float
    action: int


@dataclasses.dataclass(frozen=True)
class BljConfig:
    """Everything needed to build one environment.

    Attributes:
        rom_path: Path to the Super Mario 64 US ROM. libsm64 reads Mario's animation and texture
            data from it at load time.
        collision_path: Path to the level's ``collision.inc.c``.
        header_path: Path to ``surface_terrains.h`` for the surface type constants.
        reward: Reward weights.
        action_repeat: Frames each chosen action is held for.
        max_frames: Episode limit in frames, before action repeat.
        spawn_jitter: Uniform jitter in units applied to the spawn x and z.
        reset_states: Optional pool of states to restart from, sampled uniformly. Restarting from
            states along a demonstration is the standard remedy for an exploit that random
            exploration never stumbles into, and it is the top rung of the ladder rather than a
            default, because using it is exactly the kind of help this project is measuring.
        library_path: Optional explicit path to the built libsm64 shared library.
    """

    rom_path: str
    collision_path: str = endless_stairs._DEFAULT_COLLISION
    header_path: str = endless_stairs._DEFAULT_HEADER
    reward: RewardConfig = RewardConfig()
    action_repeat: int = 1
    max_frames: int = 3000
    spawn_jitter: float = 0.0
    reset_states: tuple[InitialState, ...] = ()
    library_path: str | None = None


def decode_action(action: int) -> tuple[float, float, bool, bool]:
    """Decodes a flat action index into stick and button state.

    Args:
        action: Index in ``[0, ACTION_SIZE)``.

    Returns:
        A tuple of stick x, stick y, whether A is held, whether Z is held.

    Raises:
        ValueError: If the index is out of range.
    """
    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"action {action} outside [0, {ACTION_SIZE})")
    buttons, direction = divmod(action, len(STICK_DIRECTIONS))
    stick_x, stick_y = STICK_DIRECTIONS[direction]
    return stick_x, stick_y, bool(buttons & 1), bool(buttons & 2)


class BljEnv(gymnasium.Env):
    """One Mario on one staircase, stepped a frame at a time."""

    metadata = {"render_modes": []}

    def __init__(self, config: BljConfig) -> None:
        """Loads the scene and starts libsm64.

        Args:
            config: Environment configuration.
        """
        self._config = config
        self._scene = endless_stairs.load_scene(config.collision_path, config.header_path)
        self._game = Sm64(config.rom_path, config.library_path)
        self._game.load_surfaces(self._scene.surfaces)
        self._inputs = MarioInputs()
        self._inputs.camLookX, self._inputs.camLookZ = 0.0, 1.0
        self._rng = np.random.default_rng()

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_SIZE)

        self._frames = 0
        self._peak_backward = 0.0
        self._stage = STAGE_GROUNDED
        self._warps = 0
        self._air_frames = 0
        self._last_action = 0
        self._previous_mario_action = 0
        self._previous_launch = 0.0
        self._success = False
        self._best_height = self._scene.spawn[1]
        self._climb = max(1.0, self._scene.goal_y - self._scene.spawn[1])

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Puts Mario back on the bottom landing.

        Args:
            seed: Seeds the spawn jitter.
            options: Unused, present for the Gymnasium signature.

        Returns:
            The first observation and its info dict.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        jitter = self._config.spawn_jitter
        spawn_x, spawn_y, spawn_z = self._scene.spawn
        if jitter:
            spawn_x += float(self._rng.uniform(-jitter, jitter))
            spawn_z += float(self._rng.uniform(-jitter, jitter))
        self._game.create_mario(spawn_x, spawn_y, spawn_z)
        if self._config.reset_states:
            self._restore(self._config.reset_states[
                int(self._rng.integers(len(self._config.reset_states)))])

        self._frames = 0
        self._peak_backward = 0.0
        self._stage = STAGE_GROUNDED
        self._warps = 0
        self._air_frames = 0
        self._last_action = 0
        self._previous_mario_action = 0
        self._previous_launch = 0.0
        self._success = False
        self._best_height = self._scene.spawn[1]

        state = self._game.tick(self._neutral_inputs())
        extra = self._game.extra_state()
        return self._observe(state, extra), self._info(state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Holds one action for ``action_repeat`` frames.

        Args:
            action: Index in ``[0, ACTION_SIZE)``.

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        stick_x, stick_y, press_a, press_z = decode_action(action)
        self._inputs.stickX, self._inputs.stickY = stick_x, stick_y
        self._inputs.buttonA = int(press_a)
        self._inputs.buttonB = 0
        self._inputs.buttonZ = int(press_z)
        self._last_action = action

        reward = 0.0
        terminated = False
        for _ in range(self._config.action_repeat):
            state = self._game.tick(self._inputs)
            extra = self._game.extra_state()
            self._frames += 1
            reward += self._accumulate(state, extra)
            if self._terminal(state):
                terminated = True
                break

        truncated = not terminated and self._frames >= self._config.max_frames
        return self._observe(state, extra), reward, terminated, truncated, self._info(state)

    def close(self) -> None:
        """Releases libsm64."""
        self._game.close()

    @property
    def game(self) -> Sm64:
        """Returns the live libsm64 handle, for recording and rendering tools.

        The recorders and the mesh dumps need what the environment's observation deliberately
        leaves out: Mario's raw state, the texture atlas and the geometry buffers libsm64 filled
        during the last tick. Exposing the handle read only is honest about that, and keeps those
        tools off ``_game``, where a rename would break them silently.

        This is not a channel for an agent. Anything reachable through here can move Mario
        without the environment's bookkeeping noticing, which would make the episode's rewards
        and its info dict disagree with what actually happened.

        Returns:
            The handle this environment ticks. Its state buffers are reused every frame.
        """
        return self._game

    @property
    def scene(self) -> Scene:
        """Returns the loaded staircase, for recording and rendering tools.

        The viewers draw the collision triangles and the warp zone, and both live here rather
        than in the observation. The scene is frozen and its surfaces are already loaded into the
        library, so reading it is safe; loading different surfaces through ``game`` is not.

        Returns:
            The scene this environment was built from.
        """
        return self._scene

    def _restore(self, initial: InitialState) -> None:
        """Puts Mario into a recorded state.

        Order matters. ``sm64_set_mario_action`` runs the decompilation's own ``set_mario_action``,
        which for an airborne action multiplies ``forwardVel`` by 1.5 and overwrites the vertical
        velocity, so the action goes first and the recorded numbers are written over the top of it.

        Args:
            initial: The state to restore.
        """
        self._game.set_action(initial.action)
        self._game.set_face_angle(initial.face_angle)
        self._game.set_forward_velocity(initial.forward_velocity)
        self._game.set_velocity(*initial.velocity)
        self._game.set_position(*initial.position)

    def _neutral_inputs(self) -> MarioInputs:
        self._inputs.stickX = self._inputs.stickY = 0.0
        self._inputs.buttonA = self._inputs.buttonB = self._inputs.buttonZ = 0
        return self._inputs

    def _accumulate(self, state: Any, extra: Any) -> float:
        """Advances episode bookkeeping for one frame and returns that frame's reward."""
        weights = self._config.reward
        reward = -weights.time_penalty

        if endless_stairs.apply_instant_warp(
                self._game, extra.floorType,
                (state.position[0], state.position[1], state.position[2]), self._scene.warp):
            self._warps += 1

        grounded = bool(extra.hasFloor) and state.position[1] - extra.floorHeight < 30.0
        if grounded and state.position[1] > self._best_height:
            reward += weights.height_coefficient * (
                (state.position[1] - self._best_height) / self._climb)
            self._best_height = state.position[1]

        velocity = state.forwardVelocity
        if velocity < self._peak_backward:
            reward += weights.speed_coefficient * (self._peak_backward - velocity)
            self._peak_backward = velocity

        if state.action == ACT_LONG_JUMP:
            self._air_frames += 1
        else:
            self._air_frames = 0

        stage = self._classify(state)
        if stage > self._stage:
            reward += weights.curriculum_bonus * (stage - self._stage)
            self._stage = stage

        if state.action == ACT_LONG_JUMP and self._previous_mario_action != ACT_LONG_JUMP:
            self._previous_launch = velocity
        self._previous_mario_action = state.action

        if not self._success and self._reached_goal(state):
            self._success = True
            reward += weights.terminal

        return reward

    def _classify(self, state: Any) -> int:
        """Returns the highest curriculum stage the episode has demonstrated."""
        mario_action = state.action
        if mario_action == ACT_LONG_JUMP and state.forwardVelocity <= _BACKWARDS_LAUNCH_SPEED:
            launched = self._previous_mario_action != ACT_LONG_JUMP
            grew = launched and self._previous_launch < _BACKWARDS_LAUNCH_SPEED and (
                state.forwardVelocity < self._previous_launch)
            return STAGE_CHAINED if grew else STAGE_BACKWARDS
        if mario_action in (ACT_LONG_JUMP, ACT_LONG_JUMP_LAND):
            return STAGE_LONG_JUMP
        if mario_action in _CROUCH_ACTIONS:
            return STAGE_CROUCHED
        return STAGE_GROUNDED

    def _reached_goal(self, state: Any) -> bool:
        return (state.position[2] <= self._scene.goal_z
                and state.position[1] >= self._scene.goal_y - 120.0)

    def _terminal(self, state: Any) -> bool:
        if self._success:
            return True
        if state.health <= 0:
            return True
        return state.position[1] < self._scene.spawn[1] - 2500.0

    def _observe(self, state: Any, extra: Any) -> np.ndarray:
        """Packs the frame into a normalized float32 vector."""
        position = state.position
        velocity = state.velocity
        mario_action = state.action
        group = mario_action & ACT_GROUP_MASK
        observation = np.array([
            position[0] / _POSITION_SCALE,
            (self._scene.goal_y - position[1]) / _POSITION_SCALE,
            (position[2] - self._scene.goal_z) / _POSITION_SCALE,
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
            float(extra.floorType == self._scene.warp.surface_type),
            (extra.peakHeight - position[1]) / 500.0,
            float(bool(mario_action & ACT_FLAG_AIR)),
            float(group == ACT_GROUP_MOVING),
            float(group == ACT_GROUP_STATIONARY),
            float(mario_action == ACT_LONG_JUMP),
            float(mario_action == ACT_LONG_JUMP_LAND),
            min(extra.actionTimer, 30) / 30.0,
            min(self._air_frames, 30) / 30.0,
            float(self._inputs.buttonA),
            float(self._inputs.buttonZ),
        ], dtype=np.float32)
        return np.clip(observation, -10.0, 10.0)

    def _info(self, state: Any) -> dict[str, Any]:
        return {
            "forward_velocity": float(state.forwardVelocity),
            "peak_backward_velocity": float(self._peak_backward),
            "best_height": float(self._best_height),
            "mario_action": action_name(state.action),
            "mario_action_id": int(state.action),
            "curriculum_stage": int(self._stage),
            "warps": int(self._warps),
            "height": float(state.position[1]),
            "success": bool(self._success),
            "frames": int(self._frames),
        }
