"""Tests for the multi Mario swarm in :mod:`src.env.swarm`.

The test that matters is :func:`test_observation_matches_env_on_scripted_run`. The swarm duplicates
``BljEnv._observe`` because that method is private to a module the swarm does not own, and a
duplicate that drifts would not raise anything, it would just quietly render a policy reacting to
numbers it was never trained on. So the duplicate is pinned against the original frame by frame on
a scripted backwards long jump, and a second test pins the whole swarm pipeline, reward included,
against a single environment driven with the same actions.

Everything that needs the native library is skipped unless both the built dylib and a ROM are
present. The layout and validation tests run anywhere.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import numpy as np
import pytest

from src.agent.drivers import Driver, scripted_driver
from src.env import endless_stairs
from src.env import swarm as swarm_module
from src.env.blj_env import ACTION_SIZE, BljConfig, BljEnv, RewardConfig
from src.env.native import MarioExtraState, MarioState
from src.env.swarm import OBSERVATION_SIZE, Member, Swarm, SwarmConfig, build_observation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROM_PATH = os.path.join(REPO_ROOT, "roms", "baserom.us.z64")
DYLIB_PATH = os.path.join(REPO_ROOT, "third_party", "libsm64", "dist", "libsm64.dylib")
SO_PATH = os.path.join(REPO_ROOT, "third_party", "libsm64", "dist", "libsm64.so")

needs_engine = pytest.mark.skipif(
    not (os.path.exists(ROM_PATH) and (os.path.exists(DYLIB_PATH) or os.path.exists(SO_PATH))),
    reason="needs roms/baserom.us.z64 and a built libsm64; run scripts/setup.sh")

SHAPED_REWARD = RewardConfig(terminal=1.0, speed_weight=0.25, curriculum_weight=0.25,
                             height_weight=0.25, time_penalty=0.001)
PARITY_FRAMES = 240
MESH_TRIANGLES = 752
OBSERVATION_TOLERANCE = 1e-3


def _synthetic_scene() -> endless_stairs.Scene:
    """Builds a scene with the real staircase's numbers but no collision data.

    Returns:
        A scene usable by :func:`build_observation` without touching the filesystem.
    """
    warp = endless_stairs.WarpZone(
        surface_type=endless_stairs.SURFACE_INSTANT_WARP_1B,
        displacement=(0.0, -205.0, 410.0),
        x_range=(-409.0, 0.0),
        y_range=(3917.0, 3994.0),
        z_range=(905.0, 1059.0),
    )
    return endless_stairs.Scene(surfaces=[], spawn=(-200.0, 3204.0, 3000.0), goal_y=4966.0,
                                goal_z=-1000.0, warp=warp, ascends_toward=(0.0, -1.0))


def _synthetic_frame(height: float, forward_velocity: float) -> tuple[MarioState, MarioExtraState]:
    """Builds one plausible frame of Mario state without running the engine.

    Args:
        height: World y to place Mario at.
        forward_velocity: Signed speed along Mario's facing.

    Returns:
        A state and an extra state, filled in enough for the observation.
    """
    state = MarioState()
    state.position[0], state.position[1], state.position[2] = -200.0, height, 3000.0
    state.velocity[0], state.velocity[1], state.velocity[2] = 1.0, -2.0, 3.0
    state.faceAngle = 0.5
    state.forwardVelocity = forward_velocity
    state.health = 0x880
    extra = MarioExtraState()
    extra.floorHeight = height - 10.0
    extra.floorNormalY = 1.0
    extra.hasFloor = 1
    extra.peakHeight = height + 50.0
    extra.actionTimer = 12
    extra.floorType = endless_stairs.SURFACE_INSTANT_WARP_1B
    return state, extra


def _replay_driver(actions: list[int]) -> Driver:
    """Wraps a fixed action sequence as a driver.

    Args:
        actions: One action index per step, in order.

    Returns:
        A driver that ignores what it sees and replays the sequence.
    """
    step = itertools.count()

    def drive(observation: Any, info: dict[str, Any]) -> int:
        del observation, info
        return actions[next(step)]

    return drive


def _run_environment(
        actions: list[int] | None) -> tuple[list[np.ndarray], list[float], list[tuple]]:
    """Drives one :class:`BljEnv` and records everything needed to rebuild its observations.

    Args:
        actions: Fixed action sequence, or None to drive the scripted backwards long jump expert.

    Returns:
        A tuple of the observations the environment returned (reset included), the per step
        rewards, and one ``(state, extra, air_frames, button_a, button_z, action)`` record per
        step, snapshotted out of the environment's own ctypes buffers.
    """
    env = BljEnv(BljConfig(rom_path=ROM_PATH, reward=SHAPED_REWARD, max_frames=1000))
    try:
        observation, info = env.reset(seed=0)
        observations = [observation.copy()]
        rewards: list[float] = []
        frames: list[tuple] = []
        driver = scripted_driver("up") if actions is None else _replay_driver(actions)
        for _ in range(PARITY_FRAMES):
            action = driver(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            observations.append(observation.copy())
            rewards.append(reward)
            frames.append((MarioState.from_buffer_copy(env._game._state),
                           MarioExtraState.from_buffer_copy(env._game._extra),
                           env._air_frames, int(env._inputs.buttonA), int(env._inputs.buttonZ),
                           action))
            if terminated or truncated:
                break
        return observations, rewards, frames
    finally:
        env.close()


def test_config_defaults_match_the_agreed_api() -> None:
    """The next two stages construct this config positionally, so its defaults are frozen."""
    config = SwarmConfig(rom_path="rom.z64")
    assert config.population == 32
    assert config.max_frames == 1200
    assert config.action_repeat == 1
    assert config.spawn_spread == 0.0
    assert config.reward == RewardConfig()
    assert config.collision_path.endswith(os.path.join("areas", "2", "collision.inc.c"))


def test_member_carries_the_agreed_fields() -> None:
    """A renderer reads these by name, so the field set is part of the contract."""
    assert [field.name for field in Member.__dataclass_fields__.values()] == [
        "index", "mario_id", "frames", "episodes", "best_height", "peak_backward", "warps",
        "success", "last_return", "episode_return"]


def test_swarm_rejects_an_empty_population() -> None:
    """A population below one has no meaning and would silently render nothing."""
    with pytest.raises(ValueError, match="population"):
        Swarm(SwarmConfig(rom_path="rom.z64", population=0))


def test_swarm_rejects_a_zero_action_repeat() -> None:
    """A zero repeat would step without ticking, so the swarm would freeze."""
    with pytest.raises(ValueError, match="action_repeat"):
        Swarm(SwarmConfig(rom_path="rom.z64", action_repeat=0))


def test_observation_is_normalized_and_clipped() -> None:
    """Every entry stays inside the box the policies were trained against."""
    scene = _synthetic_scene()
    state, extra = _synthetic_frame(3204.0, -400.0)
    observation = build_observation(state, extra, scene, air_frames=99, button_a=1, button_z=1)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(np.abs(observation) <= 10.0)
    assert observation[13] == pytest.approx(1.0)
    assert observation[21] == pytest.approx(1.0)
    assert observation[22] == pytest.approx(1.0)
    assert observation[23] == pytest.approx(1.0)


def test_observation_reads_height_as_distance_to_the_goal() -> None:
    """Climbing must move the height entry toward zero, or shaping would read backwards."""
    scene = _synthetic_scene()
    low = build_observation(*_synthetic_frame(3204.0, 0.0), scene, 0, 0, 0)
    high = build_observation(*_synthetic_frame(4900.0, 0.0), scene, 0, 0, 0)
    assert high[1] < low[1]
    assert high[1] == pytest.approx((4966.0 - 4900.0) / 4000.0, abs=1e-6)


@needs_engine
def test_observation_matches_env_on_scripted_run() -> None:
    """The duplicated observation must equal ``BljEnv._observe`` on a real scripted run.

    This is the point of the whole test file. The states come out of the environment's own buffers,
    so both constructions see byte identical inputs and any disagreement is arithmetic drift.
    """
    observations, _, frames = _run_environment(None)
    assert len(frames) >= 50
    scene = endless_stairs.load_scene()
    worst = 0.0
    for record, expected in zip(frames, observations[1:], strict=True):
        state, extra, air_frames, button_a, button_z, _ = record
        mine = build_observation(state, extra, scene, air_frames, button_a, button_z)
        worst = max(worst, float(np.max(np.abs(mine - expected))))
    assert worst < OBSERVATION_TOLERANCE, f"observation drifted by {worst}"


@needs_engine
def test_swarm_of_one_reproduces_the_environment() -> None:
    """A one Mario swarm fed the environment's actions must follow it observation for observation.

    Reward is compared too, because the height and speed terms pay on records and an off by one
    frame in the bookkeeping would show up nowhere else.
    """
    actions = [(index * 7 + 3) % ACTION_SIZE for index in range(PARITY_FRAMES)]
    observations, rewards, _ = _run_environment(actions)
    swarm = Swarm(SwarmConfig(rom_path=ROM_PATH, population=1, reward=SHAPED_REWARD,
                              max_frames=1000))
    try:
        assert np.max(np.abs(swarm.observations()[0] - observations[0])) < OBSERVATION_TOLERANCE
        worst = 0.0
        member = swarm.members()[0]
        for step, action in enumerate(actions[:len(rewards)]):
            member = swarm.step(np.array([action]))[0]
            worst = max(worst,
                        float(np.max(np.abs(swarm.observations()[0] - observations[step + 1]))))
        assert worst < OBSERVATION_TOLERANCE, f"trajectory drifted by {worst}"
        assert member.episode_return == pytest.approx(sum(rewards), abs=1e-6)
        assert member.frames == len(rewards)
    finally:
        swarm.close()


@needs_engine
def test_slots_are_independent() -> None:
    """Eight slots held on the eight stick directions must end in eight different places.

    Slot zero holds the stick up the staircase, so it also proves the shared warp geometry fires
    per Mario: it climbs, loops, and climbs again while the other seven wander the landing.
    """
    population = 8
    swarm = Swarm(SwarmConfig(rom_path=ROM_PATH, population=population, max_frames=600))
    try:
        actions = np.arange(1, population + 1)
        for _ in range(180):
            swarm.step(actions)
        centroids = {tuple(np.round(swarm.mesh(index).position.mean(axis=(0, 1)), 3))
                     for index in range(population)}
        assert len(centroids) == population
        assert len({member.mario_id for member in swarm.members()}) == population
        assert all(swarm.mesh(index).num_triangles == MESH_TRIANGLES
                   for index in range(population))
        assert swarm.members()[0].warps > 0
    finally:
        swarm.close()


@needs_engine
def test_auto_reset_advances_episodes_and_records_returns() -> None:
    """Slots must restart themselves and keep their per slot history across episodes.

    The observation row of a slot that just reset belongs to its new episode, the way a
    vectorized environment's autoreset works, so the air frame counter and both button entries
    are back to zero on the frame the reset is reported.
    """
    population = 4
    episode_frames = 40
    swarm = Swarm(SwarmConfig(rom_path=ROM_PATH, population=population,
                              max_frames=episode_frames, reward=SHAPED_REWARD, spawn_spread=80.0,
                              seed=3))
    try:
        actions = np.array([3, 3, 12, 12])
        finished: list[Member] = []
        for _ in range(episode_frames * 3):
            for member in swarm.step(actions):
                if member.frames == episode_frames:
                    finished.append(member)
                    if member.index == 0:
                        fresh = swarm.observations()[0]
                        assert np.all(fresh[21:24] == 0.0)
        assert len(finished) == population * 3
        for member in swarm.members():
            assert member.episodes == 3
            assert member.frames < episode_frames
            assert member.last_return == pytest.approx(
                [done for done in finished if done.index == member.index][-1].episode_return)
        assert all(member.last_return < 0.0 for member in swarm.members())
    finally:
        swarm.close()


@needs_engine
def test_mesh_views_alias_the_live_buffers() -> None:
    """The renderer gets views, not copies, so a step must show through the arrays it holds."""
    swarm = Swarm(SwarmConfig(rom_path=ROM_PATH, population=2, max_frames=600))
    try:
        view = swarm.mesh(0)
        before = view.position[0, 0].copy()
        for _ in range(30):
            swarm.step(np.array([3, 3]))
        assert not np.allclose(before, view.position[0, 0])
        assert view.position.base is not None
    finally:
        swarm.close()


@needs_engine
def test_spawn_spread_scatters_across_the_corridor() -> None:
    """Jitter must move Marios sideways without pushing any of them into a wall."""
    population = 16
    swarm = Swarm(SwarmConfig(rom_path=ROM_PATH, population=population, spawn_spread=150.0,
                              seed=11))
    try:
        xs = swarm.observations()[:, 0] * 4000.0
        assert len(np.unique(np.round(xs, 3))) == population
        assert np.all(xs >= swarm_module.CORRIDOR_X_MIN - 1.0)
        assert np.all(xs <= swarm_module.CORRIDOR_X_MAX + 1.0)
    finally:
        swarm.close()
