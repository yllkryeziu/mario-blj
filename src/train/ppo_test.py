"""Tests for the ladder table and the PPO harness wiring.

Nothing here needs a Super Mario 64 ROM or a built libsm64. The ladder tests are pure data
checks and run anywhere. The harness tests substitute a fake environment that implements the
observation and action spaces from the environment contract, which is enough to prove that
the hyperparameters construct a PPO, that one update runs end to end, and that the episode
recorder writes the artifacts a cluster run is collected from. They skip when
stable-baselines3 is not installed.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
from typing import TYPE_CHECKING, Any

import pytest

from src.train import ladder

sb3 = pytest.importorskip("stable_baselines3", reason="stable-baselines3 is not installed")
gymnasium = pytest.importorskip("gymnasium", reason="gymnasium is not installed")
np = pytest.importorskip("numpy", reason="numpy is not installed")

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.train import ppo

if TYPE_CHECKING:
    from numpy import ndarray

OBSERVATION_SIZE = 24
ACTION_COUNT = 36
REWARD_FIELDS = ("terminal", "speed_weight", "curriculum_weight", "height_weight", "time_penalty")


class FakeBljEnv(gymnasium.Env):
    """A stand in for ``BljEnv`` that honors the spaces and info dict of the contract.

    Episodes are short and deterministic so that a handful of PPO steps produces several
    finished episodes. Every ``success_every`` th episode reports a success, which gives the
    recorder something to compute a success rate and a first success from.

    Attributes:
        observation_space: Box of 24 normalized float32 values.
        action_space: Discrete over nine stick directions times A times Z.
    """

    def __init__(self, episode_length: int = 8, success_every: int = 3):
        """Initializes the fake environment.

        Args:
            episode_length: Steps before the episode terminates.
            success_every: Report a success on every multiple of this episode index.
        """
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(ACTION_COUNT)
        self._episode_length = episode_length
        self._success_every = success_every
        self._step = 0
        self._episode = 0

    def _observation(self) -> ndarray:
        """Returns a filler observation in the declared space.

        Returns:
            A float32 vector of the declared shape.
        """
        return np.full((OBSERVATION_SIZE,), 0.1, dtype=np.float32)

    def _info(self, success: bool) -> dict[str, Any]:
        """Builds an info dict carrying every key the contract promises.

        Args:
            success: Whether this episode reached the top landing.

        Returns:
            The info dict.
        """
        return {
            "forward_velocity": -20.0,
            "peak_backward_velocity": -100.0 - self._episode,
            "mario_action": "long_jump",
            "curriculum_stage": 3,
            "warps": 2,
            "height": 3200.0,
            "success": success,
            "frames": self._step,
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ndarray, dict[str, Any]]:
        """Resets the episode.

        Args:
            seed: Ignored beyond seeding the base class.
            options: Ignored.

        Returns:
            The first observation and an info dict.
        """
        super().reset(seed=seed)
        self._step = 0
        self._episode += 1
        return self._observation(), self._info(False)

    def step(self, action: int) -> tuple[ndarray, float, bool, bool, dict[str, Any]]:
        """Advances one step.

        Args:
            action: Discrete action index.

        Returns:
            Observation, reward, terminated, truncated and info.
        """
        self._step += 1
        terminated = self._step >= self._episode_length
        success = terminated and self._episode % self._success_every == 0
        reward = 1.0 if success else 0.01
        return self._observation(), reward, terminated, False, self._info(success)


def test_rung_table_is_complete() -> None:
    """The table holds exactly the ladder rungs, keyed by their own names.

    The expected names are written out rather than compared against ``LADDER_ORDER``, which would
    make the assertion tautological and unable to catch a rung being added or dropped.
    """
    assert tuple(ladder.RUNGS) == ("terminal", "speed", "height", "height_speed", "curriculum",
                                   "repeat")
    assert tuple(ladder.RUNGS) == ladder.LADDER_ORDER
    for name, rung in ladder.RUNGS.items():
        assert rung.name == name
        assert rung.description


def test_every_rung_pays_the_terminal_reward() -> None:
    """The terminal bonus is the one thing present on every rung."""
    for rung in ladder.RUNGS.values():
        assert rung.terminal == 1.0


def test_each_shaping_rung_adds_one_ingredient_to_terminal() -> None:
    """Every rung is the terminal rung plus a named set of additions.

    The ladder is not a single chain. ``height`` and ``speed`` are two independent branches off
    the terminal rung, and ``height_speed`` is their combination, so the invariant that localizes
    credit is that each rung's difference from ``terminal`` is exactly its own advertised
    ingredients rather than that consecutive rungs differ by one field.
    """
    expected = {
        "terminal": set(),
        "speed": {"speed_weight"},
        "height": {"height_weight"},
        "height_speed": {"height_weight", "speed_weight"},
        "curriculum": {"speed_weight", "curriculum_weight"},
        "repeat": {"speed_weight", "curriculum_weight"},
    }
    for name, fields in expected.items():
        assert _changed_fields(ladder.RUNG_TERMINAL, ladder.RUNGS[name]) == fields, name


def _changed_fields(lower: ladder.Rung, upper: ladder.Rung) -> set[str]:
    """Returns the experiment relevant fields that differ between two rungs.

    Name and description are excluded because they differ by construction and carry no
    experimental meaning.

    Args:
        lower: The rung below.
        upper: The rung above.

    Returns:
        Names of the differing fields.
    """
    ignored = {"name", "description"}
    return {
        field.name
        for field in dataclasses.fields(ladder.Rung)
        if field.name not in ignored
        and getattr(lower, field.name) != getattr(upper, field.name)
    }


def test_rung_zero_has_no_shaping() -> None:
    """Rung 0 is the unshaped control: terminal reward and nothing else."""
    rung = ladder.RUNG_TERMINAL
    assert rung.speed_weight == 0.0
    assert rung.curriculum_weight == 0.0
    assert rung.time_penalty == 0.0
    assert rung.action_repeat == 1


def test_rungs_are_frozen() -> None:
    """Rungs are immutable so a run cannot mutate the experiment out from under itself."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        ladder.RUNG_TERMINAL.terminal = 5.0  # type: ignore[misc]


def test_expand_action_repeat_produces_the_sweep() -> None:
    """The sweep yields one distinctly named rung per repeat, reward weights untouched."""
    variants = ladder.expand_action_repeat(ladder.RUNG_REPEAT)
    assert [rung.action_repeat for rung in variants] == [1, 2, 3, 4]
    assert [rung.name for rung in variants] == [
        "repeat_r1",
        "repeat_r2",
        "repeat_r3",
        "repeat_r4",
    ]
    for rung in variants:
        for field in REWARD_FIELDS:
            assert getattr(rung, field) == getattr(ladder.RUNG_CURRICULUM, field)


def test_expand_action_repeat_single_value_keeps_the_name() -> None:
    """A one element sweep has nothing to disambiguate, so the name is left alone."""
    variants = ladder.expand_action_repeat(ladder.RUNG_REPEAT, (2,))
    assert len(variants) == 1
    assert variants[0].name == "repeat"
    assert variants[0].action_repeat == 2


@pytest.mark.parametrize("repeats", [(), (1, 1), (0,), (-1,)])
def test_expand_action_repeat_rejects_bad_sweeps(repeats: tuple[int, ...]) -> None:
    """Empty, duplicated and non positive repeat values are refused."""
    with pytest.raises(ValueError):
        ladder.expand_action_repeat(ladder.RUNG_REPEAT, repeats)


def test_get_rung_resolves_expanded_names() -> None:
    """An expanded run name round trips back to its rung with the right repeat."""
    rung = ladder.get_rung("repeat_r3")
    assert rung.action_repeat == 3
    assert rung.curriculum_weight == ladder.RUNG_CURRICULUM.curriculum_weight
    with pytest.raises(KeyError):
        ladder.get_rung("nonexistent")


def test_ladder_runs_expands_only_the_sweep_rung() -> None:
    """The whole ladder is three fixed rungs plus one variant per repeat value."""
    names = ladder.run_names()
    assert names == ("terminal", "speed", "height", "height_speed", "curriculum",
                     "repeat_r1", "repeat_r2", "repeat_r3", "repeat_r4")


def test_job_matrix_varies_seeds_fastest() -> None:
    """Seeds vary fastest so all seeds of one rung form a contiguous array block."""
    matrix = ladder.job_matrix((0, 1, 2))
    assert len(matrix) == len(ladder.run_names()) * 3
    assert matrix[:3] == (("terminal", 0), ("terminal", 1), ("terminal", 2))
    assert matrix[3] == ("speed", 0)
    with pytest.raises(ValueError):
        ladder.job_matrix(())


def test_hyperparameters_defaults_are_sane() -> None:
    """The defaults validate and sit in the ranges a long horizon task needs."""
    hyperparameters = ppo.PpoHyperparameters()
    hyperparameters.validate()
    assert 1e-5 <= hyperparameters.learning_rate <= 1e-2
    assert 0.99 <= hyperparameters.gamma <= 1.0
    assert 0.8 <= hyperparameters.gae_lambda <= 1.0
    assert 0.0 < hyperparameters.ent_coef <= 0.05
    assert 0.05 <= hyperparameters.clip_range <= 0.3
    assert hyperparameters.n_steps >= 128
    assert hyperparameters.device == "cpu"


@pytest.mark.parametrize(
    "override",
    [
        {"learning_rate": 0.0},
        {"gamma": 1.5},
        {"gae_lambda": 0.0},
        {"clip_range": 1.0},
        {"ent_coef": 1.0},
        {"n_steps": 0},
        {"batch_size": 0},
        {"n_epochs": 0},
        {"max_grad_norm": 0.0},
        {"net_arch": ()},
    ],
)
def test_hyperparameters_reject_out_of_range(override: dict[str, Any]) -> None:
    """Every guarded field refuses an out of range value."""
    hyperparameters = dataclasses.replace(ppo.PpoHyperparameters(), **override)
    with pytest.raises(ValueError):
        hyperparameters.validate()


def test_resolved_batch_size_divides_the_buffer() -> None:
    """The batch size is shrunk to a divisor so PPO never silently truncates a minibatch."""
    hyperparameters = ppo.PpoHyperparameters(n_steps=100, batch_size=64)
    assert hyperparameters.buffer_size(3) == 300
    resolved = hyperparameters.resolved_batch_size(3)
    assert resolved <= 64
    assert 300 % resolved == 0
    exact = ppo.PpoHyperparameters(n_steps=512, batch_size=256)
    assert exact.resolved_batch_size(8) == 256


def test_build_vec_env_rejects_empty_vector() -> None:
    """A vector of zero environments is a configuration error, caught before any spawn."""
    with pytest.raises(ValueError):
        ppo.build_vec_env(
            ladder.RUNG_TERMINAL,
            rom_path="missing.z64",
            collision_path="missing.inc.c",
            seed=0,
            num_envs=0,
        )


def test_run_directory_layout() -> None:
    """Runs are addressed by rung then seed, which is what the collector globs."""
    path = ppo.run_directory("/out", "repeat_r3", 7)
    assert path == pathlib.Path("/out/repeat_r3/seed_7")


def test_ppo_builds_on_the_contract_spaces() -> None:
    """A PPO constructs against the 24 float observation and 36 way discrete action."""
    vec_env = DummyVecEnv([lambda: FakeBljEnv()])
    try:
        model = ppo.build_model(
            vec_env, seed=0, hyperparameters=ppo.PpoHyperparameters(n_steps=32, batch_size=16)
        )
        assert model.action_space == gymnasium.spaces.Discrete(ACTION_COUNT)
        assert model.observation_space.shape == (OBSERVATION_SIZE,)
    finally:
        vec_env.close()


def test_one_ppo_update_runs_and_records_episodes(tmp_path: pathlib.Path) -> None:
    """The harness wires up: one update runs and the episode CSV is written as it goes."""
    hyperparameters = ppo.PpoHyperparameters(
        n_steps=32, batch_size=16, n_epochs=1, net_arch=(16, 16)
    )
    vec_env = DummyVecEnv([lambda: Monitor(FakeBljEnv()) for _ in range(2)])
    csv_path = tmp_path / "episodes.csv"
    recorder = ppo.EpisodeRecorder(csv_path, action_repeat=2)
    try:
        model = ppo.build_model(vec_env, seed=0, hyperparameters=hyperparameters)
        model.learn(total_timesteps=64, callback=recorder)
    finally:
        vec_env.close()

    assert recorder.episodes > 0
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == recorder.episodes
    assert tuple(rows[0]) == ppo.EPISODE_CSV_FIELDS
    assert float(rows[0]["peak_backward_velocity"]) < 0.0
    assert int(rows[0]["frames"]) == int(rows[0]["timesteps"]) * 2


def test_metrics_capture_the_headline_numbers(tmp_path: pathlib.Path) -> None:
    """Summary carries success rate, peak velocity, warps and frames to first success."""
    hyperparameters = ppo.PpoHyperparameters(
        n_steps=32, batch_size=16, n_epochs=1, net_arch=(16, 16)
    )
    vec_env = DummyVecEnv([lambda: Monitor(FakeBljEnv(episode_length=4, success_every=2))])
    recorder = ppo.EpisodeRecorder(tmp_path / "episodes.csv", action_repeat=3)
    try:
        model = ppo.build_model(vec_env, seed=0, hyperparameters=hyperparameters)
        model.learn(total_timesteps=64, callback=recorder)
    finally:
        vec_env.close()

    metrics = recorder.summarize("repeat_r3", seed=5, total_timesteps=64)
    assert metrics.rung == "repeat_r3"
    assert metrics.seed == 5
    assert metrics.action_repeat == 3
    assert metrics.episodes >= 2
    assert 0.0 < metrics.success_rate <= 1.0
    assert metrics.peak_backward_velocity < 0.0
    assert metrics.mean_warps == pytest.approx(2.0)
    assert metrics.max_curriculum_stage == 3
    timesteps_to_first_success = metrics.timesteps_to_first_success
    assert timesteps_to_first_success is not None
    assert metrics.frames_to_first_success == timesteps_to_first_success * 3

    path = tmp_path / "metrics.json"
    ppo.write_metrics(path, metrics, ladder.get_rung("repeat_r3"))
    payload = ppo.load_metrics(path)
    assert payload["rung_config"]["action_repeat"] == 3
    assert payload["metrics"]["success_rate"] == metrics.success_rate
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_metrics_survive_a_run_with_no_finished_episode() -> None:
    """A job killed before any episode ended still summarizes without dividing by zero."""
    recorder = ppo.EpisodeRecorder(pathlib.Path("unused.csv"), action_repeat=1)
    metrics = recorder.summarize("terminal", seed=0, total_timesteps=0)
    assert metrics.episodes == 0
    assert metrics.success_rate == 0.0
    assert metrics.recent_success_rate == 0.0
    assert metrics.frames_to_first_success is None
    assert metrics.timesteps_to_first_success is None
