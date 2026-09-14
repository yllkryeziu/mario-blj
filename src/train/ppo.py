"""Proximal policy optimization harness for the backwards long jump ladder.

The algorithm is stable-baselines3 PPO, used as a dependency rather than reimplemented. The
project rule is to validate against a known-good reference before sweeping parameters, and a
hand rolled PPO would put two unvalidated things in series: if a rung failed we could not say
whether the reward shaping was insufficient or the optimizer was wrong. This module owns the
vectorization, the logging and the metric extraction, and nothing else.

Vectorization uses ``SubprocVecEnv`` with the spawn start method because one libsm64 process
can hold exactly one static surface set and the library is not thread safe. One operating
system process per environment is the only arrangement that gives each environment its own
copy of the staircase. Spawn rather than fork because forking a process that has already
opened a native shared library inherits its global state.

Outputs per run, all written under a single directory so a Slurm array task is self
contained:

* ``episodes.csv`` appended as episodes finish, so a job killed by a wall clock timeout
  still leaves a usable learning curve.
* ``metrics.json`` written at the end, carrying success rate, the peak backwards velocity
  reached anywhere in the run, mean warps per episode and frames to first success.
* ``model.zip`` the final policy, plus periodic ``checkpoints/`` so a timeout is not fatal.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import pathlib
from collections.abc import Callable
from typing import Any

import gymnasium
import numpy as np
from absl import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.train import ladder

EPISODE_CSV_FIELDS: tuple[str, ...] = (
    "episode",
    "timesteps",
    "frames",
    "return",
    "length",
    "success",
    "peak_backward_velocity",
    "warps",
    "curriculum_stage",
)


@dataclasses.dataclass(frozen=True)
class PpoHyperparameters:
    """PPO settings for a long horizon, sparse reward, 36 way discrete task.

    The defaults are chosen for the shape of this problem rather than tuned. An episode runs
    up to 3000 frames and the terminal reward at rung 0 is the only signal, so the discount
    is close to one and the entropy bonus is deliberately high: with 36 actions and a reward
    that only arrives on reaching the top landing, a policy that collapses early explores
    nothing. The network is small because the observation is 24 normalized floats, which
    also keeps this CPU bound alongside the physics.

    Attributes:
        learning_rate: Adam step size.
        n_steps: Rollout length per environment per update.
        batch_size: Minibatch size for the epochs of gradient descent.
        n_epochs: Passes over each rollout buffer.
        gamma: Discount factor, near one for the long horizon.
        gae_lambda: Generalized advantage estimation trace decay.
        clip_range: PPO policy ratio clip.
        ent_coef: Entropy bonus weight, high to sustain exploration.
        vf_coef: Value loss weight.
        max_grad_norm: Gradient clipping threshold.
        net_arch: Hidden layer widths shared by policy and value heads.
        device: Torch device. CPU is correct here, the bottleneck is the physics.
    """

    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple[int, ...] = (256, 256)
    device: str = "cpu"

    def validate(self) -> None:
        """Checks the hyperparameters are internally consistent and in sane ranges.

        Raises:
            ValueError: If a value is out of range, or if the rollout buffer produced by
                ``n_steps`` times the environment count cannot be divided into minibatches
                of ``batch_size``.
        """
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError(f"learning_rate out of range: {self.learning_rate}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be positive: {self.n_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be positive: {self.n_epochs}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma out of range: {self.gamma}")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda out of range: {self.gae_lambda}")
        if not 0.0 < self.clip_range < 1.0:
            raise ValueError(f"clip_range out of range: {self.clip_range}")
        if not 0.0 <= self.ent_coef < 1.0:
            raise ValueError(f"ent_coef out of range: {self.ent_coef}")
        if not 0.0 <= self.vf_coef <= 10.0:
            raise ValueError(f"vf_coef out of range: {self.vf_coef}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be positive: {self.max_grad_norm}")
        if not self.net_arch or any(width < 1 for width in self.net_arch):
            raise ValueError(f"net_arch must be positive widths: {self.net_arch}")

    def buffer_size(self, num_envs: int) -> int:
        """Reports the rollout buffer size for a given environment count.

        Args:
            num_envs: Number of parallel environments.

        Returns:
            The number of transitions collected per PPO update.
        """
        return self.n_steps * num_envs

    def resolved_batch_size(self, num_envs: int) -> int:
        """Returns a batch size that divides the rollout buffer.

        PPO warns and silently truncates when the buffer is not a multiple of the batch
        size. Rather than leave that to chance across the action repeat sweep, the batch
        size is shrunk to the largest divisor of the buffer that does not exceed the
        requested value.

        Args:
            num_envs: Number of parallel environments.

        Returns:
            The batch size to hand to PPO.
        """
        buffer = self.buffer_size(num_envs)
        if buffer % self.batch_size == 0:
            return self.batch_size
        for candidate in range(min(self.batch_size, buffer), 0, -1):
            if buffer % candidate == 0:
                return candidate
        return 1


@dataclasses.dataclass(frozen=True)
class RunMetrics:
    """Summary of one training run.

    Attributes:
        rung: Run name, which is the rung name possibly carrying an action repeat suffix.
        seed: Random seed the run used.
        action_repeat: Environment frames per agent action.
        total_timesteps: Agent steps requested.
        episodes: Episodes that finished during the run.
        successes: Episodes that reached the top landing.
        success_rate: Successes over episodes, zero when no episode finished.
        recent_success_rate: Success rate over the final hundred episodes.
        peak_backward_velocity: Most negative forward velocity seen anywhere in the run.
        mean_warps: Mean instant warp firings per episode.
        mean_return: Mean episode return.
        best_return: Highest episode return.
        max_curriculum_stage: Furthest curriculum stage reached.
        frames_to_first_success: Environment frames before the first success, or None.
        timesteps_to_first_success: Agent steps before the first success, or None.
    """

    rung: str
    seed: int
    action_repeat: int
    total_timesteps: int
    episodes: int
    successes: int
    success_rate: float
    recent_success_rate: float
    peak_backward_velocity: float
    mean_warps: float
    mean_return: float
    best_return: float
    max_curriculum_stage: int
    frames_to_first_success: int | None
    timesteps_to_first_success: int | None

    def as_dict(self) -> dict[str, Any]:
        """Returns the metrics as a JSON serializable dictionary.

        Returns:
            A plain dictionary of the fields.
        """
        return dataclasses.asdict(self)


class EpisodeRecorder(BaseCallback):
    """Streams finished episodes to a CSV and accumulates the run summary.

    The environment reports backwards velocity, warp count and curriculum stage in its info
    dictionary. Monitor adds the episode return and length under the ``episode`` key when an
    episode ends. This callback joins the two and appends a row per episode, flushing as it
    goes so that a killed job leaves a readable partial curve.

    Attributes:
        episodes: Episodes recorded so far.
    """

    def __init__(self, csv_path: pathlib.Path, action_repeat: int, flush_every: int = 1):
        """Initializes the recorder.

        Args:
            csv_path: File to append episode rows to. Its parent must already exist.
            action_repeat: Environment frames per agent step, used to convert agent steps
                into physics frames.
            flush_every: Flush the CSV after this many episodes.
        """
        super().__init__()
        self._csv_path = csv_path
        self._action_repeat = action_repeat
        self._flush_every = flush_every
        self._handle: Any = None
        self._writer: Any = None
        self.episodes: int = 0
        self._successes: int = 0
        self._returns: list[float] = []
        self._successes_by_episode: list[bool] = []
        self._warps: list[int] = []
        self._peak_backward_velocity: float = 0.0
        self._max_curriculum_stage: int = 0
        self._first_success_timesteps: int | None = None

    def _on_training_start(self) -> None:
        """Opens the CSV and writes a header when the file is new."""
        is_new = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        self._handle = self._csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(EPISODE_CSV_FIELDS))
        if is_new:
            self._writer.writeheader()
            self._handle.flush()

    def _on_step(self) -> bool:
        """Records any episodes that finished on this vectorized step.

        Returns:
            True, so training always continues.
        """
        for info in self.locals.get("infos", ()):
            if not isinstance(info, dict):
                continue
            velocity = float(info.get("peak_backward_velocity", 0.0))
            self._peak_backward_velocity = min(self._peak_backward_velocity, velocity)
            self._max_curriculum_stage = max(
                self._max_curriculum_stage, int(info.get("curriculum_stage", 0))
            )
            episode = info.get("episode")
            if episode is None:
                continue
            success = bool(info.get("success", False))
            warps = int(info.get("warps", 0))
            self.episodes += 1
            self._returns.append(float(episode["r"]))
            self._successes_by_episode.append(success)
            self._warps.append(warps)
            if success:
                self._successes += 1
                if self._first_success_timesteps is None:
                    self._first_success_timesteps = int(self.num_timesteps)
                    logging.info(
                        "first success at %d timesteps (%d frames), episode %d",
                        self._first_success_timesteps,
                        self._first_success_timesteps * self._action_repeat,
                        self.episodes,
                    )
            self._writer.writerow(
                {
                    "episode": self.episodes,
                    "timesteps": int(self.num_timesteps),
                    "frames": int(self.num_timesteps) * self._action_repeat,
                    "return": float(episode["r"]),
                    "length": int(episode["l"]),
                    "success": int(success),
                    "peak_backward_velocity": velocity,
                    "warps": warps,
                    "curriculum_stage": int(info.get("curriculum_stage", 0)),
                }
            )
            if self.episodes % self._flush_every == 0:
                self._handle.flush()
        return True

    def _on_training_end(self) -> None:
        """Flushes and closes the CSV."""
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._writer = None

    def summarize(self, rung: str, seed: int, total_timesteps: int) -> RunMetrics:
        """Builds the run summary from what was recorded.

        Args:
            rung: Run name.
            seed: Seed the run used.
            total_timesteps: Agent steps requested.

        Returns:
            The run metrics. Rates are zero and first success fields are None when no
            episode finished.
        """
        episodes = self.episodes
        recent = self._successes_by_episode[-100:]
        first = self._first_success_timesteps
        return RunMetrics(
            rung=rung,
            seed=seed,
            action_repeat=self._action_repeat,
            total_timesteps=total_timesteps,
            episodes=episodes,
            successes=self._successes,
            success_rate=self._successes / episodes if episodes else 0.0,
            recent_success_rate=sum(recent) / len(recent) if recent else 0.0,
            peak_backward_velocity=self._peak_backward_velocity,
            mean_warps=float(np.mean(self._warps)) if self._warps else 0.0,
            mean_return=float(np.mean(self._returns)) if self._returns else 0.0,
            best_return=float(np.max(self._returns)) if self._returns else 0.0,
            max_curriculum_stage=self._max_curriculum_stage,
            frames_to_first_success=first * self._action_repeat if first is not None else None,
            timesteps_to_first_success=first,
        )


def make_env_factory(
    rung: ladder.Rung,
    rom_path: str,
    collision_path: str,
    seed: int,
    rank: int,
    max_frames: int,
    spawn_jitter: float,
    library_path: str | None,
    monitor_dir: pathlib.Path | None,
) -> Callable[[], gymnasium.Env]:
    """Builds a picklable thunk that constructs one monitored environment.

    The thunk runs inside the worker process, so the libsm64 handle and the surface set are
    created there and never cross a process boundary. Each rank gets its own seed offset so
    the workers do not explore identical trajectories.

    Args:
        rung: Rung whose reward weights and action repeat the environment realizes.
        rom_path: Path to the Super Mario 64 ROM.
        collision_path: Path to the endless staircase collision data.
        seed: Base seed for the run.
        rank: Index of this environment within the vector.
        max_frames: Episode frame budget.
        spawn_jitter: Magnitude of random spawn displacement.
        library_path: Explicit libsm64 path, or None to let the environment find it.
        monitor_dir: Directory for this worker's Monitor CSV, or None to skip the file.

    Returns:
        A zero argument callable returning a Monitor wrapped environment.
    """

    def thunk() -> gymnasium.Env:
        from src.env.blj_env import BljEnv

        config = ladder.build_config(
            rung,
            rom_path=rom_path,
            collision_path=collision_path,
            max_frames=max_frames,
            spawn_jitter=spawn_jitter,
            library_path=library_path,
        )
        env = BljEnv(config)
        filename = str(monitor_dir / f"worker_{rank}") if monitor_dir is not None else None
        env = Monitor(env, filename=filename, info_keywords=())
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return thunk


def build_vec_env(
    rung: ladder.Rung,
    rom_path: str,
    collision_path: str,
    seed: int,
    num_envs: int,
    max_frames: int = 3000,
    spawn_jitter: float = 0.0,
    library_path: str | None = None,
    monitor_dir: pathlib.Path | None = None,
    force_subprocess: bool = True,
) -> VecEnv:
    """Builds the vectorized environment for one run.

    Args:
        rung: Rung to realize in every worker.
        rom_path: Path to the Super Mario 64 ROM.
        collision_path: Path to the endless staircase collision data.
        seed: Base seed for the run.
        num_envs: Number of parallel environments, one process each.
        max_frames: Episode frame budget.
        spawn_jitter: Magnitude of random spawn displacement.
        library_path: Explicit libsm64 path, or None to let the environment find it.
        monitor_dir: Directory for the Monitor CSVs, or None to skip those files.
        force_subprocess: Use ``SubprocVecEnv`` even for a single environment. Only tests
            should set this False, since a single in process libsm64 is fine there and
            avoids the spawn cost.

    Returns:
        A vectorized environment of ``num_envs`` workers.

    Raises:
        ValueError: If ``num_envs`` is not positive.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive: {num_envs}")
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    factories = [
        make_env_factory(
            rung,
            rom_path=rom_path,
            collision_path=collision_path,
            seed=seed,
            rank=rank,
            max_frames=max_frames,
            spawn_jitter=spawn_jitter,
            library_path=library_path,
            monitor_dir=monitor_dir,
        )
        for rank in range(num_envs)
    ]
    if not force_subprocess and num_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def build_model(
    vec_env: VecEnv,
    seed: int,
    hyperparameters: PpoHyperparameters = PpoHyperparameters(),
    tensorboard_log: str | None = None,
) -> PPO:
    """Constructs the PPO model for a run.

    Args:
        vec_env: Vectorized environment to train against.
        seed: Random seed for the policy initialization and action sampling.
        hyperparameters: PPO settings. Validated before use.
        tensorboard_log: Directory for tensorboard output, or None to disable.

    Returns:
        An untrained PPO model.

    Raises:
        ValueError: If the hyperparameters are out of range.
    """
    hyperparameters.validate()
    batch_size = hyperparameters.resolved_batch_size(vec_env.num_envs)
    if batch_size != hyperparameters.batch_size:
        logging.warning(
            "batch_size %d does not divide the %d step rollout buffer, using %d",
            hyperparameters.batch_size,
            hyperparameters.buffer_size(vec_env.num_envs),
            batch_size,
        )
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=hyperparameters.learning_rate,
        n_steps=hyperparameters.n_steps,
        batch_size=batch_size,
        n_epochs=hyperparameters.n_epochs,
        gamma=hyperparameters.gamma,
        gae_lambda=hyperparameters.gae_lambda,
        clip_range=hyperparameters.clip_range,
        ent_coef=hyperparameters.ent_coef,
        vf_coef=hyperparameters.vf_coef,
        max_grad_norm=hyperparameters.max_grad_norm,
        policy_kwargs={"net_arch": list(hyperparameters.net_arch)},
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=hyperparameters.device,
        verbose=0,
    )


def run_directory(out_dir: str, rung_name: str, seed: int) -> pathlib.Path:
    """Returns the directory a given run writes into.

    Args:
        out_dir: Root output directory shared by every run.
        rung_name: Run name.
        seed: Seed the run uses.

    Returns:
        The per run directory path. Not created.
    """
    return pathlib.Path(out_dir) / rung_name / f"seed_{seed}"


def train(
    rung_name: str,
    seed: int,
    total_timesteps: int,
    out_dir: str,
    rom_path: str,
    collision_path: str,
    num_envs: int = 8,
    max_frames: int = 3000,
    spawn_jitter: float = 0.0,
    library_path: str | None = None,
    hyperparameters: PpoHyperparameters = PpoHyperparameters(),
    checkpoint_every: int = 100_000,
    action_repeats: tuple[int, ...] = ladder.DEFAULT_ACTION_REPEAT_SWEEP,
    force_subprocess: bool = True,
) -> RunMetrics:
    """Trains one rung at one seed and writes its artifacts.

    Args:
        rung_name: Rung to train. Accepts a bare ladder name or an expanded sweep name such
            as ``repeat_r3``.
        seed: Random seed for the environment, the policy and the action sampling.
        total_timesteps: Agent steps to train for.
        out_dir: Root output directory. This run writes to ``out_dir/<rung>/seed_<seed>``.
        rom_path: Path to the Super Mario 64 ROM.
        collision_path: Path to the endless staircase collision data.
        num_envs: Parallel environments, one libsm64 process each.
        max_frames: Episode frame budget.
        spawn_jitter: Magnitude of random spawn displacement.
        library_path: Explicit libsm64 path, or None to let the environment find it.
        hyperparameters: PPO settings.
        checkpoint_every: Agent steps between checkpoints. Non positive disables them.
        action_repeats: Sweep values used when ``rung_name`` is the bare sweep rung, in
            which case the first value is taken. An explicit ``_r`` suffix wins over this.
        force_subprocess: Passed through to :func:`build_vec_env`.

    Returns:
        The metrics for this run, also written to ``metrics.json``.

    Raises:
        KeyError: If ``rung_name`` is not a known rung.
    """
    rung = ladder.get_rung(rung_name)
    if rung.name == ladder.SWEEP_RUNG:
        rung = ladder.expand_action_repeat(rung, (action_repeats[0],))[0]

    directory = run_directory(out_dir, rung.name, seed)
    directory.mkdir(parents=True, exist_ok=True)
    logging.info(
        "rung %s seed %d: action_repeat=%d num_envs=%d timesteps=%d out=%s",
        rung.name,
        seed,
        rung.action_repeat,
        num_envs,
        total_timesteps,
        directory,
    )

    vec_env = build_vec_env(
        rung,
        rom_path=rom_path,
        collision_path=collision_path,
        seed=seed,
        num_envs=num_envs,
        max_frames=max_frames,
        spawn_jitter=spawn_jitter,
        library_path=library_path,
        monitor_dir=directory / "monitor",
        force_subprocess=force_subprocess,
    )

    recorder = EpisodeRecorder(directory / "episodes.csv", action_repeat=rung.action_repeat)
    callbacks: list[BaseCallback] = [recorder]
    if checkpoint_every > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, checkpoint_every // max(1, num_envs)),
                save_path=str(directory / "checkpoints"),
                name_prefix="ppo",
            )
        )

    model = build_model(vec_env, seed=seed, hyperparameters=hyperparameters)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
        model.save(str(directory / "model"))
    finally:
        vec_env.close()

    metrics = recorder.summarize(rung.name, seed, total_timesteps)
    write_metrics(directory / "metrics.json", metrics, rung)
    logging.info(
        "rung %s seed %d done: %d episodes, success_rate=%.3f, peak_vel=%.2f",
        rung.name,
        seed,
        metrics.episodes,
        metrics.success_rate,
        metrics.peak_backward_velocity,
    )
    return metrics


def write_metrics(path: pathlib.Path, metrics: RunMetrics, rung: ladder.Rung) -> None:
    """Writes a run's metrics and the rung that produced them to JSON.

    Storing the rung alongside the metrics keeps each run directory self describing, which
    matters when results come back from a job array and the submit time flags are gone.

    Args:
        path: File to write.
        metrics: Metrics to record.
        rung: Rung the run realized.
    """
    payload = {"rung_config": dataclasses.asdict(rung), "metrics": metrics.as_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_metrics(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Reads a metrics JSON file back.

    Args:
        path: Path to a ``metrics.json`` written by :func:`write_metrics`.

    Returns:
        The parsed payload, with ``rung_config`` and ``metrics`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
