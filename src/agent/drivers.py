"""Adapters that turn a policy into a source of environment actions.

The environment exposes a 36 way discrete action space, while the scripted expert thinks in stick
deflections and button holds and a trained policy thinks in action indices. Keeping both adapters
here means every script drives the environment through the same loop, which matters more than it
looks: the backwards long jump is sensitive to the parity of the A press, so a one frame shift in
where the policy observes changes the outcome.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.agent.scripted import ScriptedBlj
from src.env.blj_env import STICK_DIRECTIONS


class Driver(Protocol):
    """Chooses an action from the latest observation and info dict."""

    def __call__(self, observation: Any, info: dict[str, Any]) -> int:
        """Returns the next action index.

        Args:
            observation: The environment's latest observation.
            info: The environment's latest info dict.

        Returns:
            An action index in ``[0, ACTION_SIZE)``.
        """
        ...


def action_index(stick_x: float, stick_y: float, press_a: bool, press_z: bool) -> int:
    """Maps a controller state onto the nearest environment action.

    Args:
        stick_x: Stick x in [-1, 1].
        stick_y: Stick y in [-1, 1].
        press_a: Whether A is held.
        press_z: Whether Z is held.

    Returns:
        The index of the closest available action.
    """
    nearest = min(
        range(len(STICK_DIRECTIONS)),
        key=lambda i: (STICK_DIRECTIONS[i][0] - stick_x) ** 2
        + (STICK_DIRECTIONS[i][1] - stick_y) ** 2)
    buttons = int(bool(press_a)) + 2 * int(bool(press_z))
    return buttons * len(STICK_DIRECTIONS) + nearest


def scripted_driver(approach: str, air_magnitude: float = 1.0) -> Driver:
    """Wraps the scripted expert as a driver.

    Args:
        approach: Cardinal the expert walks toward before reversing.
        air_magnitude: Stick magnitude held during the air phase.

    Returns:
        A driver closed over the expert's stage machine.
    """
    policy = ScriptedBlj(approach, air_magnitude=air_magnitude)

    def drive(observation: Any, info: dict[str, Any]) -> int:
        del observation
        policy.observe(info["mario_action_id"], info["forward_velocity"])
        held = policy.inputs()
        return action_index(held.stickX, held.stickY, bool(held.buttonA), bool(held.buttonZ))

    return drive


def model_driver(model_path: str, deterministic: bool = True, seed: int | None = None) -> Driver:
    """Loads a trained policy and wraps it as a driver.

    Args:
        model_path: Path to a stable-baselines3 model zip.
        deterministic: Take the argmax action when true, sample from the policy when false.
            Sampling is often the stronger choice here: the chain needs A on alternating frames,
            and an argmax policy that has learned to hold one action collapses the alternation.
        seed: Seeds the policy's sampling. Loading a model resets its generator, so without this
            every sampled rollout of one checkpoint is the same trajectory.

    Returns:
        A driver that queries the policy.

    Raises:
        ImportError: If stable-baselines3 is not installed.
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    if seed is not None:
        model.set_random_seed(seed)

    def drive(observation: Any, info: dict[str, Any]) -> int:
        del info
        action, _ = model.predict(observation, deterministic=deterministic)
        return int(action)

    return drive
