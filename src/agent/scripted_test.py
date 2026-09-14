"""Tests for the hand-written backwards long jump policy in :mod:`src.agent.scripted`.

The policy is a five stage machine driven entirely by what it observes coming back from the
engine, so it can be exercised without the native library: feed it a scripted sequence of
(action, forwardVelocity) pairs and read the controller state it asks for on each frame. One
test at the bottom does drive the real engine, and it is skipped unless both the built dylib and
a ROM are present.
"""

import itertools
import os

import pytest

from src.agent.scripted import CARDINALS, OPPOSITE, ScriptedBlj, stick_toward
from src.env.native import (
    ACT_BRAKING,
    ACT_CROUCH_SLIDE,
    ACT_IDLE,
    ACT_LONG_JUMP,
    ACT_WALKING,
    MarioInputs,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROM_PATH = os.path.join(REPO_ROOT, "roms", "baserom.us.z64")
DYLIB_PATH = os.path.join(REPO_ROOT, "third_party", "libsm64", "dist", "libsm64.dylib")
SO_PATH = os.path.join(REPO_ROOT, "third_party", "libsm64", "dist", "libsm64.so")

LONG_AIR_PHASE_CAP = 24.0

needs_engine = pytest.mark.skipif(
    not (os.path.exists(ROM_PATH) and (os.path.exists(DYLIB_PATH) or os.path.exists(SO_PATH))),
    reason="needs roms/baserom.us.z64 and a built libsm64; run scripts/setup.sh")

Frame = tuple[str, float, float, int, int, int]


def _snapshot(stage: str, inputs: MarioInputs) -> Frame:
    """Flattens one frame of controller state into a comparable tuple.

    Args:
        stage: The policy stage the inputs were built for.
        inputs: The controller state the policy returned.

    Returns:
        A (stage, stick_x, stick_y, button_a, button_b, button_z) tuple.
    """
    return (stage, round(float(inputs.stickX), 6), round(float(inputs.stickY), 6),
            int(inputs.buttonA), int(inputs.buttonB), int(inputs.buttonZ))


def _drive(policy: ScriptedBlj, observations: list[tuple[int, float]]) -> list[Frame]:
    """Runs the policy against a scripted sequence of engine observations.

    Args:
        policy: The policy under test.
        observations: One (action, forward_velocity) pair per frame, in order.

    Returns:
        One snapshot per frame, recording the stage and the controller state it produced.
    """
    frames = []
    for action, velocity in observations:
        inputs = policy.inputs()
        frames.append(_snapshot(policy.stage, inputs))
        policy.observe(action, velocity)
    return frames


def _blj_sequence(settle_frames: int, chain_frames: int) -> list[tuple[int, float]]:
    """Builds the observation sequence that walks the policy through every stage.

    Args:
        settle_frames: How many idle frames the policy is configured to settle for.
        chain_frames: How many frames to stay in the chain stage afterwards.

    Returns:
        The observation sequence.
    """
    return ([(ACT_IDLE, 0.0)] * settle_frames
            + [(ACT_WALKING, -21.0), (ACT_CROUCH_SLIDE, -21.0), (ACT_LONG_JUMP, -31.5)]
            + [(ACT_LONG_JUMP, -47.0)] * chain_frames)


def test_cardinals_are_unit_vectors_and_opposite_pairs_negate():
    """The four cardinals are axis-aligned unit sticks and OPPOSITE flips each one."""
    assert set(CARDINALS) == set(OPPOSITE)
    for key, (stick_x, stick_y) in CARDINALS.items():
        assert stick_x * stick_x + stick_y * stick_y == 1.0
        other_x, other_y = CARDINALS[OPPOSITE[key]]
        assert (other_x, other_y) == (-stick_x, -stick_y)
        assert OPPOSITE[OPPOSITE[key]] == key


def test_stage_machine_advances_settle_walk_crouch_launch_chain():
    """Each stage waits for its own engine action before handing over to the next."""
    policy = ScriptedBlj(approach="up", settle_frames=3)
    frames = _drive(policy, _blj_sequence(settle_frames=3, chain_frames=3))

    assert [frame[0] for frame in frames] == [
        "settle", "settle", "settle", "walk", "crouch", "launch", "chain", "chain", "chain"]
    assert policy.stage == "chain"


def test_settle_stage_sends_a_neutral_controller():
    """Nothing is pressed while Mario settles onto the floor."""
    policy = ScriptedBlj(approach="up", settle_frames=4)
    frames = _drive(policy, [(ACT_IDLE, 0.0)] * 4)

    for stage, stick_x, stick_y, button_a, button_b, button_z in frames:
        assert stage == "settle"
        assert (stick_x, stick_y) == (0.0, 0.0)
        assert (button_a, button_b, button_z) == (0, 0, 0)


def test_walk_stage_pushes_the_stick_toward_the_approach_and_holds_no_buttons():
    """Walking is stick-only; Z arrives one stage later."""
    policy = ScriptedBlj(approach="left", settle_frames=1, walk_magnitude=0.75)
    frames = _drive(policy, [(ACT_IDLE, 0.0), (ACT_IDLE, 0.0)])

    stage, stick_x, stick_y, button_a, button_b, button_z = frames[1]
    assert stage == "walk"
    assert (stick_x, stick_y) == (-0.75, 0.0)
    assert (button_a, button_b, button_z) == (0, 0, 0)


def test_crouch_stage_adds_z_while_keeping_the_approach_stick():
    """Z goes down before A, which is what turns the walk into a crouch slide."""
    policy = ScriptedBlj(approach="up", settle_frames=1)
    frames = _drive(policy, [(ACT_IDLE, 0.0), (ACT_WALKING, -25.0), (ACT_IDLE, 0.0)])

    assert frames[2] == ("crouch", 0.0, 1.0, 0, 0, 1)


def test_launch_stage_holds_z_and_a_together():
    """A is added on top of Z to convert the crouch slide into a long jump."""
    policy = ScriptedBlj(approach="up", settle_frames=1)
    frames = _drive(policy, [(ACT_IDLE, 0.0), (ACT_WALKING, -25.0), (ACT_CROUCH_SLIDE, -25.0),
                             (ACT_IDLE, 0.0)])

    assert frames[3] == ("launch", 0.0, 1.0, 1, 0, 1)


def test_chain_stage_reverses_the_stick_to_the_opposite_cardinal():
    """Once airborne the stick flips to oppose the approach, which is what drives Mario back."""
    policy = ScriptedBlj(approach="up", settle_frames=1, air_magnitude=0.5)
    frames = _drive(policy, _blj_sequence(settle_frames=1, chain_frames=2))

    approach_x, approach_y = CARDINALS["up"]
    for stage, stick_x, stick_y, _, _, button_z in frames[4:]:
        assert stage == "chain"
        assert (stick_x, stick_y) == (-approach_x * 0.5, -approach_y * 0.5)
        assert button_z == 1


def test_chain_stage_toggles_a_instead_of_holding_it():
    """A is released between presses.

    INPUT_A_PRESSED is latched from a rising edge, so a held A never re-triggers the long jump
    and the chain dies after one cycle. The policy must put a zero between every pair of ones.
    """
    policy = ScriptedBlj(approach="up", settle_frames=1, repress_period=2)
    frames = _drive(policy, _blj_sequence(settle_frames=1, chain_frames=7))

    chain_a = [frame[3] for frame in frames if frame[0] == "chain"]
    assert set(chain_a) == {0, 1}
    assert chain_a == [1, 0, 1, 0, 1, 0, 1]
    assert not any(a == 1 and b == 1 for a, b in itertools.pairwise(chain_a))


def test_chain_repress_period_sets_the_gap_between_a_presses():
    """A longer repress period leaves proportionally more released frames between presses.

    The press phase comes from the policy's absolute frame counter rather than from the frame
    the chain started on, so the first chain frame is not guaranteed to be a press. What is
    guaranteed is one press every repress_period frames, released in between.
    """
    policy = ScriptedBlj(approach="up", settle_frames=1, repress_period=3)
    frames = _drive(policy, _blj_sequence(settle_frames=1, chain_frames=9))

    chain_a = [frame[3] for frame in frames if frame[0] == "chain"]
    assert chain_a == [0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert sum(chain_a) == len(chain_a) // 3
    assert not any(a == 1 and b == 1 for a, b in itertools.pairwise(chain_a))


def test_walk_stage_waits_for_enough_speed():
    """A walking Mario below walk_speed is not ready to crouch slide."""
    policy = ScriptedBlj(approach="up", settle_frames=1, walk_speed=20.0)
    _drive(policy, [(ACT_IDLE, 0.0)] + [(ACT_WALKING, -5.0)] * 5)

    assert policy.stage == "walk"

    _drive(policy, [(ACT_WALKING, -20.0)])
    assert policy.stage == "crouch"


def test_walk_stage_ignores_actions_other_than_walking():
    """Braking at speed does not satisfy the walk stage, even though the speed is there."""
    policy = ScriptedBlj(approach="up", settle_frames=1)
    _drive(policy, [(ACT_IDLE, 0.0)] + [(ACT_BRAKING, -48.0)] * 5)

    assert policy.stage == "walk"


def test_stages_cannot_be_skipped():
    """Seeing ACT_LONG_JUMP while still crouching does not jump the policy to the chain."""
    policy = ScriptedBlj(approach="up", settle_frames=1)
    _drive(policy, [(ACT_IDLE, 0.0), (ACT_WALKING, -25.0)])
    assert policy.stage == "crouch"

    _drive(policy, [(ACT_LONG_JUMP, -31.5)] * 4)
    assert policy.stage == "crouch"


def test_walk_speed_threshold_uses_absolute_velocity():
    """The threshold is on magnitude, so an approach in +z counts the same as one in -z."""
    policy = ScriptedBlj(approach="down", settle_frames=1, walk_speed=20.0)
    _drive(policy, [(ACT_IDLE, 0.0), (ACT_WALKING, 24.0)])

    assert policy.stage == "crouch"


def test_stick_toward_picks_the_cardinal_that_moved_mario_the_right_way():
    """stick_toward scores each calibrated cardinal against the target direction."""
    table = {
        "up": (0.0, 400.0, 0.0),
        "down": (0.0, -400.0, 3.14159),
        "left": (-400.0, 0.0, 1.5708),
        "right": (400.0, 0.0, -1.5708),
    }

    assert stick_toward(table, 0.0, 1.0) == "up"
    assert stick_toward(table, 0.0, -1.0) == "down"
    assert stick_toward(table, 1.0, 0.0) == "right"
    assert stick_toward(table, -1.0, 0.0) == "left"
    assert stick_toward(table, 0.4, -1.0) == "down"
    assert stick_toward(table, -1.0, 0.4) == "left"


def test_stick_toward_follows_a_rotated_camera_basis():
    """When the camera basis flips z, stick up moves Mario toward -z and the table says so.

    This is the reason the policy calibrates at all instead of assuming stick up is +z.
    """
    flipped = {
        "up": (0.0, -398.0, -3.14159),
        "down": (0.0, 398.0, 0.0),
        "left": (398.0, 0.0, 1.5708),
        "right": (-398.0, 0.0, -1.5708),
    }

    assert stick_toward(flipped, 0.0, -1.0) == "up"
    assert stick_toward(flipped, 0.0, 1.0) == "down"
    assert stick_toward(flipped, 1.0, 0.0) == "left"
    assert stick_toward(flipped, -1.0, 0.0) == "right"


def test_stick_toward_always_returns_a_known_cardinal():
    """The result indexes back into CARDINALS, whatever the target direction is."""
    table = {key: (stick_x * 400.0, stick_y * 400.0, 0.0)
             for key, (stick_x, stick_y) in CARDINALS.items()}

    for target_x, target_z in [(1.0, 1.0), (-3.0, 0.1), (0.0, 0.0), (0.2, -0.9)]:
        assert stick_toward(table, target_x, target_z) in CARDINALS


@needs_engine
def test_run_chain_on_flat_ground_stays_under_the_air_phase_cap():
    """Driving the real engine over flat ground reaches the chain stage but cannot bootstrap.

    A flat floor gives an air phase of about 30 frames, which is long enough for the -16
    attractor in update_air_without_turn to pin the landing speed near -15 and cap the next
    launch at 1.5 * 16. The peak backwards speed therefore stays inside 24 no matter how many
    cycles run. Staircases are the geometry that escapes this.
    """
    from src.agent.scripted import calibrate_stick, run_chain
    from src.env import geometry
    from src.env.native import Sm64

    game = Sm64(ROM_PATH)
    try:
        surfaces = (geometry.flat_area(-400.0, 400.0, 400.0, 1600.0, 0.0)
                    + geometry.staircase(steps=40, rise=25.6, run=51.2, width=800.0,
                                         base_height=0.0, origin_z=-2048.0))
        spawn = (0.0, 60.0, 1000.0)
        table = calibrate_stick(game, surfaces, spawn)
        approach = stick_toward(table, 0.0, -1.0)

        policy = ScriptedBlj(approach=approach, settle_frames=20)
        result = run_chain(game, surfaces, spawn, policy, frames=300)
    finally:
        game.close()

    assert sorted(result) == ["cycles", "left_loop", "peak_velocity", "trace"]
    assert len(result["trace"]) == 300
    assert policy.stage == "chain"
    assert len(result["cycles"]) >= 2

    assert -LONG_AIR_PHASE_CAP < result["peak_velocity"] < -1.0
    chained = result["cycles"][1:]
    assert all(cycle["air_frames"] > 10 for cycle in chained)
    assert all(abs(cycle["launch_velocity"]) < LONG_AIR_PHASE_CAP for cycle in chained)
