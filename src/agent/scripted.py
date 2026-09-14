"""The scripted backwards long jump, the expert every measurement in this project is read against.

The chain is a five stage machine driven by Mario's own action rather than by a frame count, since
every transition in it is a physics event: walk until Mario is actually walking fast enough, crouch
until the crouch slide starts, hold A until the long jump is airborne, then repress A with the
stick reversed for as long as it lasts. Each cycle lands and relaunches with forwardVel a little
further negative, and nothing in the game clamps that, which is the exploit.

Two details are what make this work rather than incidental. The A button is pressed on alternating
frames in the chain stage, because the relaunch needs a fresh press on the frame after landing and
a held A does nothing. And the stick is camera relative, so which deflection walks Mario toward the
staircase is not something this module can know: :func:`calibrate_stick` measures it by walking in
all four cardinals and watching where he ends up.
"""

from src.env.native import (
    ACT_CROUCH_SLIDE,
    ACT_LONG_JUMP,
    ACT_LONG_JUMP_LAND,
    ACT_WALKING,
    MarioInputs,
    Sm64,
)

CARDINALS = {
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}

OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def calibrate_stick(game: Sm64, surfaces: list, spawn: tuple[float, float, float],
                    settle: int = 20, walk: int = 25) -> dict[str, tuple[float, float, float]]:
    """Measures where each of the four cardinal stick deflections actually takes Mario.

    Mario's intended yaw is resolved against the camera, so the mapping from stick to world
    direction is a property of the scene's camera vector and not of this code. Measuring it is
    both shorter and more robust than deriving it: walk in each cardinal from rest and record the
    displacement.

    The settle frames matter. Mario spawns in an airborne spin and a stick held during it is not a
    walk, so the measurement would pick up the spawn animation's own drift instead.

    Args:
        game: Live libsm64 handle. This loads surfaces and creates a Mario on it, so the caller's
            scene and Mario are both gone afterwards.
        surfaces: Flat ground to walk on. A slope would add its own displacement to the answer.
        spawn: Where to spawn for each measurement.
        settle: Frames of neutral stick before the walk, to get through the spawn spin.
        walk: Frames of full deflection to measure over.

    Returns:
        One entry per cardinal, each the (dx, dz, faceAngle) Mario reached after the walk.
    """
    table = {}
    for key, (stick_x, stick_y) in CARDINALS.items():
        game.load_surfaces(surfaces)
        game.create_mario(*spawn)
        inputs = MarioInputs()
        inputs.camLookX, inputs.camLookZ = 0.0, 1.0
        for _ in range(settle):
            state = game.tick(inputs)
        origin = (state.position[0], state.position[2])
        inputs.stickX, inputs.stickY = stick_x, stick_y
        for _ in range(walk):
            state = game.tick(inputs)
        table[key] = (state.position[0] - origin[0], state.position[2] - origin[1],
                      state.faceAngle)
    return table


def stick_toward(table: dict[str, tuple[float, float, float]],
                 target_x: float, target_z: float) -> str:
    """Picks the calibrated cardinal that moves Mario most nearly toward a direction.

    Args:
        table: Calibration from :func:`calibrate_stick`.
        target_x: x of the desired direction. It need not be normalized.
        target_z: z of the desired direction.

    Returns:
        The cardinal whose measured displacement has the largest dot product with the target.
    """
    return max(table, key=lambda key: table[key][0] * target_x + table[key][1] * target_z)


class ScriptedBlj:
    """The stage machine, as a policy that only needs Mario's action and speed to advance.

    Stages run settle, walk, crouch, launch, chain, and never go back. The interface is split into
    :meth:`inputs`, which says what to hold now, and :meth:`observe`, which advances the stage from
    what Mario did, so the same policy drives the raw libsm64 handle and the Gymnasium environment
    without either one knowing about the other.

    Attributes:
        approach: Cardinal walked before the reversal, as named by :func:`calibrate_stick`.
        walk_magnitude: Stick magnitude held while walking and crouching.
        air_magnitude: Stick magnitude held during the chain. Values above 1 are outside the
            normalized range libsm64 expects and put the run in a different physics regime.
        walk_speed: Absolute forwardVel that counts as walking fast enough to crouch.
        settle_frames: Frames of neutral stick before the walk starts.
        repress_period: Frames between A presses during the chain. The relaunch needs a new press
            rather than a held button, so 2 is the tight alternation the chain is tuned for.
        stage: Current stage name.
        frame: Frames observed so far, which is also what the A press parity is taken from.
    """

    def __init__(self, approach: str, walk_magnitude: float = 1.0,
                 air_magnitude: float = 1.0, walk_speed: float = 20.0,
                 settle_frames: int = 20, repress_period: int = 2):
        """Sets up the machine in its settle stage.

        Args:
            approach: Cardinal to walk toward before reversing.
            walk_magnitude: Stick magnitude for the walk and crouch stages.
            air_magnitude: Stick magnitude for the chain stage.
            walk_speed: Absolute forwardVel required before crouching.
            settle_frames: Frames to wait out the spawn spin.
            repress_period: Frames between A presses in the chain.
        """
        self.approach = approach
        self.walk_magnitude = walk_magnitude
        self.air_magnitude = air_magnitude
        self.walk_speed = walk_speed
        self.settle_frames = settle_frames
        self.repress_period = repress_period
        self.stage = "settle"
        self.frame = 0
        self._inputs = MarioInputs()
        self._inputs.camLookX, self._inputs.camLookZ = 0.0, 1.0

    def inputs(self) -> MarioInputs:
        """Returns the controller state to hold for the current frame.

        The stick reverses at the chain stage, from the approach cardinal to its opposite, which
        is the whole trick: Mario keeps long jumping while his stick asks for the direction he
        came from, so each relaunch takes forwardVel further negative.

        Returns:
            The single MarioInputs this policy owns, rewritten in place every call. Callers that
            record what was pressed have to copy the fields out.
        """
        walk_x, walk_y = CARDINALS[self.approach]
        air_x, air_y = CARDINALS[OPPOSITE[self.approach]]
        held = self._inputs
        held.stickX = held.stickY = 0.0
        held.buttonA = held.buttonB = held.buttonZ = 0

        if self.stage == "settle" and self.frame >= self.settle_frames:
            self.stage = "walk"
        if self.stage in ("walk", "crouch", "launch"):
            held.stickX = walk_x * self.walk_magnitude
            held.stickY = walk_y * self.walk_magnitude
        if self.stage in ("crouch", "launch", "chain"):
            held.buttonZ = 1
        if self.stage == "launch":
            held.buttonA = 1
        if self.stage == "chain":
            held.stickX = air_x * self.air_magnitude
            held.stickY = air_y * self.air_magnitude
            held.buttonA = 1 if self.frame % self.repress_period == 0 else 0
        return held

    def observe(self, action: int, forward_velocity: float) -> None:
        """Advances the stage from what Mario actually did on the last frame.

        Each transition waits for the action it needs rather than for a frame budget, because the
        timing depends on the geometry underfoot: the walk takes longer uphill, and a crouch slide
        that never starts must not be followed by a long jump attempt.

        Args:
            action: Mario's action bitfield after the last tick.
            forward_velocity: Mario's forwardVel after the last tick.
        """
        if (self.stage == "walk" and action == ACT_WALKING
                and abs(forward_velocity) >= self.walk_speed):
            self.stage = "crouch"
        elif self.stage == "crouch" and action == ACT_CROUCH_SLIDE:
            self.stage = "launch"
        elif self.stage == "launch" and action == ACT_LONG_JUMP:
            self.stage = "chain"
        self.frame += 1


def run_chain(game: Sm64, surfaces: list, spawn: tuple[float, float, float],
              policy: ScriptedBlj, frames: int) -> dict:
    """Runs one policy on one scene and reports what the chain did.

    This is the loop the geometry sweeps use, and it deliberately bypasses the Gymnasium
    environment: there is no reward, no episode limit and no instant warp, so what it measures is
    the chain against the geometry alone.

    Args:
        game: Live libsm64 handle. The scene and Mario are both replaced.
        surfaces: The scene to run on.
        spawn: Where to spawn Mario.
        policy: The policy to drive with, advanced in place.
        frames: Frames to run. The loop never stops early, so a stalled chain still returns.

    Returns:
        A dict of the per launch cycles, the peak (most negative) forwardVel, the frame the chain
        first left the long jump loop if it did, and the full per frame trace.
    """
    game.load_surfaces(surfaces)
    game.create_mario(*spawn)

    cycles = []
    air_frames = 0
    previous = 0
    peak = 0.0
    left_loop = None
    trace = []

    for frame in range(frames):
        inputs = policy.inputs()
        pressed = {"stick_x": round(inputs.stickX, 3), "stick_y": round(inputs.stickY, 3),
                   "a": int(inputs.buttonA), "b": int(inputs.buttonB), "z": int(inputs.buttonZ)}
        state = game.tick(inputs)
        policy.observe(state.action, state.forwardVelocity)

        if state.action == ACT_LONG_JUMP and previous != ACT_LONG_JUMP:
            cycles.append({"frame": frame,
                           "launch_velocity": round(state.forwardVelocity, 3),
                           "air_frames": air_frames})
            air_frames = 0
        if state.action == ACT_LONG_JUMP:
            air_frames += 1
        if state.forwardVelocity < peak:
            peak = state.forwardVelocity
        if (policy.stage == "chain" and left_loop is None
                and state.action not in (ACT_LONG_JUMP, ACT_LONG_JUMP_LAND)):
            left_loop = {"frame": frame, "action": state.action,
                         "forward_velocity": round(state.forwardVelocity, 3)}
        trace.append({"frame": frame, "action": state.action,
                      "position": [round(v, 2) for v in state.position],
                      "velocity": [round(v, 2) for v in state.velocity],
                      "forward_velocity": round(state.forwardVelocity, 3),
                      "face_angle": round(state.faceAngle, 4),
                      "health": state.health, "stage": policy.stage, "inputs": pressed})
        previous = state.action

    return {"cycles": cycles, "peak_velocity": round(peak, 3), "left_loop": left_loop,
            "trace": trace}
