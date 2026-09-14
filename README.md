# mario-blj

How much hand-holding does an RL agent need before it can perform the backwards long jump?

No results yet. This is the environment scaffold.

## The question

Super Mario 64's endless staircase cannot be climbed with fewer than 70 stars. It loops. The only
way up is the backwards long jump, a speed-accumulation glitch that took the speedrunning community
years to find.

That makes it a rare task: success is self-certifying. Reaching the top under 70 stars *is* the
exploit, so there is no judgement call about whether the agent "cheated".

The interesting result is not that an agent can be shaped into doing a BLJ. It is the measurement of
how much shaping that takes. The plan is an ablation ladder, from terminal reward only through
speed shaping, staged curriculum and tuned action repeat, reporting where the agent crosses from
never to reliably. That turns "RL could never discover this" from an opinion into a number.

The validation is slope transfer. BLJ works on many slopes, not one staircase. Train on one, test on
others. If it transfers, the policy learned the technique. If not, it memorised a spot.

## Why libsm64

`sm64_mario_tick` steps one frame of Mario's real decompiled physics:

```c
struct SM64MarioInputs  { float camLookX, camLookZ, stickX, stickY;
                          uint8_t buttonA, buttonB, buttonZ; };
struct SM64MarioState   { float position[3], velocity[3], faceAngle, forwardVelocity;
                          int16_t health; uint32_t action; ... };
```

`forwardVelocity` is the shaping signal, `action` is the curriculum stage detector, and
`sm64_static_surfaces_load` takes an arbitrary triangle mesh, so slopes can be generated
procedurally. Resets are a function call rather than a savestate. No emulator, no rendering, no
window.

The mechanism that makes the BLJ possible is present, carrying the decompilation's own bug marker
in `mario_actions_airborne.c`:

```c
//! Uncapped air speed. Net positive when moving forward.
if (m->forwardVel > dragThreshold) { m->forwardVel -= 1.0f; }
if (m->forwardVel < -16.0f)        { m->forwardVel += 2.0f; }
```

libsm64 carries Mario and surfaces but no level logic, so the endless staircase's loop trigger is
not in it. The plan is to study and train the mechanic here, then demonstrate the real staircase on
a full `sm64-port` build.

## Verified so far

- libsm64 builds clean as a universal binary on macOS
- every ctypes struct layout matches the C header, checked field by field against `offsetof`
- generated floor normals all point up; ramp heights match `length * tan(angle)` exactly

## Not yet verified

Whether the BLJ actually reproduces in libsm64. `scripts/verify_blj.py` sweeps slope angle, jump
re-press delay and stick direction, looking for a scripted input chain where `|forwardVelocity|`
grows across cycles. That is the go/no-go, and it needs a ROM.

One thing to watch: long jump entry is gated on `m->forwardVel > 10.0f` in `act_crouch_slide`, so
whether the chain sustains once velocity goes negative is exactly the empirical question.

## Setup

    ./scripts/setup.sh
    # place a Super Mario 64 US ROM at roms/baserom.us.z64
    PYTHONPATH=. python3 scripts/verify_blj.py

libsm64 reads Mario's animation and texture data from the ROM at runtime, so one is required even
though the library itself builds without it. No ROM is distributed here.
