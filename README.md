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

- libsm64 builds clean as a universal binary
- every ctypes struct layout matches the C header, checked field by field against `offsetof`
- ROM converted from v64 to z64 and hash-verified (sha1 `9bef1128...`, the canonical US release)
- Mario walks, runs, crouch-slides and long jumps under scripted input
- generated floor normals point up; ramp heights match `length * tan(angle)` exactly

## The mechanism, located in source

Three annotated sites in the decompilation carry the whole bug.

`mario.c`, the amplifier. The clamp is one-sided, so negative speed multiplies without bound:

```c
//! (BLJ's) This properly handles long jumps from getting forward speed with
//  too much velocity, but misses backwards longs allowing high negative speeds.
if ((m->forwardVel *= 1.5f) > 48.0f) { m->forwardVel = 48.0f; }
```

`mario_actions_airborne.c`, the asymmetric air drag:

```c
//! Uncapped air speed. Net positive when moving forward.
if (m->forwardVel > dragThreshold) { m->forwardVel -= 1.0f; }
if (m->forwardVel < -16.0f)        { m->forwardVel += 2.0f; }
```

`mario_actions_moving.c`, Nintendo's own fix, compiled out under `VERSION_US`. Rebuilding with
`-DVERSION_SH` makes the BLJ impossible, which is a ground-truth negative control shipped by the
original developers rather than one invented here:

```c
#ifdef VERSION_SH
    // BLJ (Backwards Long Jump) speed build up fix, crushing SimpleFlips's dreams since July 1997
    if (m->forwardVel < 0.0f) { m->forwardVel = 0.0f; }
#endif
```

## Measured dynamics

Gain per re-jump is `0.5 * |forwardVel|` from the `*= 1.5`. Loss is air decay toward an attractor at
exactly -16: below it the `+= 2.0` damping dominates, above it the stick does. Measured air decay
ranges from +1.70 to +5.70 per frame depending on stick direction relative to Mario's facing, so the
chain grows only when

    0.5 * |forwardVel|  >  decay_rate * air_frames

## Direction calibration, measured not assumed

| facing | forwardVel | travels |
| --- | ---: | --- |
| 0 | +20 | +Z |
| 0 | -20 | -Z |
| pi | +20 | -Z |
| pi | -20 | +Z |

Air decay per frame, seeded at `forwardVel = -20`, by stick relative to facing: aligned +5.70,
perpendicular +4.70, opposed +1.70. Decay is minimised when the stick points along the direction of
travel, and travel must point into rising ground for a short air time. On a uniform ramp those two
requirements fix each other, which is the whole difficulty.

## Real level geometry

`src/env/collision.py` imports `collision.inc.c` from the decompilation into libsm64 surfaces.
Castle area 3 parses to 1399 triangles with the expected surface types, and contains ~30 degree
slopes (`normal.y = 0.869`), confirming that synthetic ramps already match real geometry. Geometry
was therefore not the blocker.

## Not yet reproduced

A runaway BLJ. On every synthetic geometry tried, the stick direction that minimises air decay
(+1.70/frame) also sends Mario downhill, giving a 31-frame air time, while the direction giving a
5-frame air time maximises decay (+5.70/frame). The two requirements are coupled through the ramp,
and neither setting satisfies the growth condition. Flat ground, treads-only staircases, slippery
slopes and walls were all tried; slippery surfaces make Mario slide instead of holding the chain.

The next step is to stop searching blind and replay a documented BLJ input sequence. If a known-good
setup fails here, libsm64 is the wrong substrate, since it carries Mario and surfaces but no level
geometry or object behaviour, and the work should move to a full `sm64-port` build.

## Setup

    ./scripts/setup.sh
    # place a Super Mario 64 US ROM at roms/baserom.us.z64 (v64 input is converted automatically)
    PYTHONPATH=. python3 scripts/verify_blj.py

No ROM is distributed here.
