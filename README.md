# mario-blj

How much hand-holding does an RL agent need before it can perform the backwards long jump?

## The question

Super Mario 64's endless staircase cannot be climbed with fewer than 70 stars. It loops. The only
way up is the backwards long jump, a speed accumulation glitch that took the speedrunning community
years to find.

That makes it a rare task: success is self-certifying. Reaching the top of the staircase *is* the
exploit, so there is no judgement call about whether the agent cheated.

The interesting result is not that an agent can be shaped into doing a BLJ. It is the measurement
of how much shaping that takes. The plan is an ablation ladder, from terminal reward only through
speed shaping, staged curriculum and tuned action repeat, reporting where the agent crosses from
never to reliably. That turns "RL could never discover this" from an opinion into a number.

The validation is transfer: the same policy should work on geometry it never trained on. It does
not, and that measurement is below.

## Where this stands

An agent found the backwards long jump with no reward shaping at all.

| | result |
| --- | --- |
| substrate | libsm64, the decompiled physics as a shared library |
| fidelity | the BLJ chain is byte identical to upstream `n64decomp/sm64` @`9921382a`, and a real TAS chain replays through it 45 frames bit-identically |
| task | the real endless staircase, 1923 collision triangles from `castle_inside` area 2 |
| expert result | reaches the top landing, peak `forwardVel` **-1543.88**, 37 warp resets survived |
| throughput | 7,700 environment steps per second in one process |
| PPO result | terminal reward alone reaches a **100% success rate**, first success at **6.3M steps** |

The headline is the last row. The rung that pays nothing except for standing on the top landing,
whose episode returns take only the values 0.0 and 1.0, learned the exploit. So the answer to how
much shaping an agent needs is, for at least one seed, none. "RL could never discover this" is
false in this environment, and the interesting quantity becomes how many steps and how often.

Results are the finished ladder: 4 rungs x 6 seeds x 20M steps, all 24 runs to the same budget,
reduced into `results/learning_curves.json`.

| rung | seeds with successes | fastest discovery | best peak `forwardVel` | notes |
| --- | --- | ---: | ---: | --- |
| terminal | 1 of 6 | 6,324,840 | -5588.1 | seed 2 at 92,801 successes in 95,093 episodes |
| speed | **6 of 6** | 580,644 | -6110.0 | all six, from 0.58M to 17.2M steps |
| height | **0 of 6** | never | -146.2 | no seed ever chains; every run ends 6,660 episodes long |
| height_speed | 3 of 6 | 699,852 | -6418.4 | seeds 0, 1 and 4 only |

The sharpest signal is the last three rows against each other. Height progress alone never
crosses, in six of six seeds. Backwards speed alone crosses in all six. Adding height on top of
speed takes six of six back down to three of six, so the dead term does not merely fail to help:
it costs the term that works. Height reliably walks Mario to the barrier and stops; speed is what
carries him through it.

At n = 6 those are different grades of evidence. Speed against height, 6 of 6 against 0 of 6, is
Fisher p ~ 0.002. Height against terminal, 0 of 6 against 1 of 6, is p = 1.0, so the headline
negative result carries no support *as a rate comparison* and the argument for it below is
mechanistic instead. The interaction, 6 of 6 against 3 of 6, is p ~ 0.18 and is suggestive only.

The height rung failing is the designed outcome rather than a disappointment, and it is worth
saying why. Its reward states the goal and nothing else, and it goes flat exactly where ordinary
movement stops working, so there is no gradient across the discontinuity that the exploit lives
on. A policy that just holds the stick up the stairs earns 0.419 of it, reaches y 3942, and is
then thrown back down 302 times for nothing.

### What the policies learned was the staircase

Every run trained on one flight of stairs, so a success is ambiguous: it could be the backwards long
jump, which is a property of Mario's physics, or a sequence of inputs that suits 25.6 unit treads
51.25 units deep, which is a property of one level. `scripts/transfer_test.py` settles it by dropping
all 24 final policies, unchanged, on ten flights: the castle's own, a synthetic rebuild of its tread
geometry, and eight variations. 8 episodes each, 1,920 episodes, `results/transfer.json`. The rate
column is the mean success rate over the ten policies that solve the castle at all.

| flight | rise | run | faces between treads | policies | rate | best peak `forwardVel` |
| --- | ---: | ---: | :---: | ---: | ---: | ---: |
| the castle | 25.6 | 51.25 | yes | **10 of 24** | 1.000 | -655.9 |
| rebuilt from rise and run | 25.6 | 51.25 | yes | **9 of 24** | 0.775 | -834.6 |
| rebuilt, faces removed | 25.6 | 51.25 | no | **9 of 24** | 0.825 | -913.5 |
| rise 50, faces removed | 50 | 51.25 | no | **9 of 24** | 0.325 | -633.4 |
| rise 75, faces removed | 75 | 100 | no | **5 of 24** | 0.212 | -649.0 |
| rise 26, run 100 | 25.6 | 100 | yes | 0 of 24 | 0.000 | -52.8 |
| rise 50 | 50 | 51.25 | yes | 0 of 24 | 0.000 | -40.8 |
| rise 75 | 75 | 100 | yes | 0 of 24 | 0.000 | -40.9 |
| rise 100 | 100 | 100 | yes | 0 of 24 | 0.000 | -40.5 |
| rise 100, faces removed | 100 | 100 | no | 0 of 24 | 0.000 | -595.6 |

The rebuild is the control and it passes: 284 triangles standing in for 1923, and nine of the ten
keep the exploit. So the failures below it are facts about geometry rather than artefacts.

What they say is that the height of the vertical face between treads decides everything. At the
castle's own tread depth, raising the rise from 25.6 to 50 takes 10 of 24 to 0 of 24 and the peak
from -656 to -41, which is about two launches and no compounding. Delete those 50 unit faces and
change nothing else, and 9 of 24 come back at -633. Delete the castle's own 25.6 unit faces and
nothing happens either way. The chain does not need the risers and cannot survive tall ones. Tread
depth matters too and independently: keep the castle's faces, double the run, and all 24 collapse.

One more negative worth keeping: on `rise 100, faces removed` the chain does run away, to -595.6
against a warp band 300 units deep, and still no policy in 192 episodes climbed past the band.
Escaping means landing beyond it, not being fast below it.

`scripts/transfer_expert.py` runs the project's hand written expert on the same ten flights for a
control on the geometry itself. It runs away on exactly the two flights with no faces and a steep
rise, -454 and -579, which proves those admit a chain. It reaches only the -16 air drag attractor
everywhere else, including on the castle, where ten learned policies reach -656. So its successes
are informative and its failures are not, and the fixed two frame repress schedule that works on
abstract staircases does not reproduce what PPO found on the real one.

## The mechanism

Three annotated sites in the decompilation carry the whole bug.

`mario.c`, in `set_mario_action_airborne`, is the amplifier. The clamp is one sided, so negative
speed multiplies without bound:

```c
//! (BLJ's) This properly handles long jumps from getting forward speed with
//  too much velocity, but misses backwards longs allowing high negative speeds.
if ((m->forwardVel *= 1.5f) > 48.0f) { m->forwardVel = 48.0f; }
```

`mario_actions_airborne.c`, in `update_air_without_turn`, is the brake, carrying the
decompilation's own marker:

```c
//! Uncapped air speed. Net positive when moving forward.
if (m->forwardVel > dragThreshold) { m->forwardVel -= 1.0f; }
if (m->forwardVel < -16.0f)        { m->forwardVel += 2.0f; }
```

`mario_actions_moving.c`, in `act_long_jump_land`, holds Nintendo's own fix, compiled out under
`VERSION_US`. It is a citation rather than a control: the Shindou branch does not build against
this project's libsm64, so the negative control it would give was never actually run here.
Nintendo's own fix is evidence about what causes the bug; it is not a measurement made in this
repository.

```c
#ifdef VERSION_SH
    // BLJ (Backwards Long Jump) speed build up fix, crushing SimpleFlips's dreams since July 1997
    if (m->forwardVel < 0.0f) { m->forwardVel = 0.0f; }
#endif
```

## Why the chain lives or dies

Per cycle the chain is

    v_launch' = 1.5 * (v_launch - d * k)

for `k` air frames at decay `d` per frame. Growth therefore needs `|v| > 3 d k`. Measured `d` is
about 0.85 per frame with the stick held fully against Mario's facing and 2.35 with no stick at all.

The `+= 2.0` term makes the air phase an attractor at exactly -16. Any long air phase pins the
landing speed near -15 no matter how fast the launch was, which caps the next launch at
`1.5 * 16 = -24`. This is the real ceiling, and it is why air time is the only variable that
matters.

`ACT_LONG_JUMP` also gets half gravity, `m->vel[1] -= 2.0f` in `apply_gravity`, so a long jump
hangs for about 30 frames over level ground. Flat ground and uniform ramps therefore cannot
bootstrap at all, whatever the stick does.

A steep ramp would shorten the air phase, but `mario_floor_is_slippery` treats any floor with
`normal.y <= 0.7880108` as a slide, which is about 38 degrees, and then `should_begin_sliding`
returns true for any `forwardVel <= -1.0` and sends the landing to `ACT_BEGIN_SLIDING` instead of
back into a long jump. Stair treads are level, `normal.y = 1.0`, however steep the staircase
envelope is. That is the structural reason every real BLJ spot in the game is a staircase.

Measured on synthetic geometry, one input program, varying only the floor: 15 geometries crossed
with 4 air-stick magnitudes, 60 scripted runs (`results/blj_runaway.json`). Peaks below are at
full deflection, since mixing magnitudes across rows is what made an earlier version of this table
wrong.

| geometry | min air frames | peak `forwardVel` at stick 1.0 |
| --- | ---: | ---: |
| flat ground | 30 | -21.7 |
| ramp, 20 to 37 degrees | 18 to 24 | -23.1 |
| ramp, 40 degrees | 30 | -21.7, slides out |
| stairs, rise 75 run 100 | 1 | **-610.0** |
| stairs, rise 100 run 100 | 1 | **-585.5** |
| stairs, rise 100 run 60 | 1 | **-281.6** (-351.3 at stick 0.5) |

Eight of the 60 runs run away, and all eight share `min_air_frames == 1`: the chain needs a floor
that puts Mario back on the ground on the very next frame. Nothing else in the sweep separates the
runaways from the rest.

## The task

`src/env/endless_stairs.py` loads `castle_inside/areas/2/collision.inc.c`, the real endless
staircase: 1923 triangles, treads of rise 25.6 and run 51.2, a corridor at `x` in [-409, 0], the
bottom landing at `y = 3174` and the top landing at `y = 5018`.

The loop is not geometry. Twelve triangles carry the surface type `SURFACE_INSTANT_WARP_1B`, and
`check_instant_warp` in `level_update.c` displaces Mario by the level script's own
`INSTANT_WARP(0, 2, 0, -205, 410)` whenever his current floor is one of them, unless the save holds
70 stars. libsm64 carries Mario and surfaces but no level logic, so the environment reimplements
that one check against the same surface type and the same displacement.

The check samples Mario's floor once per frame and the warp zone is 154 units deep. Skipping it
therefore needs a per frame displacement larger than 154 units, while Mario's fastest ordinary
movement is the long jump clamp at 48. Reaching the top landing cannot be faked.

The scripted expert in `src/agent/scripted.py` is the reference: walk away from the rise, crouch
slide, long jump, then hold the stick back and re-press A on every landing. It gets thrown back
down 37 times while building speed, then one jump crosses the trigger zone inside a single frame
and it reaches `y = 5074`.

## The environment

`src/env/blj_env.py` is a Gymnasium environment.

- Observation: 24 floats, normalized. Position relative to the goal, velocity, `forwardVel`, facing,
  floor normal and height, whether the current floor is a warp trigger, action group flags, action
  timer, air frames, and the buttons held last frame.
- Action: `Discrete(36)`, nine stick directions at full deflection crossed with A and Z. Full
  deflection matters, but it is not required: full deflection roughly doubles the set of
  geometries the chain runs away on, 4 of 15 against 2 of 15 at stick 0.5, and on the two steepest
  tread profiles it survives down to 0.5 with peaks of -347.0 and -351.3. On those two the peak is
  not even monotone in stick magnitude -- rise 100 run 60 reaches -351.3 at 0.5 and only -281.6 at
  1.0 -- because more stick also means more per-frame decay.
- Reward: every term is a weight in `RewardConfig`, because the weights are the experiment. Shaping
  pays on each new record backward speed rather than on the instantaneous value, so it stays
  potential based and cannot be farmed by hovering.

libsm64 keeps one static surface set per process, so vectorized training uses one subprocess per
environment.

## The ablation ladder

| rung | reward |
| --- | --- |
| terminal | reaching the top landing, nothing else |
| speed | plus a record backward speed term |
| height | plus a record climbed height term, which states the goal and not the method |
| height_speed | both shaping terms |
| curriculum | plus a bonus per stage of a hand written recipe |

The ladder is not a single chain. `height` and `speed` are independent branches off `terminal`
and `height_speed` is their combination, so the invariant the tests hold is that each rung's
difference from `terminal` is exactly its own advertised ingredients.

`curriculum` pays for reaching each of grounded, crouched, long jump, backwards long jump and
chained. That names the action recipe rather than the goal, which is why it was the first rung to
work and why it was dropped: succeeding at it says nothing except that the answer was supplied.
Its runs are kept in the history rather than deleted, as the upper bound on how much help is
possible.

Both shaping terms pay on records rather than per frame, which the warp loop makes necessary. Any
per frame progress term would pay an agent forever for re-climbing the same steps while the loop
resets him, and looping is easier than the exploit.

Action repeat was the fourth rung and it was measuring the wrong thing. `INPUT_A_PRESSED` never
re-latches on a held button, so holding one action for k frames forces a minimum A press period of
2k. Measured with this project's own expert on the real staircase: period 2 reaches -715, period 4
reaches -31.6, period 6 reaches -20.8, period 8 reaches -22.1. Above a repeat of 1 the exploit is
not harder to learn, it is impossible to express, and the runs confirmed it with zero successes at
every repeat of 2 and above. Recording that as "shaping insufficient" would have been a false
negative, so the axis is out until A press parity is separated from action repeat and the discount
is corrected by `gamma ** action_repeat`.

## Watching it

`scripts/record_episode.py` records an episode from either the scripted expert or a trained model
and writes a replay. `tools/pack_replays.py` and `tools/make_viewer.py` turn replays into a single
self contained HTML viewer with a side view, the live controller input, the `forwardVel` trace with
the -16 attractor marked, and a frame scrubber.

## An input bug worth remembering

The earlier sweep found nothing, and the reason was the stick. `libsm64.c` does
`gController.stickX = -64.0f * inputs->stickX`, so the API wants `[-1, 1]`. The sweep passed
`stickY` values of `-64`, `0` and `+64`. At `0` there is no backwards drive at all; at `+-64` the
stick magnitude becomes 4096 and `forwardVel` jumps to -6142 in one frame, which throws Mario out
of the level before a chain can form. The sweep covered only those two useless cases and never the
range in between.

## Reproducing

    ./scripts/setup.sh                      # clone and patch libsm64 and sm64-port, then build
    # place a Super Mario 64 US ROM at roms/baserom.us.z64, sha1 9bef1128...
    make test
    PYTHONPATH=. python3 scripts/validate_env.py    # the expert beats the staircase
    PYTHONPATH=. python3 scripts/blj_runaway.py     # the geometry sweep
    PYTHONPATH=. python3 scripts/record_episode.py  # record a replay
    make viewer

Three measurements need trained policies, which are not in the repo. `scripts/transfer_test.py`
needs the 24 final models, `scripts/action_occupancy.py` needs a checkpoint directory, and
`scripts/transfer_expert.py` needs neither and can be run on a fresh clone.

No ROM is distributed here. `patches/` holds the changes this project makes to libsm64, applied by
`setup.sh` on a fresh clone: one to export Mario's floor and action detail, which the instant warp
check needs, and one to stop the Makefile listing its generated sources twice.

## Layout

    src/env/       libsm64 binding, geometry, real collision import, the staircase, the environment
    src/agent/     the scripted expert and the driver adapters
    src/train/     the ablation ladder and the PPO harness
    scripts/       setup, validation, sweeps, recording, training entry points
    tools/         replay packing and the HTML viewers
    cluster/       Slurm jobs
    patches/       changes to vendored third party code
    results/       measured output
