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

The validation is transfer. The same policy should work on geometry it never trained on.

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

Results below are from a run still in progress, 20M steps per seed. The two baselines are at
12.4M and the two height rungs at 7.2M, so the comparison across rungs is not yet fair.

| rung | seeds with successes | best peak `forwardVel` | notes |
| --- | --- | ---: | --- |
| terminal | 1 of 3 | -5588 | seed 2 at 33,606 successes in 35,856 episodes, last 500 episodes at 100% |
| speed | 1 of 3 | -6472 | seed 1, 735 successes |
| height | **0 of 6** | -56.9 | every seed pinned at return 0.44 |
| height_speed | **4 of 6** | -7060 | 990, 831, 532 and 224 successes |

The sharpest signal is the last two rows against each other. Height progress alone never crosses,
in six of six seeds. Adding the backwards speed term takes it to four of six. Height reliably
walks Mario to the barrier and stops; speed is what carries him through it.

The height rung failing is the designed outcome rather than a disappointment, and it is worth
saying why. Its reward states the goal and nothing else, and it goes flat exactly where ordinary
movement stops working, so there is no gradient across the discontinuity that the exploit lives
on. A policy that just holds the stick up the stairs earns 0.419 of it, reaches y 3942, and is
then thrown back down 302 times for nothing.

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
`VERSION_US`. Rebuilding with `-DVERSION_SH` makes the BLJ impossible, which is a ground truth
negative control shipped by the original developers rather than one invented here:

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

Measured on synthetic geometry, one input program, varying only the floor:

| geometry | air frames | peak `forwardVel` |
| --- | ---: | ---: |
| flat ground | 30 | -22.6 |
| ramp, 20 to 37 degrees | 18 to 25 | -23.1 |
| ramp, 40 degrees | 30 | -21.7, slides out |
| stairs, rise 75 run 100 | 1 | **-610.0** |
| stairs, rise 100 run 60 | 1 | **-351.3** |

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
  deflection matters. The runaway needs a stick magnitude around 0.8 to 1.0 and does not happen at
  0.5.
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
