# Handoff: everything needed to write about this project

Written 2026-09-15, after the 24 run ladder completed. Every number here was measured, and the
command that produced it is in the repo. Where a claim is uncertain it says so.

## The one sentence

An RL agent found Super Mario 64's backwards long jump from a reward that pays 1.0 for standing on
the top of the endless staircase and nothing else, so the answer to "how much reward shaping does
this need" is none, and what shaping buys is speed and reliability rather than possibility.

## Why the task is worth using

Reaching the top of the endless staircase cannot be faked. The staircase loops Mario back down,
and the loop is not geometry. Twelve of its 1923 collision triangles carry the surface type
`SURFACE_INSTANT_WARP_1B`, and `check_instant_warp` in the decompilation's
`src/game/level_update.c:537` displaces Mario by the level script's own
`INSTANT_WARP(0, 2, 0, -205, 410)` whenever his current floor is one of them, unless the save file
holds 70 or more stars.

The check samples Mario's floor once per frame and the warp zone is 154 units deep in z. Skipping
it therefore needs a per frame displacement larger than 154 units, while Mario's fastest ordinary
movement is the long jump clamp at 48. Success certifies the exploit instead of approximating it,
and there is no partial credit for almost doing a backwards long jump.

Verified directly: placing Mario on a warp triangle at y 3968, z 980 moves him to y 3763, z 1390,
exactly the declared offset.

## The mechanism, with citations

Three annotated sites in the decompilation carry the whole bug, and two of the three carry the
decompilation's own warning comments.

`mario.c`, `set_mario_action_airborne`, the `ACT_LONG_JUMP` case, is the amplifier. The clamp is
one sided, so negative speed multiplies without bound:

```c
//! (BLJ's) This properly handles long jumps from getting forward speed with
//  too much velocity, but misses backwards longs allowing high negative speeds.
if ((m->forwardVel *= 1.5f) > 48.0f) { m->forwardVel = 48.0f; }
```

`mario_actions_airborne.c`, `update_air_without_turn`, is the brake:

```c
//! Uncapped air speed. Net positive when moving forward.
if (m->forwardVel > dragThreshold) { m->forwardVel -= 1.0f; }
if (m->forwardVel < -16.0f)        { m->forwardVel += 2.0f; }
```

`mario_actions_moving.c`, `act_long_jump_land`, holds Nintendo's own fix, compiled out under
`VERSION_US`, comment and all: "BLJ (Backwards Long Jump) speed build up fix, crushing
SimpleFlips's dreams since July 1997".

## Why the chain lives or dies, and why it must be stairs

Per cycle the chain is `v' = 1.5 * (v - d*k)` for `k` air frames at decay `d` per frame, so growth
needs `|v| > 3dk`. Measured `d` is about 0.85 per frame with the stick held fully against Mario's
facing, and 2.35 with no stick at all.

The `+= 2.0` term makes the air phase an attractor at exactly -16. Any long air phase pins the
landing speed near -15 however fast the launch was, which caps the next launch at `1.5 * 16 = -24`.
Air time is therefore the only variable that matters. `apply_gravity` also gives `ACT_LONG_JUMP`
half gravity, `m->vel[1] -= 2.0f`, so a long jump hangs about 30 frames over level ground.

A steeper ramp would shorten the air phase, but `mario_floor_is_slippery` treats any floor with
`normal.y <= 0.7880108` as a slide, about 38 degrees, and then `should_begin_sliding`
(`mario_actions_moving.c:474`) returns true for any `forwardVel <= -1.0` and diverts the landing
into `ACT_BEGIN_SLIDING`. Stair treads are level, `normal.y = 1.0`, however steep the staircase
envelope is.

That is the structural reason every real backwards long jump spot in the game is a staircase, and
it is measurable. One scripted input program, varying only the floor:

| geometry | air frames | peak `forwardVel` |
| --- | ---: | ---: |
| flat ground | 30 | -22.6 |
| ramp, 20 to 37 degrees | 18 to 25 | -23.1 |
| ramp, 40 degrees | 30 | -21.7, slides out |
| stairs, rise 75 run 100 | 1 | **-610.0** |
| stairs, rise 100 run 60 | 1 | -351.3 |

## The false negative that had blocked the project

Before this work the chain was believed not to reproduce, with a wall somewhere above `|v| = 45`.
There is no wall. The earlier sweep passed `stickY` values of -64, 0 and +64 into an API that
wants `[-1, 1]`. `libsm64.c:241` does `gController.stickX = -64.0f * inputs->stickX`, so:

- at 0 there is no backwards drive at all
- at plus or minus 64 the stick magnitude becomes 4096 and `forwardVel` reaches -6142 in a single
  frame, which throws Mario out of the level before a chain can form

The sweep covered only those two useless cases and never the range between them. `blj_sweep.json`
also held one run rather than the 48 the README claimed. A second wrong belief was that a short air
phase and a low decay were coupled so that neither setting could satisfy the growth condition. They
are compatible: the stick opposed to Mario's facing **is** the stick pointing along his direction of
travel once his speed is negative.

Worth a paragraph in any write up, because the project's central negative result was an input bug.

## The substrate, and how far it can be trusted

libsm64 compiles the decompilation's Mario as a shared library. `sm64_mario_tick` steps one frame.

**Analytically identical.** A file by file diff against upstream `n64decomp/sm64` @`9921382a` found
the entire backwards long jump chain byte identical: the 1.5x amplifier, `update_air_without_turn`,
`common_air_action_step`, `act_long_jump`, `act_long_jump_land`, `perform_air_step` and the quarter
step functions.

**Behaviourally identical.** A headless `sm64-port` build replayed TASVideos movie 2016M, the 0 star
run, 8772 frames, on the cluster in 1.98 seconds, 147 times realtime, with no desync, ending in
`LEVEL_BOWSER_3` in `ACT_DIVE`. 8737 of 8772 controller records match the movie byte for byte and
all 35 mismatches sit in the pre-Mario intro, frames 118 to 169; frames 170 to 8772 have zero
mismatches. The trace holds six backwards long jump chains. The longest is castle area 1, frames
3709 to 3739, 16 re-launches, peak `forwardVel` **-7468.75**. Replaying that chain through libsm64
reproduced **45 of 45 frames bit identically** in action, position, velocity and forward velocity,
including the peak. Amplification ratios match to every digit, 1.4952 through 1.4996, and the decay
closes exactly: `1.5 * 440.2797 - 0.85 = 659.56955` against a recorded -659.5695.

**Deterministic across machines and submissions.** The scripted expert reproduces bit identically on
arm64 macOS and x86_64 Linux. An identical reward and seed reproduced a 6.3 million step trajectory
exactly across two separate cluster submissions on different nodes, and the two runs' policy weights
are bit identical, worst difference 0.000 across all 12 tensors.

**Where it diverges from the console**, all confined to the surface layer, none on the chain:
libsm64 rewrote `find_floor_from_list` to return the highest floor rather than the first hit, which
removes the decompilation's documented Surface Cucking bug; it dropped the s16 cast of position and
the `LEVEL_BOUNDARY_MAX` guard, so there is no parallel universe wrap; `FLOOR_LOWER_LIMIT` went from
-11000 to -110000 and `CELL_HEIGHT_LIMIT` from 20000 to 100000; `find_water_level` and
`find_poison_gas_level` are stubbed to constants; and `level_trigger_warp` is a no-op, so leaving the
geometry freezes Mario instead of killing him. The build is `-O0`, no `-ffast-math`, `-DVERSION_US`.

**One thing does not work.** Rebuilding with `-DVERSION_SH` to get Nintendo's own fix as a
ground truth negative control does not compile in libsm64. The control is still worth describing,
but do not claim it was run.

## The environment

Gymnasium environment, `src/env/blj_env.py`.

- Observation: 24 normalized float32. Position relative to the goal, velocity, `forwardVel`, facing,
  floor normal and height, whether the current floor is a warp trigger, action group flags, action
  timer, air frames, buttons held last frame.
- Action: `Discrete(36)`, nine stick directions at full deflection crossed with A and Z. Full
  deflection is not a simplification: the runaway needs a stick magnitude around 0.8 to 1.0 and
  does not happen at 0.5.
- Throughput, measured by `scripts/measure_throughput.py` into `results/throughput.json` on an
  eight core arm64 mac: 7,712 environment steps per second in one process with uniform random
  actions, and 7,099 in the training loop across eight environments. The loop figure is a slope
  fitted through a 16k, a 41k and a 123k step run rather than a single timing, because 1.8 s of
  that is process startup and a short run charges all of it to the rate; the middle run then sits
  0.39 s off the line. CPU only; the bottleneck is the physics and the policy is a small MLP.
- Eight environments is what the published ladder ran: `cluster/train_ladder.sbatch` asks for
  `--cpus-per-task=8` and passes `NUM_ENVS="${NUM_ENVS:-${SLURM_CPUS_PER_TASK:-12}}"`, so the
  twelve in that fallback never fired under Slurm. An earlier version of this line quoted the
  fallback as fact and a throughput number nobody had measured, and both errors reached a figure
  in the post before being caught. None of the published runs recorded their own environment
  count, which is why `RunMetrics` now carries `num_envs`.
- One libsm64 process holds one static surface set, so vectorized training uses one subprocess per
  environment with the spawn start method.

**The environment contributes no randomness.** `spawn_jitter` defaults to 0.0 and the jobs do not
set it. Three different environment seeds produce identical 300 frame trajectories, verified. A
seed therefore controls only the policy's initial weights and its action sampling. "4 of 6 seeds
solved" means four of six independently initialized agents got lucky in exploration on a fixed
puzzle, which is why outcomes bifurcate rather than cluster, and why reporting a mean across seeds
would describe no agent that exists.

## The reward design, including the bug worth writing about

The first version's shaping was unbounded. It paid `0.01` per unit of record backwards speed, which
at a peak of -6000 paid about 59 against a goal worth 1.0. The agent was being paid sixty times more
for going fast than for finishing. Every term is now a bounded fraction of the goal:

| term | pays | ceiling |
| --- | --- | ---: |
| `terminal` | reaching the top landing | 1.00 |
| `height_weight` | record height climbed while grounded | 0.25 |
| `speed_weight` | record backwards speed, saturating at the escape speed 154 | 0.25 |
| `curriculum_weight` | stages of a hand written recipe | 0.25 |

So 1.0 always means the exploit was performed, and rungs are comparable without rescaling.

Two details make the shaping honest. Every term pays only on a new **record**, because the warp
resets Mario constantly and a per frame progress term would pay him forever for re-climbing the same
steps. And height counts only while Mario has a floor under him, because a plain jump buys about 220
units of air he could otherwise farm on the spot.

Verified scale, one run each:

```
                 max possible   climber   expert   reached top
terminal                 1.00     0.000    1.000          True
speed                    1.25     0.000    1.250          True
height                   1.25     0.105    1.240          True
height_speed             1.50     0.105    1.490          True
```

The climber walks up the stairs and gets looped. It earns 0.105 of the 0.25 available, reaches
y 3942, and is then thrown back 302 times for nothing.

**Bounding the speed term is itself a result.** It took speed shaping from 2 of 3 seeds with a
fastest discovery at 3.7M steps, to 6 of 6 seeds with a fastest discovery at 0.58M.

## Results: 24 runs, 4 rungs, 6 seeds each, 20M steps, all completed

| rung | reward | seeds solved | first success (fastest) | best success rate |
| --- | --- | ---: | ---: | ---: |
| terminal | goal only | **1 of 6** | 6,324,840 | 97.6% |
| speed | + record speed | **6 of 6** | 580,644 | 99.7% |
| height | + record climb | **0 of 6** | never | 0% |
| height and speed | both | **3 of 6** | 699,852 | 99.4% |

Per seed first successes: speed at 0.58M, 1.36M, 2.04M, 3.38M, 16.45M, 17.21M. Height and speed at
0.70M, 1.45M, 7.62M. Terminal at 6.32M.

### The headline

**Pure sparse reward solved it.** The `terminal` rung pays 1.0 for standing on the top landing and
nothing else, and its episode returns take only the values 0.0 and 1.0 across all 95,093 episodes.
Seed 2 found the exploit at step 6,324,840 and ended at 92,801 successes, 97.6%, with a success rate
of 0.992 over the last 500 episodes. Mean episode length collapses from the 3000 frame truncation
ceiling to about 170 frames: it does not merely learn the trick, it learns to do it immediately.

So "RL could never discover this" is false in this environment. One seed in six, and 6.3 million
steps, but false.

### The best negative result

**Height shaping is worse than no shaping at all: 0 of 6 against terminal's 1 of 6.** Its reward
states the goal honestly, go up, and says nothing about method. It goes flat exactly where ordinary
movement stops working, so there is no gradient across the discontinuity the exploit lives on, and
the barrier becomes a comfortable place to sit.

The swarm view makes this visible. The height swarm at 4.0M steps reaches y 3942, the warp trigger
starts at 3917, and is thrown back **572 times** in 800 frames. At 9.0M, 416 times. At 16.5M, 421
times. And its peak backward velocity stays pinned at exactly **-16.00**, the air attractor, in
every panel. Sixteen million steps of training and those agents never once tried to go fast.

A dense reward that stops short of the discontinuity is not a weaker version of the sparse one. It
is a trap.

## Why Mario stomps at the start

The early behaviour looks like learning and is not. `scripts/action_occupancy.py` takes 41
checkpoints of the solving run, `terminal` seed 2, plus two null policies, and measures where the
frames go: four rollouts each, 1500 frames each, resetting as needed so a policy that finishes an
episode in 120 frames and one that never finishes are measured over the same budget. Mean over the
four rollouts, range in brackets:

| policy | frames in ground pound | frames in long jump | median peak backward vel | episodes solved |
| --- | ---: | ---: | ---: | ---: |
| uniform random | 75.1% (72.0-77.7) | 0.0% (0.0-0.0) | -15.9 | 0 of 0 |
| untrained network, before one gradient step | 73.3% (69.3-75.5) | 1.0% (0.0-2.0) | -16.0 | 0 of 0 |
| 2.0M checkpoint | 74.1% (69.6-79.6) | 0.5% (0.0-2.0) | -16.0 | 0 of 0 |
| 6.0M checkpoint | 34.0% (20.9-51.9) | 50.6% (30.8-66.0) | -35.6 | 0 of 0 |
| 6.3M checkpoint | 81.4% (79.3-83.7) | 0.0% (0.0-0.0) | -15.7 | 0 of 0 |
| 7.0M checkpoint | 8.4% (3.2-12.6) | 56.3% (49.7-66.6) | -1557.6 | 19 of 19 |
| 13.5M checkpoint | 0.0% (0.0-0.0) | 53.0% (49.9-58.1) | -619.1 | 43 of 43 |
| 20.0M checkpoint | 0.0% (0.0-0.0) | 50.1% (50.0-50.3) | -657.1 | 48 of 48 |

18 of the 36 actions hold Z and 18 press A, and A pressed while airborne with Z held is the ground
pound trigger. Ground pound is also cheap to enter and slow to leave, `ground_pound` into
`ground_pound_land`, so its share of frames far exceeds its share of actions. It is an attractor in
the action space that eats most of the early exploration budget.

The 2.0M checkpoint is indistinguishable from random by this measure: its range, 69.6 to 79.6, sits
inside the range four rollouts of uniform random cover on their own, 72.0 to 77.7. That is the
correct behaviour of PPO on a reward that is 0.0 for every episode.

What the earlier version of this table got wrong is the sentence that followed it. Before the first
success the policy does not sit still at its initialization. Every 0.5M checkpoint from 0.5M to 6.5M
is in `results/action_occupancy.json`, and the walk between them is large and not monotone: 3.5M and
5.0M already spend 8% and 22% of their frames in long jump, one 4.0M rollout spends 62%, 5.5M is
back to 0.0%, 6.0M holds 50.6% across all four rollouts and still solves nothing, and then **6.3M,
the last checkpoint before the first success at 6,324,840 steps, has reverted to 81.4% ground pound
and 0.0% long jump** -- further into the attractor than uniform random ever goes.

The mechanism is that "no reward" is not "no gradient". With every return exactly 0.0 the advantage
of a state is `0 - V(s)`, and `V` is a randomly initialized network being dragged toward zero, so
the advantages are nonzero, meaningless, and correlated across a rollout. Episodes also end by
truncation at 3000 frames, where the value target bootstraps on `V(s_T)` at `gamma = 0.999` rather
than on a terminal zero, which keeps feeding the same noise back in. So PPO spends six million
frames doing a random walk in policy space, driven by its own critic's initialization and spread out
by the entropy bonus. The walk sometimes passes near the exploit and wanders off again. The learning
curve is still a step function rather than a ramp, which is the classic hard exploration shape, but
the step is not the moment the policy first finds long jumps -- it had found and lost them repeatedly
by then. The step is the moment a success finally lands inside a rollout and the advantage means
something.

Once that happens the transition takes 0.5M frames. 6.3M solves nothing, 6.5M solves one episode in
four rollouts, 7.0M solves 19, and from 7.5M on ground pound never again exceeds 10% while long jump
holds near 50%.

## What the policies actually learned: one staircase

Every run in the ladder saw exactly one flight of stairs. A policy that reaches the top landing has
either learned the backwards long jump, which is a property of Mario's physics, or learned a
sequence of inputs that works on 25.6 unit treads 51.25 units deep, which is a property of one
level. The training curve cannot tell those apart. A second staircase can.

`scripts/transfer_test.py` drops all 24 final policies, unchanged and never fine tuned, on ten
flights: the castle's own, a synthetic rebuild of its tread geometry, and eight variations. Eight
episodes each, the first deterministic and seven sampled, 3000 frame cap, 1,920 episodes in total.
The synthetic scenes come from `src.env.endless_stairs.synthetic_scene`, which holds the castle's
spawn, its 1792 unit climb and the self similarity of its warp fixed, and varies the rise, the run
and whether the steps have vertical faces between the treads at all. "Rate" below is the mean
success rate over the ten policies that solve the castle, the only ten that can do the exploit at
all; the other fourteen never leave the ground on any flight.

| scene | rise | run | faces | tris | policies | rate | best peak | best y | expert peak |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| castle | 25.6 | 51.25 | yes | 1923 | **10 of 24** | 1.000 | -655.9 | 5018 | -31.2 |
| rebuilt | 25.6 | 51.25 | yes | 284 | **9 of 24** | 0.775 | -834.6 | 4966 | -32.8 |
| rebuilt, faces removed | 25.6 | 51.25 | no | 144 | **9 of 24** | 0.825 | -913.5 | 4995 | -32.8 |
| rise 50, faces removed | 50 | 51.25 | no | 76 | **9 of 24** | 0.325 | -633.4 | 5002 | -23.0 |
| rise 75, faces removed | 75 | 100 | no | 52 | **5 of 24** | 0.212 | -649.0 | 4995 | -454.1 |
| rise 26, run 100 | 25.6 | 100 | yes | 284 | 0 of 24 | 0.000 | -52.8 | 3971 | -33.2 |
| rise 50 | 50 | 51.25 | yes | 148 | 0 of 24 | 0.000 | -40.8 | 3952 | -22.1 |
| rise 75 | 75 | 100 | yes | 100 | 0 of 24 | 0.000 | -40.9 | 3953 | -22.2 |
| rise 100 | 100 | 100 | yes | 76 | 0 of 24 | 0.000 | -40.5 | 4002 | -22.7 |
| rise 100, faces removed | 100 | 100 | no | 40 | 0 of 24 | 0.000 | -595.6 | 4002 | -578.5 |

Goal height is 4914 to 4966 on every flight, so "best y" under 4010 means no policy climbed past the
warp band at all.

**The rebuild is the control, and it passes.** 284 triangles standing in for 1923, built from a rise
and a run rather than loaded from the decompilation, and nine of the ten castle solvers keep the
exploit on it at a 0.775 rate. A transfer failure elsewhere is therefore a fact about the geometry,
not an artefact of the rebuild. The one policy that does not survive the rebuild is `speed` seed 2,
which is 8 of 8 on the castle and 0 of 8 on a flight with the same treads; the real staircase is a
narrow shaft, its stair column spanning x -409 to 0 with 235 wall triangles around it, while the
rebuild is a 6000 unit wide flight in open space, and that policy is the one that appears to be
using the walls.

**Every change that leaves the vertical faces in place kills all 24 policies.** The peaks fall from
-656 to about -41, and -41 is roughly two launches: 1.5 times the -16 air drag attractor is -24, and
one more cycle reaches about -40 before drag wins. So the policies still crouch and still launch
backwards. What stops is the compounding.

**The height of the face between treads is what decides it, and the policies are obstructed by it
rather than using it.** At the castle's own tread depth, raising the rise from 25.6 to 50 takes 10
of 24 to 0 of 24 and the peak from -656 to -41. Delete those 50 unit faces and leave everything else
identical, and 9 of 24 come back at -633. Delete the castle's own 25.6 unit faces and nothing
happens at all: 9 of 24 either way, rate 0.775 against 0.825. The chain does not need the risers and
cannot survive tall ones.

**It is not only the faces.** `rise 26, run 100` keeps the castle's 25.6 unit faces and only doubles
the tread depth, and all 24 collapse to -52.8. Both numbers have to be near the castle's.

**Speed is necessary and not sufficient.** On `rise 100, faces removed` the chain runs away to
-595.6, more than the 300 units the warp band there is deep, and in 192 episodes no policy ever got
above y 4002, one tread above the band. Escaping the warp means landing past the band, not merely
being fast somewhere below it.

### The expert control, and what it does not license

`scripts/transfer_expert.py` runs the project's hand written expert, the one
`results/blj_runaway.json` reports on ramps and abstract staircases, on the same ten scenes through
`src.agent.scripted.run_chain`, which bypasses the environment: no reward, no episode limit, no
warp. Where it runs away the geometry demonstrably admits a chain, and it runs away on exactly the
two flights with no faces and a steep rise, -454.1 and -578.5. Everywhere else it reaches -22 to
-33, the air drag attractor, **including on the castle**, where ten learned policies reach -656.

So the expert is an informative control only where it succeeds; its failures say nothing about the
geometry. Stating the rest plainly: the fixed two frame repress schedule that runs away on abstract
staircases does not run away on the real one, and PPO found something it does not reproduce. The
report's `escapes_warp` column is a comparison of peak speed against the band depth, not an observed
escape, because `run_chain` never applies the warp.

Worth a figure: the 24 by 10 grid of per policy success rates is graded rather than binary, running
1.000, 0.775, 0.825, 0.325, 0.212 and then five columns of zero, and the same ten policies appear in
every non zero column.

## The two rungs that were cut, and why

**Action repeat was measuring a capability limit, not learning.** Re-launching the chain needs
`INPUT_A_PRESSED`, which never re-latches on a held button, so holding one action for k frames
forces a minimum A press period of 2k. Measured with the project's own scripted expert: period 2
reaches peak -715, period 4 reaches -31.6, period 6 reaches -20.8, period 8 reaches -22.1. Above a
repeat of 1 the exploit is not harder to learn, it cannot be expressed, and all nine runs confirmed
it with zero successes. Recording that as "shaping insufficient" would have been a false negative.
If revisited: separate A press parity from action repeat, and use `gamma ** action_repeat` so the
horizon is not confounded with the control resolution.

**Curriculum was dropped for being too helpful.** It paid 0.25 per stage of a hand written recipe,
grounded to crouched to long jump to backwards long jump to chained. It was the first rung to work,
2 of 3 seeds with a first success near 300k steps, roughly 21 times faster than terminal. It names
the method rather than the goal, so succeeding at it says only that the answer was supplied. Its
runs are kept as the upper bound on how much help is possible.

## Engineering worth writing about

**Mario's real mesh comes out of the API.** Every frame libsm64 returns a flat 752 triangle soup
with the bone matrices already baked, per vertex normals, a six entry colour palette, and the ROM's
704 by 64 RGBA texture atlas, wound counter clockwise in the same right handed sense the collision
loader uses, with zero degenerate triangles in 526,400 checked.

**Only 6.6% of Mario is textured**, 50 of 752 triangles, and that split is the original game's. SM64
textures the face details, the cap logo, the hands and the metal cap, and flat shades the shirt,
overalls, skin and shoes. The atlas holds four eye states, open, half lidded, closed, and X-X dead
eyes, and Mario blinks correctly during the exploit because the ROM's own animation data drives it.

**A pose dictionary makes a crowd cheap.** Transforming each frame's mesh into Mario local space and
keying on `(animID, animFrame)` collapses an 1100 frame run to **69 unique poses**, with median and
99th percentile disagreement of exactly 0.000 units and a worst case of 17.6 on about 1% of frames,
against a Mario about 160 units tall. That is 10 bytes per Mario per frame, so 32 Marios over 900
frames is 0.29 MB. Do not try to improve it with the graphics node transform: un-rotating by
`gfxAngle` with the decompilation's `mtxf_rotate_zxy` convention gives 410 units of error, because
libsm64's `geo_process_root_hack_single_node` bakes the mesh at `m->pos` with yaw only.

**Many Marios share one room for free.** Each Mario gets its own `GlobalState`, while the collision
list is a file scope static in `load_surfaces.c:23` shared by all of them. Verified with six
simultaneous Marios giving six distinct trajectories. Throughput is flat at about 7,900 Mario frames
per second from 1 to 64 Marios, so the harness adds no measurable per population cost and the
physics is the whole bill.

**The sound is the game's own, and the joke is free.** libsm64 compiles in the decompilation's audio
engine, so `sm64_audio_init` and `sm64_audio_tick` produce the ROM's samples driven by the same
`play_sound` calls Mario's actions make. `act_long_jump` plays `SOUND_MARIO_YAHOO`, and a working
chain re-enters that action on nearly every frame, so the exploit sounds like 37 seconds of
uninterrupted Yahoo. Measured peak amplitude 19661, with 1073 of 1113 frames loud.

Four details fell out of that:

- The audio engine's state is a file scope global rather than part of the per Mario `GlobalState`, so
  **one audio tick per frame renders the entire swarm** and the game's own mixer combines it. No
  mixing code required.
- It **saturates at about eight simultaneous Marios**, because the engine has a fixed voice limit and
  drops the surplus itself. RMS goes 4795, 6179, 6174 for 1, 8 and 48 Marios. You cannot make 48
  Marios louder than 8; the N64 will not let you.
- 48 Marios **clip** at full volume, peak 32768. At volume 0.4 the peaks land 16385 to 21173.
- Peak amplitude **falls** as training progresses, 21173 at 2.0M down to 16385 at 13.5M, because
  mastery means less thrashing. Competence is quieter.

Sync needs no fitting: `audio_tick` returns a constant 1088 samples per channel per frame, so
declaring the stream at `1088 * 30 = 32640` Hz makes duration equal frame count over 30 exactly,
37.100 s for 1113 frames.

## Gotchas that cost real time

- **`calibrate_stick` replaces the level.** It loads a flat ground plane to walk on, and surfaces are
  global in libsm64, so calling it while an environment holds the staircase replaces the staircase.
  Mario then spawns 3200 units above a floor at y 0 and falls out of the level. It fails silently and
  produces a short run with a nonsense peak.
- **Coordinate products overflow.** The in triangle tests in `find_floor_from_list` use s32
  arithmetic, so differences of coordinates must keep products under 2^31. A 30000 by 8000 ramp
  works; 200000 fails.
- **Solving seeds advance more slowly in wall clock**, because finishing episodes in 170 frames
  instead of 3000 means far more resets. Reading an in flight log's newest bin therefore takes the
  median over whichever seeds are ahead, which are the ones that never learned, and draws a cliff to
  zero that no seed experienced. Restrict medians to bins every seed reached.
- **Never report a mean across seeds here.** Outcomes bifurcate: one seed at 1.00 and five at 0.00.
  Report seeds solved out of N.
- **Rendering catches what code review cannot.** Two figure bugs got through a clean `node --check`:
  a page that rendered completely blank because a helper dereferenced a rung that had been removed,
  and end labels colliding because a de-collision loop contained `index * 0`. Screenshot the output
  and look at it.

## Where everything is

- **Repo**: `github.com/yllkryeziu/mario-blj`, private, 9 stacked PRs from `main`. Local checkout at
  `/Users/yll/Desktop/mario-blj`. 96 tests, ruff 0, pyright 0.
- **Cluster**: `ssh berlin1`, `/fast/project/HFMI_SynergyUnit/yll/mario-blj`. Slurm, 224 CPUs per
  node, 7 usable nodes, always `--exclude=hai002` because that node is broken.
- **Final results**: `results/ladder_v2/<rung>/seed_N/` with `episodes.csv`, `metrics.json` and
  checkpoints. Aggregated curves in `results/learning_curves.json`.
- **Committed reductions**, all small enough to live in the repo and all that the post needs:
  `results/episode_stats.json` (42 KB, `tools/prep_episodes.py`, all 975,396 episodes reduced per
  run and clipped to 20,000,000 steps), `results/action_occupancy.json` (159 KB,
  `scripts/action_occupancy.py`, 41 checkpoints by 4 rollouts), `results/transfer.json` (530 KB,
  `scripts/transfer_test.py`, 240 policy by scene rows) and `results/transfer_expert.json`
  (`scripts/transfer_expert.py`).
- **Fetched artefacts, gitignored**: `data/ladder_v2/` (48 files, 53 MB, the per run `episodes.csv`
  and `metrics.json`), `data/models_v2/` (24 final `model.zip`, 43 MB) and
  `data/checkpoints/terminal_seed_2/` (41 checkpoints, 74 MB, every 0.5M steps plus 6,299,748, the
  last one before the first success). Re-fetch with `tar cf -` over `ssh berlin1`; the cluster alias
  `yll` is `cd /fast/project/HFMI_SynergyUnit/yll`.
- **Patches to vendored code**: `patches/`, applied by `scripts/setup.sh` on a fresh clone. One
  exposes Mario's floor and action detail as a 124 byte struct checked with `offsetof` on both sides;
  one stops the libsm64 Makefile listing its generated sources twice, which failed a second build
  with 247 duplicate symbols.

### The viewers

| what | url |
| --- | --- |
| Reward, success rate and episode length per rung | `claude.ai/code/artifact/fa7ee008-c0eb-4145-a3c7-749e45591fe3` |
| Film strip, terminal rung, six checkpoints from untrained | `claude.ai/code/artifact/fb0fbe07-c7c5-481a-8553-969fa07b052e` |
| Four rungs, sixteen panels | `claude.ai/code/artifact/2c750bb7-ee7d-4117-bc33-944ba1086f7a` |
| Mario in 3D with the game's audio | `claude.ai/code/artifact/cbcc9037-9e54-43e1-90a9-c775edb3fb49` |
| Side view with the live controller panel | `claude.ai/code/artifact/d300eb25-0315-48a2-99e8-c2b7abe016f1` |

## Honest limitations to state in the post

- **One seed in six for the headline.** Terminal only solved it once out of six. The claim is that it
  is possible without shaping, not that it is reliable without shaping.
- **Six seeds is thin** for an outcome that bifurcates. 6 of 6 against 3 of 6 is suggestive, not
  established.
- **One task, one level, and the transfer test came back negative.** The policies learned the spot,
  not the technique: 10 of 24 solve the castle, 9 of 24 solve a rebuild of its tread geometry, and 0
  of 24 survive any change that leaves the vertical faces between treads in place. See "What the
  policies actually learned". This is a result to publish rather than a gap to apologise for, but it
  does bound every claim about what was learned.
- **The transfer test evaluates final policies only**, 8 episodes each, no fine tuning and no
  retraining on the new flights. It answers "does this policy work there", not "is this exploit
  learnable there". The second question is a training run per flight and was not done.
- **The negative control did not run.** `-DVERSION_SH` does not compile in libsm64.
- **The film strip's terminal checkpoints** came from the v1 run directory. The v1 and v2 terminal
  policies are bit identical because that rung's reward never changed, verified weight by weight, so
  the data is correct, but the provenance takes a sentence to explain.
- **No real game replay yet.** Exporting the policy's inputs as a Mupen64 `.m64` and replaying them
  in the actual castle is the obvious next artifact. The pieces exist: sm64-port replays movies on
  the cluster and the recorder already stores per frame controller state. The obstacle is that the
  real game's stick is camera relative while libsm64 derives its camera from a synthetic
  `camLookX/camLookZ`, so a faithful replay wants a closed loop bridge that reads the real camera
  yaw rather than an open loop movie.
