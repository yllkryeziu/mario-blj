"""Asks where the frames go, action by action, before and after the policy learns anything.

The first millions of steps look like learning and are not. A reward that pays only for reaching
the top landing is exactly 0.0 until the first success, so there is no gradient, and a checkpoint
taken in that window should be indistinguishable from the network it started as. Saying so needs a
measurement rather than a curve, because the curve is flat either way.

The measurement is occupancy: of N frames, how many did Mario spend inside each action. It
separates the two hypotheses cleanly. Uniform random and an untrained network give the occupancy
of the action space itself, which is the null. A checkpoint from before the first success either
matches that null, in which case the flat curve means what it says, or it does not, in which case
something was learned that the reward never asked for.

Ground pound is the action to watch. Eighteen of the thirty six actions hold Z and eighteen press
A, and A pressed while airborne with Z held is the ground pound trigger, so the action space
itself steers Mario into it. It is also cheap to enter and slow to leave, ``ground_pound`` into
``ground_pound_land``, so its share of frames runs well above its share of actions. Long jump is
the action to watch on the other side, because a working chain re-enters it on nearly every frame.

Every rollout runs to a frame budget rather than to an episode boundary, resetting as needed,
because a solving policy finishes an episode in about 120 frames and a random one never finishes
at all. Four rollouts per policy, each with its own sampling seed, is what turns a single number
into a range.
"""

import argparse
import collections
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
_COLLISION_PARTS = ("third_party", "sm64-port", "levels", "castle_inside", "areas", "2",
                    "collision.inc.c")

FRAMES = 1500
ROLLOUTS = 4
SEED_BASE = 20250915

# Both halves of the ground pound, because the landing is what makes it expensive: entering costs
# one frame and leaving costs several, so counting only ``ground_pound`` would understate the
# attractor by roughly the length of its own recovery.
GROUND_POUND = ("ground_pound", "ground_pound_land")
LONG_JUMP = ("long_jump",)

# Standing still long enough puts Mario to sleep, and libsm64's ``act_waking_up`` calls a
# ``stop_sound`` that never returns. Stochastic sampling from a 36 way distribution does not hold
# one no op for 1,500 frames, so this is a guard rather than a workaround, but it is cheap.
SLEEP_ACTIONS = frozenset((0x0C400202, 0x0C000203, 0x0C000204))


def rollout(env, choose, frames: int, seed: int) -> dict:
    """Runs one policy for a fixed number of frames and counts the actions Mario was in.

    Args:
        env: The environment, already constructed.
        choose: Callable from observation to action index.
        frames: Frame budget. Episodes reset until it is spent.
        seed: Seed for the first reset. Later resets advance it, so a policy cannot be scored on
            one lucky spawn.

    Returns:
        Counts keyed by action name, with the episode and success tallies alongside.
    """
    from src.env.native import action_name

    counts: collections.Counter = collections.Counter()
    observation, _ = env.reset(seed=seed)
    episodes = 0
    successes = 0
    peak = 0.0
    best_height = 0.0
    warps = 0
    for _ in range(frames):
        observation, _, terminated, truncated, info = env.step(int(choose(observation)))
        counts[action_name(info["mario_action_id"])] += 1
        # The environment reports these per episode and cumulatively, so a rollout that spans
        # resets has to take the extremes itself.
        peak = min(peak, float(info["peak_backward_velocity"]))
        best_height = max(best_height, float(info["best_height"]))
        asleep = info["mario_action_id"] in SLEEP_ACTIONS
        if terminated or truncated or asleep:
            episodes += 1
            successes += bool(info["success"])
            warps += int(info["warps"])
            observation, _ = env.reset(seed=seed + 1000 + episodes)
    return {"frames": frames, "seed": seed, "episodes": episodes, "successes": successes,
            "peak_backward_velocity": round(peak, 2), "best_height": round(best_height, 1),
            "warps": warps, "counts": dict(counts)}


def occupancy(counts: dict, names) -> float:
    """Shares the frames spent in a group of actions.

    Args:
        counts: Action name to frame count.
        names: Action names forming the group.

    Returns:
        The group's share of the rollout's frames, as a fraction.
    """
    total = sum(counts.values())
    return sum(counts.get(name, 0) for name in names) / total if total else 0.0


def spread(values: list[float]) -> dict:
    """Reduces the rollouts of one policy to a mean and a range.

    Args:
        values: One value per rollout.

    Returns:
        Mean, minimum and maximum as percentages, rounded to one decimal.
    """
    return {"mean": round(100 * statistics.mean(values), 1),
            "min": round(100 * min(values), 1),
            "max": round(100 * max(values), 1)}


def discover(directory: str) -> list[tuple[str, int, str]]:
    """Finds the checkpoints to measure.

    Args:
        directory: Directory of ``ppo_<steps>_steps.zip`` files.

    Returns:
        Label, step count and path, ordered by step count. Empty if the directory is absent.
    """
    if not os.path.isdir(directory):
        return []
    found = []
    for entry in os.listdir(directory):
        match = re.fullmatch(r"ppo_(\d+)_steps\.zip", entry)
        if match is None:
            continue
        steps = int(match.group(1))
        found.append((f"{steps / 1e6:.1f}M checkpoint", steps, os.path.join(directory, entry)))
    return sorted(found, key=lambda row: row[1])


def main() -> None:
    """Measures every policy and writes the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default=os.path.join(
        ROOT, "data", "checkpoints", "terminal_seed_2"))
    parser.add_argument("--rung", default="terminal")
    parser.add_argument("--rom", default=os.path.join(ROOT, "roms", "baserom.us.z64"))
    parser.add_argument("--library", default=None)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--out", default=os.path.join(RESULTS, "action_occupancy.json"))
    args = parser.parse_args()

    import numpy
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from src.env.blj_env import BljEnv
    from src.train.ladder import build_config, get_rung
    from src.train.ppo import build_model

    config = build_config(get_rung(args.rung), args.rom,
                          os.path.join(ROOT, *_COLLISION_PARTS), library_path=args.library)
    env = BljEnv(config)
    rows = []
    try:
        plan: list[tuple[str, int | None, str | None]] = [
            ("uniform random", None, None),
            ("untrained network", 0, None),
        ]
        plan += [(label, steps, path) for label, steps, path in discover(args.checkpoints)]

        for label, steps, path in plan:
            runs = []
            for index in range(args.rollouts):
                seed = SEED_BASE + index
                if path is not None:
                    model = PPO.load(path, device="cpu")
                    model.set_random_seed(seed)
                    def choose(observation, model=model):
                        action, _ = model.predict(observation, deterministic=False)
                        return action
                elif label == "uniform random":
                    generator = numpy.random.default_rng(seed)
                    def choose(observation, generator=generator):
                        return generator.integers(env.action_space.n)
                else:
                    # A different initialization per rollout, not just different sampling: the
                    # question is what an untrained network does, and the answer should not
                    # depend on one draw of the weights.
                    model = build_model(DummyVecEnv([lambda: env]), seed=seed)
                    model.set_random_seed(seed)
                    def choose(observation, model=model):
                        action, _ = model.predict(observation, deterministic=False)
                        return action
                runs.append(rollout(env, choose, args.frames, seed))
                print(f"  {label:<20} rollout {index}  "
                      f"gp {100 * occupancy(runs[-1]['counts'], GROUND_POUND):5.1f}%  "
                      f"lj {100 * occupancy(runs[-1]['counts'], LONG_JUMP):5.1f}%  "
                      f"peak {runs[-1]['peak_backward_velocity']:9.1f}  "
                      f"{runs[-1]['successes']}/{runs[-1]['episodes']} episodes solved",
                      flush=True)

            totals: collections.Counter = collections.Counter()
            for run in runs:
                totals.update(run["counts"])
            rows.append({
                "policy": label,
                "steps": steps,
                "rollouts": runs,
                "ground_pound": spread([occupancy(run["counts"], GROUND_POUND) for run in runs]),
                "long_jump": spread([occupancy(run["counts"], LONG_JUMP) for run in runs]),
                "episodes": sum(run["episodes"] for run in runs),
                "successes": sum(run["successes"] for run in runs),
                "best_peak": min(run["peak_backward_velocity"] for run in runs),
                "median_peak": round(statistics.median(
                    run["peak_backward_velocity"] for run in runs), 1),
                "best_height": max(run["best_height"] for run in runs),
                "top_actions": [[name, count] for name, count in totals.most_common(8)],
            })
    finally:
        env.close()

    report = {"rung": args.rung, "frames": args.frames, "rollouts": args.rollouts,
              "checkpoints_from": os.path.relpath(args.checkpoints, ROOT),
              "ground_pound_actions": list(GROUND_POUND), "long_jump_actions": list(LONG_JUMP),
              "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\n{'policy':<22}{'ground pound':>22}{'long jump':>22}"
          f"{'median peak':>13}{'solved':>10}")
    for row in rows:
        gp, lj = row["ground_pound"], row["long_jump"]
        cells = (f"{gp['mean']:.1f}% ({gp['min']:.1f}-{gp['max']:.1f})",
                 f"{lj['mean']:.1f}% ({lj['min']:.1f}-{lj['max']:.1f})",
                 f"{row['median_peak']:.1f}",
                 f"{row['successes']}/{row['episodes']}")
        print(f"{row['policy']:<22}{cells[0]:>22}{cells[1]:>22}{cells[2]:>13}{cells[3]:>10}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
