"""Reduces the ladder's episode logs to the small table the write up plots.

Training wrote one row per episode, 975,740 of them across 24 runs and 53 MB of CSV, which is the
right granularity to keep and the wrong granularity to publish. ``results/learning_curves.json``
already carries the 250,000 step binning the curves use; what it cannot answer is anything at
episode resolution, and three of the claims worth making live there.

The first is the discovery moment. A run that spends six million frames at exactly zero reward and
then never stops succeeding has a single episode where that changes, and the interesting question
is how many episodes it took to go from the first success to a reliable one. Step bins average
that away.

The second is the shape of the return distribution. Under the terminal rung the return is exactly
zero or exactly one, so the distribution is two spikes and the mean is a success rate wearing a
disguise; under the shaped rungs it is a continuum whose bulk sits well below one, which is what
makes a mean return look like progress when nothing has been solved. Both are histograms over
episodes, not over bins.

The third is episode length, which is the cheapest available detector of what a run is actually
doing. An episode ends at the frame budget, or when Mario dies, or when he reaches the landing, so
a run whose lengths are all exactly the budget has never finished anything, and a bimodal run is
one that sometimes does.
"""

import argparse
import collections
import csv
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BIN = 500
RETURN_EDGES = [0.0, 0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 0.999, 1.0, 1.5]
LENGTH_EDGES = [0, 60, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 2400, 2999, 3000]
FIRST_SUCCESSES = 64
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def read_run(path: str) -> dict[str, list]:
    """Reads one ``episodes.csv`` into columns.

    Args:
        path: Path to the log.

    Returns:
        A mapping from column name to a list of parsed values.

    Raises:
        OSError: If the file cannot be read.
        KeyError: If the log is missing a column this reducer needs.
    """
    columns: dict[str, list] = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            columns["episode"].append(int(row["episode"]))
            columns["timesteps"].append(int(row["timesteps"]))
            columns["frames"].append(int(row["frames"]))
            columns["return"].append(float(row["return"]))
            columns["length"].append(int(row["length"]))
            columns["success"].append(int(row["success"]))
            columns["peak"].append(float(row["peak_backward_velocity"]))
            columns["warps"].append(int(row["warps"]))
            columns["stage"].append(int(row["curriculum_stage"]))
    return columns


def clip(columns: dict[str, list], limit: int) -> dict[str, list]:
    """Drops the episodes that finished past the step budget.

    Stable Baselines ends a rollout on a multiple of ``n_steps`` times the environment count, not
    on the requested total, so a 20,000,000 step run logs a few episodes beyond it.
    ``results/learning_curves.json`` bins only up to the budget, so this reducer has to clip the
    same way or the two files disagree about how many episodes a run had. On the one solving
    terminal seed that is 38 episodes, all of them successes, which would otherwise show up as an
    unexplained discrepancy in the headline number.

    Args:
        columns: Columns from :func:`read_run`.
        limit: Step budget to clip to, or zero to keep everything.

    Returns:
        The columns, truncated to the episodes at or below the budget.
    """
    if not limit:
        return columns
    keep = sum(1 for value in columns["timesteps"] if value <= limit)
    return {name: values[:keep] for name, values in columns.items()}


def histogram(values, edges: list[float]) -> list[int]:
    """Counts values into half open bins.

    The last edge is inclusive on both sides so that a value sitting exactly on it, which is what
    a truncated episode's length and a solved episode's return both do, lands in a bin rather than
    being dropped.

    Args:
        values: Values to count.
        edges: Bin edges, ascending, with one more edge than bins.

    Returns:
        One count per bin.
    """
    counts = [0] * (len(edges) - 1)
    for value in values:
        for index in range(len(counts)):
            low, high = edges[index], edges[index + 1]
            if low <= value < high or (index == len(counts) - 1 and value == high):
                counts[index] += 1
                break
    return counts


def quantiles(values) -> dict[str, float]:
    """Returns the quantiles in :data:`QUANTILES`, computed without numpy."""
    ordered = sorted(values)
    return {f"q{int(q * 100):02d}": round(ordered[min(len(ordered) - 1,
                                                      int(q * (len(ordered) - 1)))], 4)
            for q in QUANTILES}


def reduce_run(rung: str, seed: int, columns: dict[str, list]) -> dict:
    """Reduces one run to its published row.

    Args:
        rung: Rung name.
        seed: Run seed.
        columns: Columns from :func:`read_run`.

    Returns:
        The run's counts, its discovery moment, a rolling success rate at :data:`BIN` episode
        resolution and the two distributions.
    """
    successes = columns["success"]
    episodes = len(successes)
    hits = [index for index, value in enumerate(successes) if value]
    first = hits[0] if hits else None

    rate, steps = [], []
    for start in range(0, episodes - episodes % BIN, BIN):
        window = successes[start:start + BIN]
        rate.append(round(sum(window) / len(window), 4))
        steps.append(columns["timesteps"][start + len(window) - 1])

    tail = successes[-1000:]
    return {
        "rung": rung,
        "seed": seed,
        "episodes": episodes,
        "successes": sum(successes),
        "success_rate": round(sum(successes) / episodes, 5),
        "final_rate_1000": round(sum(tail) / len(tail), 4),
        "first_success": None if first is None else {
            "episode": columns["episode"][first],
            "timesteps": columns["timesteps"][first],
            "frames": columns["frames"][first],
            "zero_return_episodes": sum(1 for value in columns["return"][:first] if value == 0.0),
        },
        "first_success_episodes": [columns["episode"][index] for index in hits[:FIRST_SUCCESSES]],
        "bin_rate": rate,
        "bin_timesteps": steps,
        "return_quantiles": quantiles(columns["return"]),
        "length_quantiles": quantiles(columns["length"]),
        "return_histogram": histogram(columns["return"], RETURN_EDGES),
        "length_histogram": histogram(columns["length"], LENGTH_EDGES),
        "best_peak": round(min(columns["peak"]), 2),
        "mean_warps_last_1000": round(statistics.mean(columns["warps"][-1000:]), 3),
        "mean_stage_last_1000": round(statistics.mean(columns["stage"][-1000:]), 3),
    }


def _first_label(first: dict | None) -> str:
    """Returns a short description of where a run first succeeded, for the console table.

    Args:
        first: The run's first success record, or None if it never succeeded.

    Returns:
        The episode and frame count, or "never".
    """
    if first is None:
        return "never"
    return f"episode {first['episode']}, {first['frames']} frames"


def main() -> None:
    """Reduces every run under the logs directory and writes the report.

    Raises:
        FileNotFoundError: If the logs directory holds no ``episodes.csv``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default=os.path.join(ROOT, "data", "ladder_v2"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "episode_stats.json"))
    parser.add_argument("--max_timesteps", type=int, default=20_000_000)
    args = parser.parse_args()

    runs = []
    for rung in sorted(os.listdir(args.logs)):
        rung_dir = os.path.join(args.logs, rung)
        if not os.path.isdir(rung_dir):
            continue
        for entry in sorted(os.listdir(rung_dir), key=lambda name: int(name.removeprefix("seed_"))):
            path = os.path.join(rung_dir, entry, "episodes.csv")
            if not os.path.exists(path):
                continue
            seed = int(entry.removeprefix("seed_"))
            columns = clip(read_run(path), args.max_timesteps)
            row = reduce_run(rung, seed, columns)
            runs.append(row)
            first = row["first_success"]
            print(f"{rung:>12} seed {seed}  {row['episodes']:7d} episodes  "
                  f"{row['successes']:7d} successes  "
                  f"first at {_first_label(first):>34}"
                  f"  final rate {row['final_rate_1000']:.3f}")
    if not runs:
        raise FileNotFoundError(f"no episodes.csv under {args.logs}")

    report = {
        "bin": BIN,
        "max_timesteps": args.max_timesteps,
        "return_edges": RETURN_EDGES,
        "length_edges": LENGTH_EDGES,
        "total_episodes": sum(row["episodes"] for row in runs),
        "total_successes": sum(row["successes"] for row in runs),
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, separators=(",", ":"))
    print(f"\n{report['total_episodes']} episodes, {report['total_successes']} successes")
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
