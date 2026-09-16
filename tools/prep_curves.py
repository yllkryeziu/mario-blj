"""Reduces the raw ladder logs into the payload the learning curve page reads.

The page compares reward shaping regimes, so the unit is a rung, and the thing that matters inside
a rung is how much the seeds disagree. Every seed is therefore kept as its own trace and the
median is taken per bin across whichever seeds reached that bin, which lets a run that stopped
early end where it stopped instead of dragging the median down.
"""

from __future__ import annotations

import json
import os
import statistics

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHOWN = ("terminal", "speed", "height", "height_speed")

REWARD_SHOWN = ("terminal", "speed", "height", "height_speed")

SERIES = {
    "terminal": ("#2a78d6", "#3987e5"),
    "speed": ("#eb6834", "#d95926"),
    "height": ("#1baf7a", "#199e70"),
    "height_speed": ("#eda100", "#c98500"),
}

LABELS = {
    "terminal": ("Terminal only", "1.0 for standing on the top landing, nothing else"),
    "speed": ("Speed shaping", "plus a record backwards speed term"),
    "height": ("Height shaping", "plus a record climbed height term"),
    "height_speed": ("Height and speed", "both shaping terms"),
}


def settled(run: dict) -> list[dict]:
    """Returns a run's bins excluding the final one, which is still filling.

    These runs are read while they are in flight, so the newest bin holds only the episodes that
    happened to finish before the log was copied. On a seed that has learned the exploit, most of
    those are still in progress and score zero, which draws a cliff to the floor at the right edge
    of every curve and reads as a collapse that did not happen.

    Args:
        run: One run record.

    Returns:
        Every bin except the last.
    """
    return run["points"][:-1]


def median_curve(seeds: list[dict], key: str) -> list[list[float]]:
    """Returns the per bin median of one series across seeds.

    Args:
        seeds: Per seed run records.
        key: Point field to take the median of.

    Returns:
        A list of [steps, median] pairs, covering only the bins every seed reached.

    Bins past the slowest seed's frontier are dropped. Seeds that solve the task finish episodes
    in about 150 frames instead of running to the 3000 frame ceiling, so they spend far more time
    resetting and advance through timesteps more slowly in wall clock. Reading a partially filled
    frontier therefore takes the median over whichever seeds happen to be ahead, which are the
    ones that never learned, and draws a cliff to zero that no seed experienced.
    """
    buckets: dict[int, list[float]] = {}
    for seed in seeds:
        for point in settled(seed):
            buckets.setdefault(point["steps"], []).append(point[key])
    full = len(seeds)
    return [[steps, round(statistics.median(values), 4)]
            for steps, values in sorted(buckets.items()) if len(values) == full]


def return_modes(runs: list[dict]) -> dict:
    """Splits a rung's final return into the two values its seeds actually take.

    Return on this task is bimodal rather than continuous: a seed either has the exploit, in which
    case it scores 1.0 for the landing plus whatever its shaping terms are worth, or it does not,
    in which case it scores the shaping alone. There is no middle. A median across six seeds
    therefore reports which side of three the population fell on and not how well anybody is
    doing, and its height is set by how much shaping the rung adds rather than by how many seeds
    solved. Both halves are kept so a figure can say which of the two it is drawing.

    Args:
        runs: Per seed run records for one rung.

    Returns:
        The median final return of the solving seeds and of the seeds that never solved, each with
        its count and its range, and None where that half of the split is empty. The median rather
        than the mean, to match the curves the figure draws, and the range alongside it because
        one height seed collapses to 0.005 and a single number would hide it.
    """
    split = {}
    for label, predicate in (("solved", lambda r: r["first_success"] is not None),
                             ("never", lambda r: r["first_success"] is None)):
        finals = [settled(r)[-1]["ret"] for r in runs if predicate(r) and settled(r)]
        split[label] = {
            "seeds": len(finals),
            "return": round(statistics.median(finals), 4) if finals else None,
            "range": [round(min(finals), 4), round(max(finals), 4)] if finals else None,
        }
    return split


def dominated_bins(above: list[list[float]], below: list[list[float]]) -> dict:
    """Counts the bins on which one median curve sits above another.

    This exists for one comparison, and it is the point of the figure. Height shaping never once
    reaches the landing and the landing-only reward does, yet height's median return is the higher
    of the two, because it is paid for climbing and the other is paid for nothing short of the
    goal. Quoting the count rather than eyeballing the chart keeps the prose honest if either
    curve is ever recomputed.

    Args:
        above: The curve expected to be higher, as [steps, value] pairs.
        below: The curve to compare it against.

    Returns:
        The number of shared bins, how many of them the first curve leads on, and the first and
        last shared bin with both values.
    """
    first, second = dict(above), dict(below)
    shared = sorted(set(first) & set(second))
    leads = [steps for steps in shared if first[steps] > second[steps]]
    return {
        "bins": len(shared),
        "leadingBins": len(leads),
        "firstBin": {"steps": shared[0], "above": first[shared[0]], "below": second[shared[0]]},
        "lastBin": {"steps": shared[-1], "above": first[shared[-1]], "below": second[shared[-1]]},
    }


def main() -> None:
    """Writes results/curves_page.json and prints one line per rung."""
    with open(os.path.join(_ROOT, "results", "learning_curves.json"), encoding="utf-8") as handle:
        raw = json.load(handle)

    rungs = []
    for name in SHOWN:
        runs = sorted((r for r in raw["runs"] if r["rung"] == name), key=lambda r: r["seed"])
        title, subtitle = LABELS[name]
        rungs.append({
            "name": name,
            "title": title,
            "subtitle": subtitle,
            "seeds": [{
                "seed": r["seed"],
                "first_success": r["first_success"],
                "successes": r["successes"],
                "episodes": r["episodes"],
                "best_peak": r["best_peak"],
                "budget": settled(r)[-1]["steps"] if settled(r) else 0,
                "rate": [[p["steps"], p["rate"]] for p in settled(r)],
                "length": [[p["steps"], p["length"]] for p in settled(r)],
                "ret": [[p["steps"], p["ret"]] for p in settled(r)],
            } for r in runs],
            "median_rate": median_curve(runs, "rate"),
            "median_length": median_curve(runs, "length"),
            "median_return": median_curve(runs, "ret"),
            "max_return": round(max((p["ret"] for r in runs for p in r["points"]), default=1.0), 3),
            "return_modes": return_modes(runs),
            "light": SERIES[name][0],
            "dark": SERIES[name][1],
            "reward_shown": name in REWARD_SHOWN,
            "solved": sum(1 for r in runs if r["first_success"] is not None),
            "total": len(runs),
        })

    by_name = {rung["name"]: rung for rung in rungs}
    payload = {
        "bin": raw["bin"],
        "rungs": rungs,
        "height_over_terminal": dominated_bins(by_name["height"]["median_return"],
                                               by_name["terminal"]["median_return"]),
    }
    out = os.path.join(_ROOT, "results", "curves_page.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    for rung in rungs:
        firsts = [s["first_success"] for s in rung["seeds"]]
        named = [f"{value / 1e6:.2f}M" for value in firsts if value]
        print(f"{rung['title']:<20} {rung['solved']}/{rung['total']} solved   "
              f"budget {max(s['budget'] for s in rung['seeds']) / 1e6:.1f}M   "
              f"first success {', '.join(named) if named else 'never'}")
    lead = payload["height_over_terminal"]
    print(f"\nheight median return leads terminal on "
          f"{lead['leadingBins']}/{lead['bins']} bins, "
          f"{lead['lastBin']['above']} against {lead['lastBin']['below']} at the last")
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
