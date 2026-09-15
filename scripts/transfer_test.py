"""Asks whether the trained policies learned the chain or memorized one staircase.

Every run in the ladder saw exactly one flight of stairs: the castle's, with 25.6 unit treads
51.25 units deep. A policy that reaches the top landing there has either learned the backwards
long jump chain, which is a property of Mario's physics, or learned a sequence of inputs that
happens to work on that specific tread size, which is a property of one level. The two are
indistinguishable from the training curve and easy to separate with a second staircase.

This script drops each saved policy, unchanged and never fine tuned, onto five flights it never
saw and one it did. The first synthetic flight rebuilds the castle's own tread geometry from
:func:`src.env.endless_stairs.synthetic_scene`, which is the control that separates a transfer
failure from a rebuild artefact: if a policy solves the castle and fails the rebuild, the rebuild
is wrong, not the policy. The others vary rise and run away from it.

Two numbers come back per flight. Success is the task's own self certifying criterion, reaching
the top landing, and it is the number that answers the question asked. Peak backwards velocity is
the number that explains the answer, because it is the same quantity
``results/blj_runaway.json`` reports for the hand written expert on the same geometries, so a
learned policy and a scripted one can be compared directly on flights neither was built for.
"""

import argparse
import collections
import json
import multiprocessing
import os
import queue as _queue
import statistics
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

MAX_FRAMES = 3000
EPISODE_SEED_BASE = 20250915

# Standing still long enough puts Mario to sleep, and libsm64's ``act_waking_up`` calls a
# ``stop_sound`` that never returns, so the next frame spins in C forever. Training never met it
# because a policy sampled from a 36 way distribution with entropy pressure does not hold one
# no op for 1,500 consecutive frames; a deterministic rollout of a policy that learned nothing
# does exactly that. Ending the episode as soon as he falls asleep is both the workaround and the
# honest reading: an episode in which Mario has gone to sleep is over.
SLEEP_ACTIONS = frozenset((0x0C400202, 0x0C000203, 0x0C000204))

SCENES: dict[str, tuple[float, float, bool] | None] = {
    "castle": None,
    "rebuilt_rise26_run51": (25.6, 51.25, True),
    "rise50_run51": (50.0, 51.25, True),
    "rise26_run100": (25.6, 100.0, True),
    "rise75_run100": (75.0, 100.0, True),
    "rise100_run100": (100.0, 100.0, True),
    "rise100_run100_norisers": (100.0, 100.0, False),
    "rise75_run100_norisers": (75.0, 100.0, False),
    "rise50_run51_norisers": (50.0, 51.25, False),
    "rebuilt_rise26_run51_norisers": (25.6, 51.25, False),
}


def scene_for(label: str):
    """Builds one scene by label.

    Args:
        label: A key of :data:`SCENES`.

    Returns:
        None for the castle, which the environment loads from the decompilation itself, or a
        synthetic scene for every other label.

    Raises:
        KeyError: If the label is not in :data:`SCENES`.
    """
    from src.env.endless_stairs import synthetic_scene

    treads = SCENES[label]
    if treads is None:
        return None
    rise, run, risers = treads
    return synthetic_scene(rise, run, risers=risers)


def run_episodes(task: tuple[str, str, int, str, str, str | None, int]) -> dict:
    """Evaluates one saved policy on one scene, in its own process.

    libsm64 keeps its state in C statics and terminates process wide, so a worker builds exactly
    one environment and one library instance, then loads every policy it was given against it.
    That is also why the scene is fixed per task rather than per episode.

    Args:
        task: Scene label, rung name, seed, model path, ROM path, library path and episode count.

    Returns:
        A row per episode under ``episodes``, with the run's identity alongside.
    """
    label, rung_name, seed, model_path, rom_path, library_path, episodes = task

    from stable_baselines3 import PPO

    from src.env.blj_env import BljConfig, BljEnv
    from src.train.ladder import build_config, get_rung

    rung = get_rung(rung_name)
    config = build_config(rung, rom_path, os.path.join(ROOT, *_COLLISION_PARTS),
                          max_frames=MAX_FRAMES, library_path=library_path)
    config = BljConfig(**{**config.__dict__, "scene": scene_for(label)})

    env = BljEnv(config)
    rows = []
    try:
        model = PPO.load(model_path, device="cpu")
        for episode in range(episodes):
            deterministic = episode == 0
            model.set_random_seed(EPISODE_SEED_BASE + episode)
            observation, _ = env.reset(seed=EPISODE_SEED_BASE + episode)
            info: dict = {}
            while True:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated or info["mario_action_id"] in SLEEP_ACTIONS:
                    break
            rows.append({
                "episode": episode,
                "deterministic": deterministic,
                "success": bool(info["success"]),
                "peak_backward_velocity": round(float(info["peak_backward_velocity"]), 2),
                "best_height": round(float(info["best_height"]), 1),
                "warps": int(info["warps"]),
                "curriculum_stage": int(info["curriculum_stage"]),
                "frames": int(info["frames"]),
                "asleep": info["mario_action_id"] in SLEEP_ACTIONS,
            })
    finally:
        env.close()

    return {"scene": label, "rung": rung_name, "seed": seed, "episodes": rows}


_COLLISION_PARTS = ("third_party", "sm64-port", "levels", "castle_inside", "areas", "2",
                    "collision.inc.c")


def discover(models_dir: str, rungs: list[str] | None,
             seeds: list[int] | None) -> list[tuple[str, int, str]]:
    """Finds the saved policies to evaluate.

    Args:
        models_dir: Directory laid out as ``<rung>/seed_<n>/model.zip``.
        rungs: Rung names to keep, or None for all of them.
        seeds: Seeds to keep, or None for all of them.

    Returns:
        A sorted list of rung name, seed and model path.

    Raises:
        FileNotFoundError: If no model is found.
    """
    found = []
    for rung_name in sorted(os.listdir(models_dir)):
        if rungs and rung_name not in rungs:
            continue
        rung_dir = os.path.join(models_dir, rung_name)
        if not os.path.isdir(rung_dir):
            continue
        for entry in sorted(os.listdir(rung_dir)):
            if not entry.startswith("seed_"):
                continue
            seed = int(entry.removeprefix("seed_"))
            if seeds is not None and seed not in seeds:
                continue
            path = os.path.join(rung_dir, entry, "model.zip")
            if os.path.exists(path):
                found.append((rung_name, seed, path))
    if not found:
        raise FileNotFoundError(f"no <rung>/seed_<n>/model.zip under {models_dir}")
    return found


def summarize(rows: list[dict]) -> dict:
    """Reduces one run's episodes on one scene to the row that goes in the report.

    Args:
        rows: Episode rows from :func:`run_episodes`.

    Returns:
        Success count and rate, the deterministic episode's outcome on its own, and the peak
        backwards velocity as both the best and the median over episodes. The median is reported
        because one lucky rollout is not transfer.
    """
    peaks = [row["peak_backward_velocity"] for row in rows]
    deterministic = next((row for row in rows if row["deterministic"]), None)
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": round(sum(row["success"] for row in rows) / len(rows), 3),
        "deterministic_success": None if deterministic is None else deterministic["success"],
        "best_peak": min(peaks),
        "median_peak": round(statistics.median(peaks), 2),
        "max_stage": max(row["curriculum_stage"] for row in rows),
        "mean_warps": round(statistics.mean(row["warps"] for row in rows), 1),
        "best_height": max(row["best_height"] for row in rows),
        "asleep": sum(row["asleep"] for row in rows),
    }


def read_journal(path: str) -> list[dict]:
    """Reads the rows an earlier invocation already finished.

    Args:
        path: Path to the journal, which need not exist.

    Returns:
        The rows, or an empty list.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _worker(queue, task: tuple) -> None:
    """Runs one task and puts its outcome on the queue.

    Module level rather than a closure because the spawn start method pickles the target, and
    spawn is required here for the same reason training uses it: libsm64 keeps process wide state.

    Args:
        queue: Queue to report on.
        task: Task tuple for :func:`run_episodes`.
    """
    try:
        queue.put(("ok", run_episodes(task)))
    except BaseException as error:                      # noqa: BLE001 - reported, not swallowed
        queue.put(("error", {"scene": task[0], "rung": task[1], "seed": task[2],
                             "error": f"{type(error).__name__}: {error}"}))


def run_all(tasks: list[tuple], jobs: int, timeout: float):
    """Runs the tasks in parallel, yielding each result as it lands.

    One process per task, rather than a pool, because a task is a whole libsm64 instance and the
    library can take the process down with it. A pool that loses a worker to a signal deadlocks
    and every completed result dies with it, which is what happened on the first full run: two
    tasks never returned and the other 142 were lost with them. Here a task that dies or overruns
    is reported by name and the rest of the matrix still finishes.

    Args:
        tasks: Task tuples for :func:`run_episodes`.
        jobs: How many to run at once.
        timeout: Seconds a single task may take before it is killed.

    Yields:
        One result dict per task that completed.
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    pending = list(tasks)
    running: list[tuple] = []

    def drain():
        while True:
            try:
                kind, payload = queue.get_nowait()
            except _queue.Empty:
                return
            if kind == "ok":
                yield payload
            else:
                print(f"  task failed: {payload}")

    while pending or running:
        while pending and len(running) < jobs:
            task = pending.pop(0)
            process = context.Process(target=_worker, args=(queue, task), daemon=True)
            process.start()
            running.append((process, task, time.monotonic()))
        yield from drain()
        for entry in list(running):
            process, task, started = entry
            process.join(timeout=0.2)
            if process.exitcode is None and time.monotonic() - started > timeout:
                print(f"  task timed out after {timeout:.0f}s: {task[0]} {task[1]} seed {task[2]}")
                process.kill()
                process.join()
            if process.exitcode is not None:
                running.remove(entry)
                if process.exitcode != 0:
                    print(f"  worker for {task[0]} {task[1]} seed {task[2]} "
                          f"exited with {process.exitcode}")
    yield from drain()


def main() -> None:
    """Runs every policy on every scene and writes the report.

    Raises:
        FileNotFoundError: If the models directory holds nothing to evaluate.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default=os.path.join(ROOT, "data", "models_v2"))
    parser.add_argument("--rom", default=os.path.join(ROOT, "roms", "baserom.us.z64"))
    parser.add_argument("--library", default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument("--rungs", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", default=os.path.join(RESULTS, "transfer.json"))
    parser.add_argument("--journal", default=os.path.join(RESULTS, "transfer_journal.jsonl"))
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    runs = discover(args.models_dir, args.rungs, args.seeds)
    results = read_journal(args.journal)
    finished = {(row["scene"], row["rung"], row["seed"]) for row in results}
    tasks = [(label, rung_name, seed, path, args.rom, args.library, args.episodes)
             for label in args.scenes
             for rung_name, seed, path in runs
             if (label, rung_name, seed) not in finished]
    print(f"{len(runs)} policies on {len(args.scenes)} scenes, "
          f"{args.episodes} episodes each: {len(tasks)} tasks on {args.jobs} workers"
          f"{f', {len(finished)} already in the journal' if finished else ''}")

    for done, result in enumerate(run_all(tasks, args.jobs, args.timeout), start=1):
        row = summarize(result["episodes"])
        result = {**result, "summary": row}
        results.append(result)
        with open(args.journal, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        print(f"[{done:3d}/{len(tasks)}] {result['scene']:>26}  "
              f"{result['rung']:>12} seed {result['seed']}  "
              f"{row['successes']}/{row['episodes']} success  "
              f"peak {row['best_peak']:9.1f}  median {row['median_peak']:9.1f}")
    results = [row for row in results
               if row["scene"] in args.scenes
               and (not args.rungs or row["rung"] in args.rungs)]

    from src.env.endless_stairs import minimum_escape_speed

    scenes = {}
    for label in args.scenes:
        scene = scene_for(label)
        if scene is None:
            from src.env.endless_stairs import load_scene
            scene = load_scene(os.path.join(ROOT, *_COLLISION_PARTS))
        rise, run, risers = SCENES[label] or (25.6, 51.25, True)
        scenes[label] = {
            "rise": rise, "run": run, "risers": risers,
            "synthetic": SCENES[label] is not None,
            "goal_y": scene.goal_y, "goal_z": scene.goal_z,
            "escape_speed": minimum_escape_speed(scene.warp),
            "triangles": len(scene.surfaces),
        }

    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for result in results:
        by_scene[result["scene"]].append(result)
    print()
    for label in args.scenes:
        rows = by_scene[label]
        solving = [r for r in rows if r["summary"]["successes"]]
        best = min((r["summary"]["best_peak"] for r in rows), default=0.0)
        print(f"{label:>21}  {len(solving):2d} of {len(rows)} policies reach the landing, "
              f"best peak {best:9.1f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"episodes_per_run": args.episodes, "max_frames": MAX_FRAMES,
                   "scenes": scenes,
                   "runs": sorted(results, key=lambda r: (r["scene"], r["rung"], r["seed"]))},
                  handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
