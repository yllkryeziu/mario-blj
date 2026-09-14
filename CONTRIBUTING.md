# Contributing

This repo measures how much reward shaping an RL agent needs before it can perform the Super
Mario 64 backwards long jump. The ablation ladder is the result, so reproducibility matters more
than speed. Everything below exists to keep a run from three weeks ago comparable to a run today.

## Getting set up

```bash
scripts/setup.sh                        # clone and build third_party/libsm64
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                 # runtime deps plus pytest, ruff, pyright
```

libsm64 reads Mario's animations and textures out of a real ROM at runtime. Put a Super Mario 64
US ROM at `roms/baserom.us.z64`. It is gitignored and never committed.

`make` picks up `.venv/bin/python` automatically when that directory exists. Point it somewhere
else with `make test PYTHON=/path/to/python`.

## Make targets

| Target | What it does |
| --- | --- |
| `make setup` | runs `scripts/setup.sh`: clone and build libsm64, then check for the ROM |
| `make test` | `pytest` over `src/`, collecting `*_test.py` |
| `make lint` | `ruff check` over the whole tree except `third_party/` |
| `make typecheck` | `pyright` in basic mode, resolving imports against `$(PYTHON)` |
| `make format` | `ruff format` followed by `ruff check --fix` |
| `make viewer` | packs `results/replay_*.json` and renders `results/viewer.html` |
| `make clean` | deletes caches, build products and the rendered viewer |

## Style

Google Python Style Guide, 100 column lines, enforced by `ruff` with pycodestyle, pyflakes,
isort, pydocstyle on the google convention, pep8-naming, pyupgrade and bugbear enabled.

Four rules carry most of the weight:

- Full type annotations on every signature, including `-> None`.
- Google-style docstrings with `Args`, `Returns` and `Raises` on every module, class and public
  function. The docstring is where the reasoning lives.
- No inline comments. If a line needs explaining, the explanation belongs in the docstring of the
  thing that contains it, or the line needs rewriting.
- Frozen dataclasses for configuration, and no global mutable state.

Prose in docstrings, the README and any report follows the same voice as the rest of the repo:
plain declarative sentences, no em dashes, no filler.

One thing to know before running `make format`: the tree is currently written with continuation
lines aligned under the opening parenthesis, and `ruff format` uses black's one-argument-per-line
style instead. Running it will reformat nearly every file. Do that in a single deliberate commit
of its own rather than mixing it into a change, or skip it and rely on `make lint` alone.

## Tests

Test files are named `*_test.py` and sit next to the code they cover, so `src/env/geometry.py` is
tested by `src/env/geometry_test.py`. `make test` runs the whole suite in about a second.

Two rules keep the suite usable:

- A test that needs a ROM, the built dylib, or the `third_party/sm64-port` checkout is guarded by
  `pytest.mark.skipif` on the path it needs, with a reason naming the missing file. The suite has
  to pass on a fresh clone with nothing built.
- Assertions are on measured behaviour rather than on shapes. The collision tests pin the real
  1923 tangible triangles in castle_inside area 2 and the real 154 unit span of the instant warp
  zone. When a test asserts a physical bound, its docstring says where the bound comes from.

The measurements the tests encode are settled facts about the substrate, not guesses. Before
changing one of those constants, check whether the code changed or the reference did.

## Layout

```
src/env/native.py          ctypes binding to libsm64, one struct per libsm64 type
src/env/geometry.py        synthetic surface generators: planes, ramps, staircases
src/env/collision.py       parser for the decompilation's collision.inc.c files
src/env/endless_stairs.py  the real castle_inside area 2 scene, including the instant warp
src/env/blj_env.py         the Gymnasium environment and its reward configuration
src/agent/scripted.py      the hand-written reference BLJ policy
src/agent/drivers.py       action drivers shared by the scripted and learned agents
src/train/                 PPO training and the ablation ladder
scripts/                   entry points: setup, validation, recording, training
tools/                     replay packing and the standalone HTML viewer
third_party/libsm64        the substrate, verified byte-identical to upstream sm64 for the BLJ
third_party/sm64-port      the decompilation, read for collision and level data
results/                   recorded replays and sweep output
```

## Validating a change

Validate against a known-good reference before sweeping parameters. The scripted policy in
`src/agent/scripted.py` is that reference: it reproduces the runaway on a staircase and it
provably cannot on flat ground. If a change to the environment or the substrate breaks the
scripted policy, the sweep results after it mean nothing.
