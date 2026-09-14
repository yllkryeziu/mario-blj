"""Tests for the collision.inc.c parser in :mod:`src.env.collision`.

Two kinds of test live here. The synthetic ones build a small collision file in a temporary
directory and pin down the parser's contract: which surface types it drops, how it honours the
triangle count on COL_TRI_INIT, and what it does with an out-of-range vertex index. The rest
parse the real castle_inside area 2 file, which is the endless staircase the whole project is
aimed at, and are skipped when third_party is not checked out.
"""

import itertools
import os

import pytest

from src.env.collision import bounds, parse_collision, surface_constants
from src.env.native import Surface

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SM64_PORT = os.path.join(REPO_ROOT, "third_party", "sm64-port")
SURFACE_HEADER = os.path.join(SM64_PORT, "include", "surface_terrains.h")
ENDLESS_STAIRS_COLLISION = os.path.join(
    SM64_PORT, "levels", "castle_inside", "areas", "2", "collision.inc.c")

SURFACE_INSTANT_WARP_1B = 0x1B
ENDLESS_STAIRS_TRIANGLES = 1923
DECLARED_TRIANGLES = 2019
NO_CAM_COLLISION_TRIANGLES = 96
WARP_TRIANGLES = 12
WARP_BOUNDS = ((-409, 0), (3917, 3994), (905, 1059))
CORRIDOR_TREADS = 152
CORRIDOR_TREAD_LEVELS = 72

needs_sm64_port = pytest.mark.skipif(
    not os.path.exists(ENDLESS_STAIRS_COLLISION),
    reason="third_party/sm64-port is not checked out; run scripts/setup.sh")

SYNTHETIC_COLLISION = """
const Collision test_collision[] = {
    COL_INIT(),
    COL_VERTEX_INIT(0x6),
    COL_VERTEX(0, 0, 0),
    COL_VERTEX(100, 0, 0),
    COL_VERTEX(100, 0, 100),
    COL_VERTEX(0, 0, 100),
    COL_VERTEX(-50, 25, -50),
    COL_VERTEX(-50, 25, 50),
    COL_TRI_INIT(SURFACE_DEFAULT, 2),
    COL_TRI(0, 1, 2),
    COL_TRI(0, 2, 3),
    COL_TRI_INIT(SURFACE_CAMERA_BOUNDARY, 1),
    COL_TRI(0, 1, 4),
    COL_TRI_INIT(SURFACE_INSTANT_WARP_1B, 1),
    COL_TRI(4, 5, 0),
    COL_TRI_STOP(),
    COL_END(),
};
"""

SYNTHETIC_CONSTANTS = {
    "SURFACE_DEFAULT": 0x00,
    "SURFACE_CAMERA_BOUNDARY": 0x72,
    "SURFACE_INSTANT_WARP_1B": SURFACE_INSTANT_WARP_1B,
}


def _vertices(surface: Surface) -> list[tuple[int, int, int]]:
    """Copies a surface's three integer vertices out of the ctypes array.

    Args:
        surface: A parsed surface.

    Returns:
        The three vertices as plain integer triples, in the order the file declared them.
    """
    return [(int(surface.vertices[slot][0]), int(surface.vertices[slot][1]),
             int(surface.vertices[slot][2])) for slot in range(3)]


@pytest.fixture
def synthetic_collision(tmp_path) -> str:
    """Writes the synthetic collision file to disk.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The path to the written file.
    """
    path = tmp_path / "collision.inc.c"
    path.write_text(SYNTHETIC_COLLISION, encoding="utf-8")
    return str(path)


def test_parse_collision_reads_vertices_in_declaration_order(synthetic_collision: str):
    """Triangle vertex indices resolve against the COL_VERTEX list in file order."""
    surfaces = parse_collision(synthetic_collision, SYNTHETIC_CONSTANTS)

    assert _vertices(surfaces[0]) == [(0, 0, 0), (100, 0, 0), (100, 0, 100)]
    assert _vertices(surfaces[1]) == [(0, 0, 0), (100, 0, 100), (0, 0, 100)]


def test_parse_collision_drops_camera_only_surfaces(synthetic_collision: str):
    """Camera boundary and no-cam-collision blocks are intangible, so they are dropped."""
    surfaces = parse_collision(synthetic_collision, SYNTHETIC_CONSTANTS)

    assert len(surfaces) == 3
    assert SYNTHETIC_CONSTANTS["SURFACE_CAMERA_BOUNDARY"] not in {s.type for s in surfaces}
    assert sorted({s.type for s in surfaces}) == [0x00, SURFACE_INSTANT_WARP_1B]


def test_parse_collision_honours_the_per_block_triangle_count(synthetic_collision: str):
    """A COL_TRI_INIT block claims exactly its declared number of triangles, no more."""
    surfaces = parse_collision(synthetic_collision, SYNTHETIC_CONSTANTS)

    default = [s for s in surfaces if s.type == 0x00]
    warps = [s for s in surfaces if s.type == SURFACE_INSTANT_WARP_1B]
    assert len(default) == 2
    assert len(warps) == 1
    assert _vertices(warps[0]) == [(-50, 25, -50), (-50, 25, 50), (0, 0, 0)]


def test_parse_collision_skips_unknown_surface_names(synthetic_collision: str):
    """A surface name missing from the constants table is skipped rather than guessed at."""
    partial = {"SURFACE_DEFAULT": 0x00}
    surfaces = parse_collision(synthetic_collision, partial)

    assert len(surfaces) == 2
    assert {s.type for s in surfaces} == {0x00}


def test_parse_collision_skips_out_of_range_vertex_indices(tmp_path):
    """A triangle referencing a vertex past the end of the list is dropped, not clamped."""
    path = tmp_path / "collision.inc.c"
    path.write_text(
        "COL_VERTEX(0, 0, 0)\nCOL_VERTEX(1, 0, 0)\n"
        "COL_TRI_INIT(SURFACE_DEFAULT, 2)\nCOL_TRI(0, 1, 9)\nCOL_TRI(0, 1, 1)\n",
        encoding="utf-8")

    surfaces = parse_collision(str(path), {"SURFACE_DEFAULT": 0x00})

    assert len(surfaces) == 1
    assert _vertices(surfaces[0]) == [(0, 0, 0), (1, 0, 0), (1, 0, 0)]


def test_parse_collision_propagates_the_terrain_argument(synthetic_collision: str):
    """The terrain code is uniform across a parsed area and defaults to zero."""
    assert all(s.terrain == 0 for s in parse_collision(synthetic_collision, SYNTHETIC_CONSTANTS))

    stone = parse_collision(synthetic_collision, SYNTHETIC_CONSTANTS, terrain=1)
    assert all(s.terrain == 1 for s in stone)
    assert all(s.force == 0 for s in stone)


def test_bounds_returns_inclusive_extents_per_axis():
    """Extents from bounds enclose every vertex of every surface handed to it."""
    surface = Surface()
    for slot, vertex in enumerate([(-409, 3917, 905), (0, 3994, 1059), (-200, 3950, 1000)]):
        for axis in range(3):
            surface.vertices[slot][axis] = vertex[axis]

    assert bounds([surface]) == ((-409, 0), (3917, 3994), (905, 1059))


def test_bounds_spans_the_union_of_several_surfaces(synthetic_collision: str):
    """Extents come from the union of all supplied surfaces."""
    surfaces = parse_collision(synthetic_collision, SYNTHETIC_CONSTANTS)

    assert bounds(surfaces) == ((-50, 100), (0, 25), (-50, 100))


@needs_sm64_port
def test_surface_constants_reads_the_instant_warp_code():
    """SURFACE_INSTANT_WARP_1B is 0x1b, the type that drives the endless stairs loop."""
    constants = surface_constants(SURFACE_HEADER)

    assert constants["SURFACE_INSTANT_WARP_1B"] == SURFACE_INSTANT_WARP_1B
    assert constants["SURFACE_DEFAULT"] == 0x00
    assert constants["SURFACE_CAMERA_BOUNDARY"] == 0x72
    assert constants["SURFACE_NO_CAM_COLLISION"] == 0x76
    assert len(constants) > 100


@needs_sm64_port
def test_endless_stairs_area_yields_1923_tangible_triangles():
    """castle_inside area 2 declares 2019 triangles and 96 of them are camera-only."""
    constants = surface_constants(SURFACE_HEADER)
    surfaces = parse_collision(ENDLESS_STAIRS_COLLISION, constants)

    assert len(surfaces) == ENDLESS_STAIRS_TRIANGLES
    assert DECLARED_TRIANGLES - NO_CAM_COLLISION_TRIANGLES == ENDLESS_STAIRS_TRIANGLES
    assert constants["SURFACE_NO_CAM_COLLISION"] not in {s.type for s in surfaces}


@needs_sm64_port
def test_endless_stairs_warp_triangles_sit_where_measured():
    """The 12 instant-warp triangles occupy a 154 unit slice of the corridor in z.

    That slice length is why beating the loop needs |forwardVel| above about 154, well past the
    long jump clamp of 48, which makes reaching the top landing self-certifying.
    """
    constants = surface_constants(SURFACE_HEADER)
    surfaces = parse_collision(ENDLESS_STAIRS_COLLISION, constants)
    warps = [s for s in surfaces if s.type == SURFACE_INSTANT_WARP_1B]

    assert len(warps) == WARP_TRIANGLES
    assert bounds(warps) == WARP_BOUNDS

    (x_min, x_max), _, (z_min, z_max) = bounds(warps)
    assert x_max - x_min == 409
    assert z_max - z_min == 154


@needs_sm64_port
def test_endless_stairs_corridor_reaches_both_landings():
    """The parsed area spans the bottom landing at y 3174 and the top landing at y 5018."""
    constants = surface_constants(SURFACE_HEADER)
    surfaces = parse_collision(ENDLESS_STAIRS_COLLISION, constants)

    heights = {vertex[1] for surface in surfaces for vertex in _vertices(surface)}
    assert 3174 in heights
    assert 5018 in heights

    corridor = [s for s in surfaces
                if all(-409 <= vertex[0] <= 0 for vertex in _vertices(s))]
    assert len(corridor) > 100
    _, (y_min, y_max), _ = bounds(corridor)
    assert y_min <= 3174
    assert y_max >= 5018


@needs_sm64_port
def test_endless_stairs_treads_are_level():
    """Every tread inside the corridor is level, and they climb 25.6 units on average.

    The rise lands on an integer grid as an alternating 26, 25 pattern, so no individual tread
    is tilted. Level treads mean normal.y == 1 and an air phase of zero to two frames, which is
    what lets the 1.5x long jump amplifier compound instead of decaying toward the -16 attractor.
    """
    constants = surface_constants(SURFACE_HEADER)
    surfaces = parse_collision(ENDLESS_STAIRS_COLLISION, constants)

    treads = []
    for surface in surfaces:
        vertices = _vertices(surface)
        if not all(-409 <= vertex[0] <= 0 for vertex in vertices):
            continue
        if not all(3174 <= vertex[1] <= 5018 for vertex in vertices):
            continue
        if len({vertex[1] for vertex in vertices}) == 1:
            treads.append(surface)

    assert len(treads) == CORRIDOR_TREADS
    heights = sorted({vertices[0][1] for vertices in map(_vertices, treads)})
    assert len(heights) == CORRIDOR_TREAD_LEVELS
    assert {b - a for a, b in itertools.pairwise(heights)} == {25, 26}

    average_rise = (heights[-1] - heights[0]) / (len(heights) - 1)
    assert average_rise == pytest.approx(25.6, abs=0.01)
