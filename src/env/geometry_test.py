"""Tests for the synthetic surface generators in :mod:`src.env.geometry`.

The generators feed libsm64's static collision loader directly, so the properties that matter
are the ones the collision code reads back: the winding of each triangle (which decides whether
a surface is a floor or a ceiling), the exact integer vertex coordinates, and the tread layout
of a staircase. None of these tests touch the native library.
"""

import math

import pytest

from src.env import geometry
from src.env.native import (
    SURFACE_DEFAULT,
    SURFACE_SLIPPERY,
    TERRAIN_SLIDE,
    TERRAIN_STONE,
    Surface,
)

STAIR_RISE = 25.6
STAIR_RUN = 51.2
STAIR_ANGLE_DEGREES = math.degrees(math.atan2(STAIR_RISE, STAIR_RUN))


def _vertices(surface: Surface) -> list[tuple[int, int, int]]:
    """Copies a surface's three integer vertices out of the ctypes array.

    Args:
        surface: A surface produced by one of the generators.

    Returns:
        The three vertices as plain integer triples, in winding order.
    """
    return [(int(surface.vertices[slot][0]), int(surface.vertices[slot][1]),
             int(surface.vertices[slot][2])) for slot in range(3)]


def _normal(surface: Surface) -> tuple[int, int, int]:
    """Recomputes a surface normal the way the decompilation's read_surface_data does.

    The cross product is taken as (v1 - v0) x (v2 - v0) and left unnormalised, which preserves
    the sign of every component while keeping the arithmetic exact in integers.

    Args:
        surface: A surface produced by one of the generators.

    Returns:
        The unnormalised normal as an integer triple.
    """
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = _vertices(surface)
    ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
    vx, vy, vz = x2 - x0, y2 - y0, z2 - z0
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def _axis_extent(surfaces: list[Surface], axis: int) -> tuple[int, int]:
    """Returns the inclusive minimum and maximum vertex coordinate along one axis.

    Args:
        surfaces: The surfaces to measure.
        axis: 0 for x, 1 for y, 2 for z.

    Returns:
        A (minimum, maximum) pair of integers.
    """
    values = [vertex[axis] for surface in surfaces for vertex in _vertices(surface)]
    return (min(values), max(values))


@pytest.fixture
def restore_material():
    """Restores the module-level surface material after a test mutates it.

    Yields:
        None. The fixture exists only for its teardown.
    """
    saved = (geometry.SURFACE_TYPE, geometry.TERRAIN_TYPE)
    yield
    geometry.set_surface_material(*saved)


def test_ground_plane_is_two_upward_triangles():
    """A ground plane is a single quad, split into two triangles, both facing up."""
    surfaces = geometry.ground_plane(size=8000.0, height=0.0)

    assert len(surfaces) == 2
    for surface in surfaces:
        nx, ny, nz = _normal(surface)
        assert ny > 0
        assert (nx, nz) == (0, 0)
    assert _axis_extent(surfaces, 0) == (-4000, 4000)
    assert _axis_extent(surfaces, 1) == (0, 0)
    assert _axis_extent(surfaces, 2) == (-4000, 4000)


def test_ground_plane_height_offsets_every_vertex():
    """The height argument shifts the whole plane and nothing else."""
    surfaces = geometry.ground_plane(size=1000.0, height=3174.0)

    assert _axis_extent(surfaces, 1) == (3174, 3174)
    assert _axis_extent(surfaces, 0) == (-500, 500)


def test_ramp_rise_equals_length_times_tan_angle():
    """The far edge of a ramp sits exactly length * tan(angle) above the near edge."""
    length = STAIR_RUN * 30.0
    surfaces = geometry.ramp(length=length, width=1200.0, angle_degrees=STAIR_ANGLE_DEGREES)

    expected_rise = round(length * math.tan(math.radians(STAIR_ANGLE_DEGREES)))
    assert expected_rise == round(STAIR_RISE * 30.0)
    assert _axis_extent(surfaces, 1) == (0, expected_rise)
    assert _axis_extent(surfaces, 2) == (0, round(length))


@pytest.mark.parametrize("angle_degrees", [5.0, 15.0, 26.565051177077994, 38.0, 45.0, 60.0])
def test_ramp_rise_matches_tangent_at_every_angle(angle_degrees: float):
    """Every ramp angle produces the rise its tangent predicts, after integer rounding."""
    length = 1000.0
    surfaces = geometry.ramp(length=length, width=200.0, angle_degrees=angle_degrees)

    assert _axis_extent(surfaces, 1) == (0, round(length * math.tan(math.radians(angle_degrees))))


def test_ramp_triangles_face_upward():
    """Both halves of a ramp are floors, and the incline tilts only along z."""
    surfaces = geometry.ramp(length=2000.0, width=800.0, angle_degrees=30.0)

    assert len(surfaces) == 2
    for surface in surfaces:
        nx, ny, nz = _normal(surface)
        assert ny > 0
        assert nx == 0
        assert nz < 0


def test_ramp_base_height_and_origin_z_translate_the_quad():
    """base_height and origin_z move the ramp without changing its rise."""
    surfaces = geometry.ramp(length=1000.0, width=400.0, angle_degrees=45.0,
                             base_height=3174.0, origin_z=2550.0)

    assert _axis_extent(surfaces, 1) == (3174, 4174)
    assert _axis_extent(surfaces, 2) == (2550, 3550)


def test_staircase_tread_count_and_heights():
    """A staircase of n steps is 2n triangles whose treads climb by exactly one rise each."""
    steps = 30
    surfaces = geometry.staircase(steps=steps, rise=STAIR_RISE, run=STAIR_RUN, width=800.0)

    assert len(surfaces) == 2 * steps
    heights = sorted({vertex[1] for surface in surfaces for vertex in _vertices(surface)})
    assert heights == [round(STAIR_RISE * (index + 1)) for index in range(steps)]


def test_staircase_treads_are_perfectly_level():
    """Treads are level however steep the staircase envelope is.

    This is the structural reason a staircase can bootstrap a backwards long jump while a ramp
    of the same slope cannot: a level tread gives normal.y == 1 and therefore an air phase of
    zero to two frames, and it never trips mario_floor_is_slippery.
    """
    surfaces = geometry.staircase(steps=20, rise=STAIR_RISE, run=STAIR_RUN, width=800.0)

    for surface in surfaces:
        vertices = _vertices(surface)
        assert len({vertex[1] for vertex in vertices}) == 1
        nx, ny, nz = _normal(surface)
        assert ny > 0
        assert (nx, nz) == (0, 0)


def test_staircase_envelope_matches_the_endless_stairs_slope():
    """Rise 25.6 over run 51.2 reproduces the 26.565 degree castle_inside envelope."""
    steps = 40
    surfaces = geometry.staircase(steps=steps, rise=STAIR_RISE, run=STAIR_RUN, width=800.0)

    z_min, z_max = _axis_extent(surfaces, 2)
    y_min, y_max = _axis_extent(surfaces, 1)
    envelope = math.degrees(math.atan2(y_max - y_min + round(STAIR_RISE), z_max - z_min))
    assert envelope == pytest.approx(26.565, abs=0.01)


def test_staircase_risers_are_vertical_walls():
    """Risers double the triangle count and carry a horizontal normal, so they are walls."""
    steps = 6
    treads = geometry.staircase(steps=steps, rise=STAIR_RISE, run=STAIR_RUN, width=800.0)
    with_risers = geometry.staircase(steps=steps, rise=STAIR_RISE, run=STAIR_RUN, width=800.0,
                                     risers=True)

    assert len(treads) == 2 * steps
    assert len(with_risers) == 4 * steps
    riser_normals = [_normal(surface) for surface in with_risers[2::4]] + [
        _normal(surface) for surface in with_risers[3::4]]
    for _, ny, _ in riser_normals:
        assert ny == 0


def test_staircase_origin_and_base_place_the_first_tread():
    """The first tread sits one rise above base_height, starting at origin_z."""
    surfaces = geometry.staircase(steps=3, rise=STAIR_RISE, run=STAIR_RUN, width=400.0,
                                  base_height=3174.0, origin_z=-1100.0)

    assert _axis_extent(surfaces, 1) == (3174 + 26, 3174 + 77)
    assert _axis_extent(surfaces, 2) == (-1100, -1100 + round(STAIR_RUN * 3))


def test_flat_area_spans_exactly_the_requested_rectangle():
    """flat_area covers the half-open corridor it is given, at a single height."""
    surfaces = geometry.flat_area(x_min=-409.0, x_max=0.0, z_min=2550.0, z_max=4750.0,
                                  height=3174.0)

    assert len(surfaces) == 2
    assert _axis_extent(surfaces, 0) == (-409, 0)
    assert _axis_extent(surfaces, 1) == (3174, 3174)
    assert _axis_extent(surfaces, 2) == (2550, 4750)
    for surface in surfaces:
        assert _normal(surface)[1] > 0


def test_flat_area_covers_its_interior_corners():
    """The two triangles between them touch all four corners of the rectangle."""
    surfaces = geometry.flat_area(-409.0, 0.0, 2550.0, 4750.0, height=3174.0)

    corners = {(vertex[0], vertex[2]) for surface in surfaces for vertex in _vertices(surface)}
    assert corners == {(-409, 2550), (-409, 4750), (0, 2550), (0, 4750)}


def test_slope_course_is_a_ground_plane_plus_a_ramp():
    """slope_course concatenates the two generators in that order."""
    surfaces = geometry.slope_course(angle_degrees=20.0, length=4000.0, width=1200.0)

    assert len(surfaces) == 4
    assert _vertices(surfaces[0]) == _vertices(geometry.ground_plane()[0])
    assert _vertices(surfaces[2]) == _vertices(
        geometry.ramp(4000.0, 1200.0, 20.0)[0])


def test_vertices_are_rounded_to_int32_half_to_even():
    """Vertex coordinates go through int(round(...)), which rounds halves to even.

    Stair rises of 25.6 are fractional, so the rounding rule is load-bearing: tread heights come
    out 26, 51, 77 rather than 25, 51, 76.
    """
    assert _axis_extent(geometry.flat_area(0.0, 1.0, 0.0, 1.0, height=25.6), 1) == (26, 26)
    assert _axis_extent(geometry.flat_area(0.0, 1.0, 0.0, 1.0, height=-25.6), 1) == (-26, -26)
    assert _axis_extent(geometry.flat_area(0.0, 1.0, 0.0, 1.0, height=0.5), 1) == (0, 0)
    assert _axis_extent(geometry.flat_area(0.0, 1.0, 0.0, 1.0, height=1.5), 1) == (2, 2)
    assert _axis_extent(geometry.flat_area(0.0, 1.0, 0.0, 1.0, height=2.5), 1) == (2, 2)


def test_generated_surfaces_default_to_stone_with_no_special_type():
    """Without an explicit material every surface is SURFACE_DEFAULT on TERRAIN_STONE."""
    surfaces = geometry.ground_plane() + geometry.staircase(4, STAIR_RISE, STAIR_RUN, 400.0)

    for surface in surfaces:
        assert surface.type == SURFACE_DEFAULT
        assert surface.terrain == TERRAIN_STONE
        assert surface.force == 0


def test_set_surface_material_applies_to_later_generators(restore_material):
    """set_surface_material changes the material of every surface built afterwards."""
    geometry.set_surface_material(SURFACE_SLIPPERY, TERRAIN_SLIDE)
    slippery = geometry.ground_plane()

    for surface in slippery:
        assert surface.type == SURFACE_SLIPPERY
        assert surface.terrain == TERRAIN_SLIDE

    geometry.set_surface_material(SURFACE_DEFAULT, TERRAIN_STONE)
    for surface in geometry.ground_plane():
        assert surface.type == SURFACE_DEFAULT
        assert surface.terrain == TERRAIN_STONE
