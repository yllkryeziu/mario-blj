"""Synthetic collision scenes, for asking which geometry the speed chain needs.

The real target is the castle staircase, but a run there answers only one question at a time. The
generators here build the controlled variants the sweeps in ``scripts/`` use instead: flat ground
to calibrate the stick against, a uniform ramp to vary the slope angle on its own, and a staircase
to vary rise and run independently. ``results/blj_runaway.json`` is the outcome of driving all
three through the same scripted policy.

Two conventions are forced by the substrate. Vertices are integers, because libsm64's surface
struct stores them that way, so every generator rounds and a caller cannot ask for sub unit
placement. And winding decides whether a triangle is a floor or a ceiling, so the quad helper
orders the corners for an upward normal rather than trusting the caller to pass them in a
particular order.

Surface type and terrain are module state that :func:`set_surface_material` changes, which is how
a whole scene is made slippery without threading a material through every generator.
"""

import math

from src.env.native import SURFACE_DEFAULT, TERRAIN_STONE, Surface

SURFACE_TYPE = SURFACE_DEFAULT
TERRAIN_TYPE = TERRAIN_STONE

Point = tuple[float, float, float]


def _triangle(a: Point, b: Point, c: Point) -> Surface:
    surface = Surface()
    surface.type = SURFACE_TYPE
    surface.force = 0
    surface.terrain = TERRAIN_TYPE
    for index, point in enumerate((a, b, c)):
        for axis in range(3):
            surface.vertices[index][axis] = int(round(point[axis]))
    return surface


def _upward_normal_y(a: Point, b: Point, c: Point) -> float:
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    return uz * vx - ux * vz


def _floor_quad(a: Point, b: Point, c: Point, d: Point) -> list[Surface]:
    if _upward_normal_y(a, b, c) < 0:
        a, b, c, d = a, d, c, b
    return [_triangle(a, b, c), _triangle(a, c, d)]


def ground_plane(size: float = 8000.0, height: float = 0.0) -> list[Surface]:
    """Builds one square floor centred on the origin.

    This is the control scene. The stick calibration in ``src.agent.scripted`` runs on it because
    a flat floor has no slope term, so the displacement it measures is the stick mapping alone.

    Args:
        size: Edge length in world units.
        height: y of the floor.

    Returns:
        Two triangles wound to face up.
    """
    half = size / 2.0
    return _floor_quad(
        (-half, height, -half), (half, height, -half),
        (half, height, half), (-half, height, half))


def ramp(length: float, width: float, angle_degrees: float,
         base_height: float = 0.0, origin_z: float = 0.0) -> list[Surface]:
    """Builds one uniform slope rising toward positive z.

    A ramp isolates the slope angle from the tread geometry a staircase adds, which is what makes
    it the right scene for asking whether the chain needs steps at all or only a gradient.

    Args:
        length: Extent along z.
        width: Extent along x, centred on x zero.
        angle_degrees: Slope angle. The rise follows from the length and the tangent.
        base_height: y of the near edge.
        origin_z: z of the near edge.

    Returns:
        Two triangles wound to face up.
    """
    half = width / 2.0
    rise = length * math.tan(math.radians(angle_degrees))
    far = origin_z + length
    return _floor_quad(
        (-half, base_height, origin_z), (half, base_height, origin_z),
        (half, base_height + rise, far), (-half, base_height + rise, far))


def staircase(steps: int, rise: float, run: float, width: float,
              base_height: float = 0.0, origin_z: float = 0.0,
              risers: bool = False) -> list[Surface]:
    """Builds a flight of flat treads ascending toward positive z.

    Rise and run are separate arguments rather than an angle because the sweep varies them
    independently: the chain cares about the height of the step it lands on as well as the overall
    gradient, and those are the same angle at many different step sizes.

    The risers are off by default. A tread on its own leaves Mario's collision nothing vertical to
    catch on, which is the permissive case; turning them on adds the wall the real staircase has.

    Args:
        steps: Number of treads.
        rise: Height gained per step.
        run: Depth of each tread along z.
        width: Extent along x, centred on x zero.
        base_height: y of the floor the first step rises from.
        origin_z: z of the near edge of the first tread.
        risers: Whether to add the vertical face under each tread.

    Returns:
        Two triangles per tread, plus two more per riser when risers are on.
    """
    half = width / 2.0
    surfaces: list[Surface] = []
    for index in range(steps):
        y = base_height + rise * (index + 1)
        z0 = origin_z + run * index
        z1 = z0 + run
        surfaces += _floor_quad((-half, y, z0), (half, y, z0), (half, y, z1), (-half, y, z1))
        if risers:
            surfaces += [
                _triangle((-half, y - rise, z0), (half, y - rise, z0), (half, y, z0)),
                _triangle((-half, y - rise, z0), (half, y, z0), (-half, y, z0)),
            ]
    return surfaces


def slope_course(angle_degrees: float, length: float = 4000.0,
                 width: float = 1200.0) -> list[Surface]:
    """Builds a ramp with a ground plane under it, so a run cannot fall off the near edge.

    Args:
        angle_degrees: Slope angle of the ramp.
        length: Extent of the ramp along z.
        width: Extent of the ramp along x.

    Returns:
        The plane's triangles followed by the ramp's.
    """
    return ground_plane() + ramp(length, width, angle_degrees)


def set_surface_material(surface_type: int, terrain: int) -> None:
    """Sets the material every later generated surface carries.

    Material is process state here because it is a property of an experiment rather than of a
    triangle: a sweep asks what the chain does on ice, not what one quad does. Surfaces already
    built keep the material they were built with, so this has to be called before the scene.

    Args:
        surface_type: One of the SURFACE_ constants from ``src.env.native``.
        terrain: One of the TERRAIN_ constants from ``src.env.native``.
    """
    global SURFACE_TYPE, TERRAIN_TYPE
    SURFACE_TYPE, TERRAIN_TYPE = surface_type, terrain


def flat_area(x_min: float, x_max: float, z_min: float, z_max: float,
              height: float = 0.0) -> list[Surface]:
    """Builds one rectangular floor from explicit bounds.

    ``ground_plane`` is centred on the origin, which is wrong for an approach run: the scripted
    policy needs thousands of units of floor behind the spawn and very little in front of it. This
    takes the bounds directly so a scene can be laid out asymmetrically.

    Args:
        x_min: Low x edge.
        x_max: High x edge.
        z_min: Low z edge.
        z_max: High z edge.
        height: y of the floor.

    Returns:
        Two triangles wound to face up.
    """
    return _floor_quad(
        (x_min, height, z_min), (x_max, height, z_min),
        (x_max, height, z_max), (x_min, height, z_max))
