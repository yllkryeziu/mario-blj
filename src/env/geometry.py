import math

from src.env.native import SURFACE_DEFAULT, TERRAIN_STONE, Surface

Point = tuple[float, float, float]


def _triangle(a: Point, b: Point, c: Point) -> Surface:
    surface = Surface()
    surface.type = SURFACE_DEFAULT
    surface.force = 0
    surface.terrain = TERRAIN_STONE
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
    half = size / 2.0
    return _floor_quad(
        (-half, height, -half), (half, height, -half),
        (half, height, half), (-half, height, half))


def ramp(length: float, width: float, angle_degrees: float,
         base_height: float = 0.0, origin_z: float = 0.0) -> list[Surface]:
    half = width / 2.0
    rise = length * math.tan(math.radians(angle_degrees))
    far = origin_z + length
    return _floor_quad(
        (-half, base_height, origin_z), (half, base_height, origin_z),
        (half, base_height + rise, far), (-half, base_height + rise, far))


def staircase(steps: int, rise: float, run: float, width: float,
              base_height: float = 0.0, origin_z: float = 0.0) -> list[Surface]:
    half = width / 2.0
    surfaces: list[Surface] = []
    for index in range(steps):
        y = base_height + rise * (index + 1)
        z0 = origin_z + run * index
        z1 = z0 + run
        surfaces += _floor_quad((-half, y, z0), (half, y, z0), (half, y, z1), (-half, y, z1))
        surfaces += [
            _triangle((-half, y - rise, z0), (half, y - rise, z0), (half, y, z0)),
            _triangle((-half, y - rise, z0), (half, y, z0), (-half, y, z0)),
        ]
    return surfaces


def slope_course(angle_degrees: float, length: float = 4000.0,
                 width: float = 1200.0) -> list[Surface]:
    return ground_plane() + ramp(length, width, angle_degrees)
