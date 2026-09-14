"""Reads the decompilation's ``collision.inc.c`` files as collision data.

Those files are C, but only nominally: they are lists of ``COL_VERTEX`` and ``COL_TRI`` macro
calls that a build turns into an array of integers. Parsing the macros textually gets the same
numbers without a toolchain, which is what lets this project load the castle's real staircase
instead of an approximation of it, on a machine with no compiler and no ROM extraction step.

The parse is deliberately forgiving. A surface type this project has no constant for is skipped
rather than guessed at, an out of range vertex index drops its triangle rather than clamping it,
and the three camera only surface types are dropped as intangible. On ``castle_inside`` area 2
that last rule accounts for 96 of the 2019 declared triangles, leaving the 1923 Mario can stand
on. ``collision_test`` pins all of it, including those counts.
"""

import re

from src.env.native import Surface

VERTEX_RE = re.compile(r"COL_VERTEX\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
TRI_INIT_RE = re.compile(r"COL_TRI_INIT\(\s*([A-Za-z0-9_]+)\s*,\s*(\d+)\s*\)")
TRI_RE = re.compile(r"COL_TRI\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")

SURFACE_FLOOR_SKIP = {
    "SURFACE_CAMERA_BOUNDARY",
    "SURFACE_NO_CAM_COLLISION",
    "SURFACE_NO_CAM_COL_SLIPPERY",
}


def surface_constants(header_path: str) -> dict[str, int]:
    """Reads the SURFACE_ constants out of ``surface_terrains.h``.

    The collision files name their surface types and the header holds the values, so the two have
    to be read together. Taking the values from the vendored header rather than hard coding them
    means the surface type that makes the staircase a loop is whatever the checkout says it is.

    Args:
        header_path: Path to ``surface_terrains.h``.

    Returns:
        A mapping from constant name to value, for the decimal and hex spellings alike.

    Raises:
        OSError: If the header cannot be read.
    """
    values: dict[str, int] = {}
    pattern = re.compile(r"#define\s+(SURFACE_[A-Z0-9_]+)\s+(0x[0-9A-Fa-f]+|\d+)")
    with open(header_path, encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                values[match.group(1)] = int(match.group(2), 0)
    return values


def parse_collision(path: str, constants: dict[str, int],
                    terrain: int = 0) -> list[Surface]:
    """Turns one ``collision.inc.c`` into surfaces libsm64 can load.

    The file's structure is what makes this work: every ``COL_TRI_INIT`` names a surface type and
    declares how many triangles follow it, so the count in the macro, rather than any bracket
    matching, is what ends a block. Vertex indices are resolved against the ``COL_VERTEX`` list in
    file order.

    Args:
        path: Path to the collision file.
        constants: Surface name to value mapping, from :func:`surface_constants`. A block whose
            name is missing is skipped, so a partial table narrows the scene rather than failing.
        terrain: Terrain type written onto every surface. The files do not carry one.

    Returns:
        Every tangible triangle, in declaration order.

    Raises:
        OSError: If the collision file cannot be read.
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    vertices = [(int(a), int(b), int(c)) for a, b, c in VERTEX_RE.findall(text)]
    surfaces: list[Surface] = []

    for init in TRI_INIT_RE.finditer(text):
        name, count = init.group(1), int(init.group(2))
        if name in SURFACE_FLOOR_SKIP or name not in constants:
            continue
        block = text[init.end():]
        for tri in list(TRI_RE.finditer(block))[:count]:
            indices = [int(tri.group(i)) for i in (1, 2, 3)]
            if any(index >= len(vertices) for index in indices):
                continue
            surface = Surface()
            surface.type = constants[name]
            surface.force = 0
            surface.terrain = terrain
            for slot, index in enumerate(indices):
                for axis in range(3):
                    surface.vertices[slot][axis] = vertices[index][axis]
            surfaces.append(surface)
    return surfaces


def bounds(surfaces: list[Surface]) -> tuple[tuple[int, int], ...]:
    """Measures the axis aligned extent of a set of surfaces.

    This is how the scene locates itself. The warp zone's depth along z, which sets the speed the
    exploit has to beat, is read off the bounds of the twelve warp triangles rather than written
    down as a constant.

    Args:
        surfaces: Surfaces to measure.

    Returns:
        One inclusive (minimum, maximum) pair per axis, in x, y, z order.

    Raises:
        ValueError: If the list is empty, since an empty extent has no meaning.
    """
    extents = []
    for axis in range(3):
        values = [s.vertices[i][axis] for s in surfaces for i in range(3)]
        extents.append((min(values), max(values)))
    return tuple(extents)
