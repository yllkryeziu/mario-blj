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
    extents = []
    for axis in range(3):
        values = [s.vertices[i][axis] for s in surfaces for i in range(3)]
        extents.append((min(values), max(values)))
    return tuple(extents)
