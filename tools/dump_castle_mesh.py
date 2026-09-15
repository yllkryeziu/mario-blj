"""Extracts the endless staircase's renderable geometry out of the decompilation.

The environment loads the staircase from ``collision.inc.c``, which carries surface types but no
texture coordinates, so every existing viewer draws the room as tinted collision triangles. The
renderable version of the same room lives in a display list -- ``areas/2/13/model.inc.c``, the only
model in area 2 spanning the full y 3174..5018 band -- and that one has real UVs and names the four
castle textures it uses. This reads it.

Only the subset of F3DEX2 the castle rooms actually use is interpreted: a vertex-buffer load, a
texture bind, the two triangle commands, and nested display lists. Everything to do with combiner
modes, fog and render modes is deliberately ignored, because the target is a WebGL viewer with its
own shading model rather than an N64 emulator.

Two details are load-bearing and easy to get wrong. The fourth field of a ``Vtx`` is a normal here,
not a vertex colour, because the room lights itself with ``gsSPLight``; read it as RGBA and the
stairs come out in garish primaries. And texture coordinates are S10.5 fixed point whose values run
past 800 texels, so they have to be divided by 32 and then by the texture's pixel size and sampled
with wrapping -- clamping smears every tread into a streak.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import zlib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = os.path.join(
    _ROOT, "third_party", "sm64-port", "levels", "castle_inside", "areas", "2", "13", "model.inc.c"
)
DEFAULT_TEXTURE_DIR = os.path.join(_ROOT, "third_party", "sm64-port", "textures", "inside")

# bin/inside.c pairs each `inside_090SSSSS` symbol with exactly one extracted PNG, and the segment
# offset in the symbol is the offset in the filename, so the mapping needs no manifest.
TEXTURE_SYMBOL = re.compile(r"^inside_090(?P<offset>[0-9A-Fa-f]{5})$")
TEXTURE_FILE = "inside_castle_textures.{offset}.rgba16.png"

_VTX_ARRAY = re.compile(
    r"static\s+const\s+Vtx\s+(?P<name>\w+)\s*\[\s*\]\s*=\s*\{(?P<body>.*?)\}\s*;", re.DOTALL
)
_VTX_ENTRY = re.compile(
    r"\{\s*\{\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}\s*,"      # position
    r"\s*(-?\w+)\s*,"                                                    # unused flag
    r"\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*\}\s*,"                            # S10.5 texture coords
    r"\s*\{\s*(0x[0-9A-Fa-f]+|-?\d+)\s*,\s*(0x[0-9A-Fa-f]+|-?\d+)\s*,"   # normal xyz
    r"\s*(0x[0-9A-Fa-f]+|-?\d+)\s*,\s*(0x[0-9A-Fa-f]+|-?\d+)\s*\}\s*\}\s*\}",
    re.DOTALL,
)
_GFX_ARRAY = re.compile(
    r"(?:static\s+)?const\s+Gfx\s+(?P<name>\w+)\s*\[\s*\]\s*=\s*\{(?P<body>.*?)\}\s*;", re.DOTALL
)
_SET_TEXTURE = re.compile(r"gsDPSetTextureImage\s*\(\s*\w+\s*,\s*\w+\s*,\s*\d+\s*,\s*(\w+)\s*\)")
_SET_TILE_SIZE = re.compile(
    r"gsDPSetTileSize\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,"
    r"\s*\(\s*(\d+)\s*-\s*1\s*\)\s*<<[^,]+,\s*\(\s*(\d+)\s*-\s*1\s*\)\s*<<[^)]+\)"
)
_VERTEX = re.compile(r"gsSPVertex\s*\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
_TRI1 = re.compile(r"gsSP1Triangle\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\w+\s*\)")
_TRI2 = re.compile(
    r"gsSP2Triangles\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\w+\s*,"
    r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\w+\s*\)"
)
_SUB_LIST = re.compile(r"gsSPDisplayList\s*\(\s*(\w+)\s*\)")
_COMBINE = re.compile(r"gsDPSetCombineMode\s*\(\s*(\w+)\s*,")


def _signed_byte(token: str) -> int:
    """Reads one ``Vtx`` normal component, which the decompilation writes as an unsigned hex byte.

    Args:
        token: The literal as it appears in the source, either ``0x..`` or a decimal integer.

    Returns:
        The component in -128..127.
    """
    value = int(token, 16) if token.lower().startswith("0x") else int(token)
    return value - 256 if value > 127 else value


def parse_vertex_arrays(source: str) -> dict[str, list[dict]]:
    """Collects every ``Vtx`` table in the file.

    Args:
        source: Contents of a ``model.inc.c``.

    Returns:
        Array name to list of vertices, each ``{position, uv_fixed, normal}``.
    """
    arrays: dict[str, list[dict]] = {}
    for match in _VTX_ARRAY.finditer(source):
        vertices = []
        for entry in _VTX_ENTRY.finditer(match.group("body")):
            x, y, z, _flag, u, v, nx, ny, nz, _alpha = entry.groups()
            vertices.append(
                {
                    "position": (int(x), int(y), int(z)),
                    "uv_fixed": (int(u), int(v)),
                    "normal": (_signed_byte(nx), _signed_byte(ny), _signed_byte(nz)),
                }
            )
        arrays[match.group("name")] = vertices
    return arrays


def parse_display_lists(source: str) -> dict[str, list[tuple[str, tuple]]]:
    """Reduces every ``Gfx`` table to the command subset this extractor understands.

    Args:
        source: Contents of a ``model.inc.c``.

    Returns:
        Display list name to an ordered list of ``(opcode, operands)`` pairs.
    """
    lists: dict[str, list[tuple[str, tuple]]] = {}
    for match in _GFX_ARRAY.finditer(source):
        commands: list[tuple[str, tuple]] = []
        for line in match.group("body").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if found := _SET_TEXTURE.search(line):
                commands.append(("texture", (found.group(1),)))
            if found := _SET_TILE_SIZE.search(line):
                commands.append(("tile_size", (int(found.group(1)), int(found.group(2)))))
            if found := _COMBINE.search(line):
                commands.append(("combine", (found.group(1),)))
            if found := _VERTEX.search(line):
                commands.append(
                    ("vertex", (found.group(1), int(found.group(2)), int(found.group(3))))
                )
            if found := _TRI1.search(line):
                commands.append(("tri", tuple(int(g) for g in found.groups())))
            if found := _TRI2.search(line):
                groups = [int(g) for g in found.groups()]
                commands.append(("tri", tuple(groups[0:3])))
                commands.append(("tri", tuple(groups[3:6])))
            if found := _SUB_LIST.search(line):
                commands.append(("call", (found.group(1),)))
        lists[match.group("name")] = commands
    return lists


class _Walker:
    """Interprets a display list tree into per-material triangle batches.

    The N64 render state is global and survives a ``gsSPDisplayList``, which is exactly how this
    room works: the entry list binds a tile size and then calls three sublists that each bind their
    own texture image. So the walker carries one mutable state across the whole traversal rather
    than scoping it per list.
    """

    def __init__(self, vertex_arrays: dict[str, list[dict]],
                 display_lists: dict[str, list[tuple[str, tuple]]]):
        self.vertex_arrays = vertex_arrays
        self.display_lists = display_lists
        self.texture: str | None = None
        self.tile_size: tuple[int, int] | None = None
        self.textured = True
        self.buffer: list[dict] = []
        self.batches: dict[tuple, list[tuple[dict, dict, dict]]] = {}
        self.skipped_vertex_offsets = 0

    def _key(self) -> tuple:
        """Returns the material key triangles are grouped under."""
        if not self.textured or self.texture is None:
            return ("untextured", None)
        return (self.texture, self.tile_size)

    def walk(self, name: str, depth: int = 0) -> None:
        """Executes one display list, recursing into the lists it calls.

        Args:
            name: Display list symbol to execute.
            depth: Recursion depth, used only to refuse pathological nesting.
        """
        if depth > 8 or name not in self.display_lists:
            return
        for opcode, operands in self.display_lists[name]:
            if opcode == "texture":
                self.texture = operands[0]
            elif opcode == "tile_size":
                self.tile_size = operands
            elif opcode == "combine":
                # G_CC_SHADE means vertex shading with no texture fetch at all.
                self.textured = operands[0] != "G_CC_SHADE"
            elif opcode == "vertex":
                array, count, offset = operands
                if offset != 0:
                    self.skipped_vertex_offsets += 1
                self.buffer = self.vertex_arrays.get(array, [])[:count]
            elif opcode == "tri":
                if all(index < len(self.buffer) for index in operands):
                    triangle = tuple(self.buffer[index] for index in operands)
                    self.batches.setdefault(self._key(), []).append(triangle)
            elif opcode == "call":
                self.walk(operands[0], depth + 1)


def png_size(path: str) -> tuple[int, int]:
    """Reads a PNG's pixel dimensions from its IHDR without decoding the image.

    Args:
        path: Path to a PNG file.

    Returns:
        ``(width, height)``.
    """
    with open(path, "rb") as handle:
        header = handle.read(24)
    return struct.unpack(">II", header[16:24])


def decode_png_rgba(path: str) -> tuple[int, int, bytearray]:
    """Decodes an 8-bit PNG to RGBA8, enough of the format for the extracted castle textures.

    The extracted textures are written by ``tools/n64graphics`` as non-interlaced 8-bit RGBA or
    grey+alpha, so only those two colour types and the five standard filters are handled.

    Args:
        path: Path to a PNG file.

    Returns:
        ``(width, height, rgba)`` with four bytes per pixel, top row first.

    Raises:
        ValueError: If the file uses a PNG feature the castle textures never use.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width = height = depth = colour = 0
    idat = bytearray()
    offset = 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset:offset + 4])
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour, _comp, _filt, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or interlace != 0 or colour not in (4, 6):
                raise ValueError(f"{path}: unsupported PNG (depth {depth}, colour {colour})")
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        offset += 12 + length

    channels = 4 if colour == 6 else 2
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray(width * height * 4)
    previous = bytearray(stride)
    position = 0
    for row in range(height):
        filter_type = raw[position]
        position += 1
        line = bytearray(raw[position:position + stride])
        position += stride
        for index in range(stride):
            left = line[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                line[index] = (line[index] + left) & 0xFF
            elif filter_type == 2:
                line[index] = (line[index] + up) & 0xFF
            elif filter_type == 3:
                line[index] = (line[index] + ((left + up) >> 1)) & 0xFF
            elif filter_type == 4:
                estimate = left + up - upper_left
                distances = (
                    abs(estimate - left), abs(estimate - up), abs(estimate - upper_left)
                )
                nearest = (left, up, upper_left)[distances.index(min(distances))]
                line[index] = (line[index] + nearest) & 0xFF
            elif filter_type != 0:
                raise ValueError(f"{path}: unknown PNG filter {filter_type}")
        base = row * width * 4
        for pixel in range(width):
            source = pixel * channels
            if channels == 4:
                out[base + pixel * 4:base + pixel * 4 + 4] = line[source:source + 4]
            else:
                grey, alpha = line[source], line[source + 1]
                out[base + pixel * 4:base + pixel * 4 + 4] = bytes((grey, grey, grey, alpha))
        previous = line
    return width, height, out


def texture_path(symbol: str, texture_dir: str) -> str | None:
    """Maps a texture symbol to the PNG the build extracts it from.

    Args:
        symbol: A ``inside_090SSSSS`` symbol as it appears in a display list.
        texture_dir: Directory holding the extracted ``inside_castle_textures.*.png`` files.

    Returns:
        Path to the PNG, or None if the symbol is not one of the castle's segment textures.
    """
    match = TEXTURE_SYMBOL.match(symbol)
    if not match:
        return None
    offset = match.group("offset").upper()
    candidate = os.path.join(texture_dir, TEXTURE_FILE.format(offset=offset[:5]))
    if os.path.exists(candidate):
        return candidate
    # The symbols carry a five-digit segment offset while the filenames drop the leading zero.
    candidate = os.path.join(texture_dir, TEXTURE_FILE.format(offset=offset.lstrip("0") or "0"))
    return candidate if os.path.exists(candidate) else None


def extract(model_path: str, entry: str | None, texture_dir: str,
            embed_textures: bool = False) -> dict:
    """Extracts one room's renderable mesh.

    Args:
        model_path: Path to a ``model.inc.c``.
        entry: Display list to walk. Defaults to the file's only non-static ``Gfx`` table, which is
            the room's exported entry point.
        texture_dir: Directory holding the extracted texture PNGs.
        embed_textures: Inline each texture's PNG bytes as base64, which is what the viewers want
            because they are single self-contained files.

    Returns:
        A dict with one entry per material, each carrying flat position, uv and normal lists.

    Raises:
        ValueError: If no entry display list can be determined.
    """
    with open(model_path, encoding="utf-8") as handle:
        source = handle.read()
    vertex_arrays = parse_vertex_arrays(source)
    display_lists = parse_display_lists(source)

    if entry is None:
        exported = re.findall(r"^const\s+Gfx\s+(\w+)\s*\[", source, re.MULTILINE)
        if len(exported) != 1:
            raise ValueError(f"{model_path}: expected one exported Gfx, found {exported}")
        entry = exported[0]

    walker = _Walker(vertex_arrays, display_lists)
    walker.walk(entry)

    materials = []
    for (symbol, tile_size), triangles in walker.batches.items():
        positions: list[int] = []
        uvs: list[float] = []
        normals: list[int] = []
        width = height = 32
        png = None if symbol == "untextured" else texture_path(symbol, texture_dir)
        if png:
            width, height = png_size(png)
        for triangle in triangles:
            for vertex in triangle:
                positions.extend(vertex["position"])
                normals.extend(vertex["normal"])
                # gfx_pc.c:861-868 with uls = 0 and a unit texture scale reduces to this.
                u, v = vertex["uv_fixed"]
                uvs.extend((u / 32.0 / width, v / 32.0 / height))
        materials.append(
            {
                "texture": symbol,
                "texture_png": os.path.relpath(png, _ROOT) if png else None,
                "texture_size": [width, height],
                "tile_size": list(tile_size) if tile_size else None,
                "triangles": len(triangles),
                "positions": positions,
                "uvs": [round(value, 6) for value in uvs],
                "normals": normals,
            }
        )
        if embed_textures and png:
            with open(png, "rb") as handle:
                materials[-1]["png_base64"] = base64.b64encode(handle.read()).decode("ascii")
    materials.sort(key=lambda material: -material["triangles"])

    all_positions = [value for material in materials for value in material["positions"]]
    bounds = {
        axis: [min(all_positions[index::3]), max(all_positions[index::3])]
        for index, axis in enumerate("xyz")
    }
    return {
        "source": os.path.relpath(model_path, _ROOT),
        "entry": entry,
        "vertex_arrays": len(vertex_arrays),
        "display_lists": len(display_lists),
        "triangles": sum(material["triangles"] for material in materials),
        "skipped_vertex_offsets": walker.skipped_vertex_offsets,
        "bounds": bounds,
        "materials": materials,
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model.inc.c to extract")
    parser.add_argument("--entry", default=None, help="display list to walk")
    parser.add_argument("--texture_dir", default=DEFAULT_TEXTURE_DIR,
                        help="extracted PNG directory")
    parser.add_argument("--out", default=None, help="write the mesh as JSON here")
    parser.add_argument("--summary", action="store_true", help="print a per-material summary")
    parser.add_argument("--embed_textures", action="store_true",
                        help="inline each texture PNG as base64")
    args = parser.parse_args()

    mesh = extract(args.model, args.entry, args.texture_dir, args.embed_textures)
    if args.summary:
        print(f"{mesh['source']}  entry={mesh['entry']}")
        print(f"  {mesh['vertex_arrays']} vertex arrays, {mesh['display_lists']} display lists, "
              f"{mesh['triangles']} triangles, {mesh['skipped_vertex_offsets']} non-zero v0")
        for axis, (low, high) in mesh["bounds"].items():
            print(f"  {axis}: {low} .. {high}")
        for material in mesh["materials"]:
            uvs = material["uvs"]
            span = (min(uvs), max(uvs)) if uvs else (0, 0)
            print(f"  {material['texture']:<18} {material['triangles']:>4} tris  "
                  f"{material['texture_size'][0]}x{material['texture_size'][1]}  "
                  f"uv {span[0]:.2f}..{span[1]:.2f}  {material['texture_png']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(mesh, handle, separators=(",", ":"))
        print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
