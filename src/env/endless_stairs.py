"""The castle's endless staircase, loaded from the decompilation's own collision data.

The staircase is the canonical backwards long jump target because it cannot be climbed any other
way. Its loop is not geometry. Twelve of its triangles carry the surface type
``SURFACE_INSTANT_WARP_1B``, and ``check_instant_warp`` in the decompilation's
``src/game/level_update.c`` displaces Mario by a fixed offset whenever his current floor is one of
them, unless the save file holds 70 or more stars. libsm64 carries Mario and surfaces but no level
logic, so this module reimplements that one check against the same surface type and the same
displacement the level script declares.

Beating the loop requires crossing the warp zone inside a single frame. The zone is 154 units deep
in z and Mario's fastest ordinary movement is the long jump clamp at 48 units per frame, so the
task certifies itself: reaching the top landing is only possible with a speed no ordinary action
can produce.
"""

from __future__ import annotations

import dataclasses
import os

from src.env.collision import parse_collision, surface_constants
from src.env.native import Sm64, Surface

SURFACE_INSTANT_WARP_1B = 0x001B

_DEFAULT_COLLISION = os.path.join(
    "third_party", "sm64-port", "levels", "castle_inside", "areas", "2", "collision.inc.c")
_DEFAULT_HEADER = os.path.join(
    "third_party", "libsm64", "src", "decomp", "include", "surface_terrains.h")


@dataclasses.dataclass(frozen=True)
class WarpZone:
    """The bounding box of the staircase's instant warp triangles.

    Attributes:
        surface_type: Surface type that triggers the warp.
        displacement: Offset applied to Mario's position when the warp fires, taken from the
            level script's ``INSTANT_WARP`` entry.
        x_range: Inclusive x extent of the triggering triangles.
        y_range: Inclusive y extent of the triggering triangles.
        z_range: Inclusive z extent of the triggering triangles.
    """

    surface_type: int
    displacement: tuple[float, float, float]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]

    @property
    def depth(self) -> float:
        """Returns the depth of the zone along z, the axis the staircase ascends."""
        return self.z_range[1] - self.z_range[0]


@dataclasses.dataclass(frozen=True)
class Scene:
    """A loaded staircase ready to hand to libsm64.

    Attributes:
        surfaces: Every collision triangle of the area, warp triangles included.
        spawn: Position on the bottom landing where an episode starts.
        goal_y: Height of the top landing, the success threshold.
        goal_z: Depth of the top landing, reached by travelling toward negative z.
        warp: The instant warp this staircase loops through.
        ascends_toward: Unit vector in the xz plane pointing up the staircase.
    """

    surfaces: list[Surface]
    spawn: tuple[float, float, float]
    goal_y: float
    goal_z: float
    warp: WarpZone
    ascends_toward: tuple[float, float]


def _extent(surfaces: list[Surface], axis: int) -> tuple[float, float]:
    values = [surface.vertices[corner][axis] for surface in surfaces for corner in range(3)]
    return float(min(values)), float(max(values))


def load_scene(collision_path: str = _DEFAULT_COLLISION,
               header_path: str = _DEFAULT_HEADER,
               spawn: tuple[float, float, float] = (-200.0, 3204.0, 3000.0),
               displacement: tuple[float, float, float] = (0.0, -205.0, 410.0)) -> Scene:
    """Loads the endless staircase from the decompilation's collision data.

    Args:
        collision_path: Path to ``castle_inside/areas/2/collision.inc.c``.
        header_path: Path to ``surface_terrains.h``, for the surface type constants.
        spawn: Episode start, on the bottom landing a little above the floor.
        displacement: Instant warp offset. The default is the castle's own
            ``INSTANT_WARP(0, 2, 0, -205, 410)`` from ``levels/castle_inside/script.c``.

    Returns:
        A populated scene.

    Raises:
        FileNotFoundError: If either path is missing.
        ValueError: If the collision data holds no instant warp triangles, which would mean the
            loop is absent and the task would not be self certifying.
    """
    constants = surface_constants(header_path)
    surfaces = parse_collision(collision_path, constants)
    warp_type = constants.get("SURFACE_INSTANT_WARP_1B", SURFACE_INSTANT_WARP_1B)
    warp_surfaces = [surface for surface in surfaces if surface.type == warp_type]
    if not warp_surfaces:
        raise ValueError(f"no instant warp triangles in {collision_path}")

    warp = WarpZone(
        surface_type=warp_type,
        displacement=displacement,
        x_range=_extent(warp_surfaces, 0),
        y_range=_extent(warp_surfaces, 1),
        z_range=_extent(warp_surfaces, 2),
    )
    _, max_y = _extent(surfaces, 1)
    min_z, _ = _extent(surfaces, 2)
    return Scene(
        surfaces=surfaces,
        spawn=spawn,
        goal_y=4966.0,
        goal_z=-1000.0,
        warp=warp,
        ascends_toward=(0.0, -1.0),
    )


def apply_instant_warp(game: Sm64, floor_type: int, position: tuple[float, float, float],
                       warp: WarpZone) -> bool:
    """Runs the decompilation's ``check_instant_warp`` for one frame.

    The real check reads ``gMarioState->floor`` once per frame and does not care whether Mario is
    grounded, so a frame spent airborne directly above a warp triangle still triggers it. That is
    why only raw speed defeats the loop.

    Args:
        game: Live libsm64 handle, used to move Mario if the warp fires.
        floor_type: Surface type of Mario's current floor, or -1 when he has none.
        position: Mario's current position.
        warp: The warp to test against.

    Returns:
        True if the warp fired and Mario was displaced.
    """
    if floor_type != warp.surface_type:
        return False
    game.set_position(position[0] + warp.displacement[0],
                      position[1] + warp.displacement[1],
                      position[2] + warp.displacement[2])
    return True


def minimum_escape_speed(warp: WarpZone) -> float:
    """Returns the speed below which the warp zone cannot be skipped.

    A once per frame check cannot be dodged unless consecutive samples straddle the zone, which
    needs a per frame displacement larger than the zone's own depth.
    """
    return warp.depth
