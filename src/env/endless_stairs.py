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


def synthetic_scene(rise: float, run: float,
                    climb: float = 1792.0,
                    warp_treads: int = 3,
                    loop_treads: int = 8,
                    warp_height: float = 3917.0,
                    width: float = 6000.0,
                    base_height: float = 3174.0,
                    spawn: tuple[float, float, float] = (-200.0, 3204.0, 3000.0),
                    landing_z: float = 2800.0,
                    risers: bool = True) -> Scene:
    """Builds an endless staircase with treads of a chosen size, for transfer tests.

    The castle's staircase is one point in a two dimensional space of tread geometries, and a
    policy trained on it cannot be asked whether it learned the chain or memorized that one
    flight without a second flight to try. This generator makes those flights, using the same
    :func:`src.env.geometry.staircase` the scripted geometry sweep ran on, and wraps the result
    in a scene carrying a warp band so that success is decided the same self certifying way it is
    on the real staircase: reach the top landing, which no ordinary movement speed can do.

    Three properties of the castle are held rather than re invented, because each one is visible
    in the observation and changing it silently would confound the test. The spawn keeps its world
    coordinates. The climb from spawn to goal is the castle's, so a steeper flight is a shorter
    one rather than a taller one. And the warp is self similar the way the level script's
    ``INSTANT_WARP(0, 2, 0, -205, 410)`` is, displacing Mario by whole treads, which is what makes
    the loop endless rather than merely obstructive. The band is ``warp_treads`` treads deep
    because the castle's 154 unit band is three of its 51.25 unit treads, so the escape speed
    scales with the tread depth instead of being pinned to a constant the geometry does not
    support.

    What does change with rise and run is the depth of the flight along z, since a flight that
    climbs a fixed height with taller steps needs fewer of them. That is the geometry change under
    test and it is reported alongside the result rather than hidden.

    Args:
        rise: Height gained per tread.
        run: Depth of each tread along z.
        climb: Height from the spawn to the goal, defaulting to the castle's.
        warp_treads: Depth of the warp band in treads. Its z extent sets the escape speed.
        loop_treads: Treads the warp displaces Mario back down, the castle's eight.
        warp_height: Height the band starts at, defaulting to the castle's.
        width: Extent of the flight along x, centred on the spawn's x.
        base_height: y of the bottom landing.
        spawn: Episode start, in the castle's own coordinates.
        landing_z: z of the near edge of the first tread.
        risers: Whether each tread gets the vertical face under it that the castle's has. On by
            default because the castle has them, but a riser is as tall as the rise, so a flight
            built from tall steps also has tall walls, and turning them off is the control that
            separates a chain the tread size broke from one a wall stopped.

    Returns:
        A scene ascending toward negative z, like the castle's.

    Raises:
        ValueError: If the flight is too short to hold the warp band and its loop, which would
            leave the staircase climbable and the task no longer self certifying.
    """
    from src.env.geometry import flat_area, staircase

    steps = max(1, round(climb / rise))
    if steps < loop_treads + warp_treads:
        raise ValueError(
            f"rise {rise} climbs {climb} in {steps} treads, too few for a {warp_treads} tread "
            f"band above a {loop_treads} tread loop")

    half = width / 2.0
    goal_y = base_height + rise * steps
    top_z = landing_z - run * steps
    surfaces = flat_area(spawn[0] - half, spawn[0] + half, landing_z, landing_z + 3200.0,
                         base_height)
    surfaces += flat_area(spawn[0] - half, spawn[0] + half, top_z - 2000.0, top_z, goal_y)
    flight = staircase(steps, rise, -run, width, base_height, landing_z, risers=risers)
    for surface in flight:
        surface.vertices[0][0] += int(round(spawn[0]))
        surface.vertices[1][0] += int(round(spawn[0]))
        surface.vertices[2][0] += int(round(spawn[0]))

    first = next(index for index in range(steps) if base_height + rise * (index + 1) >= warp_height)
    band = range(first, min(first + warp_treads, steps))
    heights = {int(round(base_height + rise * (index + 1))) for index in band}
    marked = [surface for surface in flight
              if len({surface.vertices[corner][1] for corner in range(3)}) == 1
              and surface.vertices[0][1] in heights]
    if not marked:
        raise ValueError(f"no treads to mark as warp triangles at heights {sorted(heights)}")
    for surface in marked:
        surface.type = SURFACE_INSTANT_WARP_1B

    warp = WarpZone(
        surface_type=SURFACE_INSTANT_WARP_1B,
        displacement=(0.0, -rise * loop_treads, run * loop_treads),
        x_range=_extent(marked, 0),
        y_range=_extent(marked, 1),
        z_range=_extent(marked, 2),
    )
    return Scene(
        surfaces=surfaces + flight,
        spawn=spawn,
        goal_y=goal_y - 52.0,
        goal_z=top_z - 91.0,
        warp=warp,
        ascends_toward=(0.0, -1.0),
    )
