"""Rasterises a swarm container into a video, without a browser or a GPU.

``tools/make_viewer_swarm.py`` already turns a container into an interactive page, and that page
is the right thing for a reader who wants to scrub. It is the wrong thing for a figure: it is 5.4
MB of WebGL that cannot hand back an image file, so getting a frame out of it means installing a
headless browser and driving ``canvas.toDataURL`` one frame at a time.

This tool takes the other route. A swarm at swarm scale does not need Mario's mesh at all -- at
sixty four bodies in a room two hundred units wide each Mario is a few pixels, and the pose
dictionary's whole 2.5 MB of geometry resolves to a dot. So this draws dots, in one orthographic
side elevation, and spends its pixels on the two things that actually carry the story: where the
bodies are against the staircase, and where they have been.

The projection is ``(z, y)``, not ``(x, y)``. The staircase climbs as z falls -- measured
correlation between z and y along the collision mesh is -0.856 -- so mapping z to screen x with
the axis reversed makes the ascent read left to right and bottom to top, the way a reader already
expects a graph of progress to read. The x axis is the shaft's narrow dimension, 409 units against
1844 of climb, and projecting it would show a wall.

Trails are a decaying float buffer rather than a line list. Every frame the buffer is multiplied
by ``--trail_decay`` and each body stamps into it, so a body that hovers burns in and a body that
crosses the frame in two frames leaves a faint streak, which is exactly the contrast between a
policy that has found the exploit and one that has not. Sixty four line lists would need
per-episode breaks to avoid drawing a stroke across a respawn; a buffer needs no such bookkeeping,
because a reset simply stops stamping in the old place.

Output is rawvideo RGB24 on a pipe to ffmpeg, which is the only dependency beyond numpy, and the
labels are burned in through the same generated-ASS route ``tools/make_ass.py`` uses for the
injected game render, so the two figure families look like they came from one place.
"""

import argparse
import math
import os
import subprocess
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.dump_swarm import decode_stream, read_container

# A checkpoint's name in the grid containers is "<rung>@<steps>M", which is how this tool knows
# which panel a run belongs to and where it sits on the training axis.
NAME_SEPARATOR = "@"

# Status bits, repeated from the container header so a caller can read this file alone. The
# container is still the authority; check_status_bits asserts the two agree.
STATUS_RESET = 0x01
STATUS_SUCCESS = 0x02
STATUS_WARPED = 0x04
STATUS_AIRBORNE = 0x08
STATUS_RECORD_SPEED = 0x10

# Body colours, RGB. Airborne reads amber because in this room being off the ground is the whole
# point; a warp flashes red because it is the failure the exploit exists to beat; a success flashes
# green and is the only colour that outranks the others.
COLOR_GROUNDED = (150, 168, 196)
COLOR_AIRBORNE = (240, 176, 64)
COLOR_WARPED = (232, 72, 72)
COLOR_SUCCESS = (96, 232, 128)
COLOR_RECORD = (255, 246, 210)

# Background, in the same slate the viewers use, plus the two annotations every panel carries.
COLOR_BACKDROP = (14, 17, 24)
COLOR_WARP_BAND = (120, 36, 40)
COLOR_GOAL_LINE = (86, 132, 96)
COLOR_TRAIL = (86, 150, 232)
COLOR_PANEL_EDGE = (44, 52, 66)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Builds the command line.

    Args:
        argv: Arguments without the program name.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--swarm", required=True, help="Container tools/dump_swarm.py wrote.")
    parser.add_argument("--out", required=True, help="Video to write.")
    parser.add_argument("--layout", default="grid", choices=("grid", "single"),
                        help="One panel per rung on a shared step axis, or one panel per run.")
    parser.add_argument("--panel_width", type=int, default=480, help="Panel width in pixels.")
    parser.add_argument("--panel_height", type=int, default=540, help="Panel height in pixels.")
    parser.add_argument("--columns", type=int, default=2, help="Panels per row in a grid.")
    parser.add_argument("--panel_order", default="",
                        help="Comma separated panel names, in placement order. Panels the "
                             "container holds and this does not name are appended, sorted.")
    parser.add_argument("--panel_labels", default="",
                        help="Comma separated display labels, parallel to --panel_order. A panel "
                             "without one is labelled with its own name.")
    parser.add_argument("--caption", default="",
                        help="Line burned along the bottom of the whole video, for the legend.")
    parser.add_argument("--caption_height", type=int, default=34,
                        help="Pixels of strip added below the panels for --caption to sit in, so "
                             "the legend never covers a body.")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    parser.add_argument("--frames_per_run", type=int, default=0,
                        help="Frames to show per run, 0 for every frame the container holds.")
    parser.add_argument("--sample", default="spread", choices=("spread", "prefix"),
                        help="When a run is shown in fewer frames than it holds, spread the "
                             "sampled frames evenly across it or take its prefix.")
    parser.add_argument("--trail_decay", type=float, default=0.90,
                        help="Multiplier applied to the trail buffer each frame.")
    parser.add_argument("--body_radius", type=float, default=3.4, help="Body radius in pixels.")
    parser.add_argument("--crf", type=int, default=20, help="libx264 quality.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary.")
    parser.add_argument("--font", default="/System/Library/Fonts/Supplemental/Andale Mono.ttf",
                        help="Font file for the burned in labels.")
    parser.add_argument("--z_range", type=float, nargs=2, default=None,
                        help="World z the panel spans, as min max. Fitted to the bodies when "
                             "not given.")
    parser.add_argument("--y_range", type=float, nargs=2, default=None,
                        help="World y the panel spans, as min max. Fitted to the bodies when "
                             "not given.")
    parser.add_argument("--margin", type=float, default=90.0,
                        help="World units of padding added around a fitted range.")
    parser.add_argument("--fit_percentile", type=float, default=0.4,
                        help="Percentile trimmed from each end when fitting a range, so one body "
                             "that falls into the void does not shrink every other body to a "
                             "dot. The bodies this drops are counted and reported.")
    parser.add_argument("--x_cull", type=float, nargs=2, default=(-700.0, 300.0),
                        help="World x kept when drawing collision triangles, as min max.")
    return parser.parse_args(argv)


def check_status_bits(header: dict[str, Any]) -> None:
    """Asserts the container's status bits are the ones this file assumes.

    Args:
        header: Parsed container header.

    Raises:
        ValueError: If a bit moved, since every colour decision below would be wrong.
    """
    expected = {
        "reset": STATUS_RESET,
        "success": STATUS_SUCCESS,
        "warped": STATUS_WARPED,
        "airborne": STATUS_AIRBORNE,
        "record_speed": STATUS_RECORD_SPEED,
    }
    if header.get("status_bits") != expected:
        raise ValueError(f"container status bits {header.get('status_bits')} are not {expected}")


def fit_ranges(runs: list["Run"], header: dict[str, Any], margin: float,
               percentile: float = 0.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fits the panel's world ranges to where the bodies actually are.

    Fitting matters more than it sounds. The staircase's own collision extent stops at z -1091,
    but the spawn sits at z 3000 with up to ``--spawn_spread`` of jitter, so a panel framed on the
    geometry alone drops every body off the left edge for the first second of every episode --
    silently, because a body outside the panel is simply not stamped.

    Fitting to the outright extremes is its own trap in the other direction: a single Mario who
    walks off the staircase falls to y 1113, two thousand units below anything else in the room,
    and framing on him spends half the panel's height on empty shaft. So the fit trims a
    percentile off each end, and :func:`count_offscreen` says how many bodies that cost.

    Trimming introduces its own hazard, though, which is that the two annotations a reader needs
    -- the warp band and the landing -- are thresholds nobody may have reached yet. A panel fitted
    to an early checkpoint's bodies alone puts the landing off the top edge, and then the figure
    has no destination in it. So the fitted range is unioned with the warp band and the goal
    before it is used, and those are never trimmed.

    Args:
        runs: Every run to be drawn.
        header: Parsed container header, for the warp band and the goal.
        margin: World units of padding to add on each side.
        percentile: Percentile to trim from each end, 0 for the outright extremes.

    Returns:
        A pair of the z range and the y range, each as min then max.
    """
    warp, goal = header["warp"], header["goal"]
    required = {
        1: [warp["y_range"][0], warp["y_range"][1], goal["goal_y"]],
        2: [warp["z_range"][0], warp["z_range"][1],
            goal["goal_z"] - goal["tolerance"], goal["goal_z"] + goal["tolerance"]],
    }
    axes = {}
    for axis in (1, 2):
        samples = np.concatenate([run.position[:, :, axis].ravel() for run in runs])
        if percentile > 0.0:
            lo, hi = np.percentile(samples, [percentile, 100.0 - percentile])
        else:
            lo, hi = samples.min(), samples.max()
        lo = min(float(lo), min(required[axis]))
        hi = max(float(hi), max(required[axis]))
        axes[axis] = (lo - margin, hi + margin)
    return axes[2], axes[1]


def count_offscreen(runs: list["Run"], projection: "Projection") -> int:
    """Counts body samples that fall outside the panel.

    Args:
        runs: Every run to be drawn.
        projection: The panel projection.

    Returns:
        How many ``(frame, slot)`` samples project outside the panel's pixels.
    """
    total = 0
    for run in runs:
        columns, rows = projection.to_pixels(run.position[:, :, 2], run.position[:, :, 1])
        outside = ((columns < 0) | (columns >= projection.width)
                   | (rows < 0) | (rows >= projection.height))
        total += int(outside.sum())
    return total


class Projection:
    """Maps sm64 world space onto one panel's pixels.

    The mapping is orthographic and axis aligned, ``z`` to screen x with the axis reversed so the
    climb reads left to right, and ``y`` to screen y with the axis reversed so up is up.
    """

    def __init__(self, z_range: tuple[float, float], y_range: tuple[float, float],
                 width: int, height: int) -> None:
        """Builds a projection.

        Args:
            z_range: World z the panel spans, as min then max.
            y_range: World y the panel spans, as min then max.
            width: Panel width in pixels.
            height: Panel height in pixels.
        """
        self.z_min, self.z_max = float(z_range[0]), float(z_range[1])
        self.y_min, self.y_max = float(y_range[0]), float(y_range[1])
        self.width, self.height = int(width), int(height)

    def to_pixels(self, z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Projects world coordinates.

        Args:
            z: World z values.
            y: World y values.

        Returns:
            A pair of float pixel columns and rows, unclipped.
        """
        u = (self.z_max - np.asarray(z, dtype=np.float64)) / (self.z_max - self.z_min)
        v = (self.y_max - np.asarray(y, dtype=np.float64)) / (self.y_max - self.y_min)
        return u * self.width, v * self.height


def fill_triangle(canvas: np.ndarray, points: np.ndarray, color: tuple[int, int, int],
                  weight: float = 1.0) -> None:
    """Fills one projected triangle into a panel, with no antialiasing.

    A scanline fill over the triangle's own bounding box is enough here: the staircase is 355
    triangles inside the cull box and the background is drawn once per panel, not once per frame.

    Args:
        canvas: ``(height, width, 3)`` float32 panel, modified in place.
        points: ``(3, 2)`` array of pixel columns and rows.
        color: RGB to blend toward.
        weight: Blend weight in 0..1.
    """
    height, width = canvas.shape[:2]
    x_lo = max(int(math.floor(points[:, 0].min())), 0)
    x_hi = min(int(math.ceil(points[:, 0].max())) + 1, width)
    y_lo = max(int(math.floor(points[:, 1].min())), 0)
    y_hi = min(int(math.ceil(points[:, 1].max())) + 1, height)
    if x_hi <= x_lo or y_hi <= y_lo:
        return
    xs = np.arange(x_lo, x_hi, dtype=np.float64) + 0.5
    ys = np.arange(y_lo, y_hi, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    (ax, ay), (bx, by), (cx, cy) = points
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(area) < 1e-9:
        return
    w0 = ((bx - ax) * (grid_y - ay) - (grid_x - ax) * (by - ay)) / area
    w1 = ((grid_x - ax) * (cy - ay) - (cx - ax) * (grid_y - ay)) / area
    inside = (w0 >= 0.0) & (w1 >= 0.0) & (w0 + w1 <= 1.0)
    if not inside.any():
        return
    patch = canvas[y_lo:y_hi, x_lo:x_hi]
    tint = np.asarray(color, dtype=np.float32)
    patch[inside] = patch[inside] * (1.0 - weight) + tint * weight


def fill_rect(canvas: np.ndarray, x_lo: float, y_lo: float, x_hi: float, y_hi: float,
              color: tuple[int, int, int], weight: float = 1.0) -> None:
    """Blends an axis aligned rectangle into a panel.

    Args:
        canvas: ``(height, width, 3)`` float32 panel, modified in place.
        x_lo: Left pixel column.
        y_lo: Top pixel row.
        x_hi: Right pixel column.
        y_hi: Bottom pixel row.
        color: RGB to blend toward.
        weight: Blend weight in 0..1.
    """
    height, width = canvas.shape[:2]
    left = max(int(math.floor(min(x_lo, x_hi))), 0)
    right = min(int(math.ceil(max(x_lo, x_hi))), width)
    top = max(int(math.floor(min(y_lo, y_hi))), 0)
    bottom = min(int(math.ceil(max(y_lo, y_hi))), height)
    if right <= left or bottom <= top:
        return
    patch = canvas[top:bottom, left:right]
    patch[...] = patch * (1.0 - weight) + np.asarray(color, dtype=np.float32) * weight


def draw_backdrop(header: dict[str, Any], payload: bytes, projection: Projection,
                  x_cull: tuple[float, float]) -> np.ndarray:
    """Rasterises the staircase once, as a panel background.

    The collision mesh is tinted by what a surface does rather than by where it points: a tread is
    what Mario stands on and reads bright, a riser reads dim, and the twelve
    ``SURFACE_INSTANT_WARP_1B`` triangles read red, which is the same convention the WebGL viewers
    use so a reader who has seen one recognises the other.

    Args:
        header: Parsed container header.
        payload: Container payload bytes.
        projection: Panel projection.
        x_cull: World x kept, as min then max.

    Returns:
        A ``(height, width, 3)`` float32 panel.
    """
    canvas = np.zeros((projection.height, projection.width, 3), dtype=np.float32)
    canvas[...] = np.asarray(COLOR_BACKDROP, dtype=np.float32)

    warp = header["warp"]
    band_x0, band_y0 = projection.to_pixels(np.array([projection.z_max]),
                                            np.array([warp["y_range"][1]]))
    band_x1, band_y1 = projection.to_pixels(np.array([projection.z_min]),
                                            np.array([warp["y_range"][0]]))
    fill_rect(canvas, band_x0[0], band_y0[0], band_x1[0], band_y1[0], COLOR_WARP_BAND, 0.55)

    vertices = decode_stream(header["streams"]["surface_vertices"], payload)
    vertices = vertices.reshape(-1, 3, 3).astype(np.float64)
    types = decode_stream(header["streams"]["surface_types"], payload)
    centroid = vertices.mean(axis=1)
    keep = ((centroid[:, 0] >= x_cull[0]) & (centroid[:, 0] <= x_cull[1])
            & (centroid[:, 1] >= projection.y_min - 200.0)
            & (centroid[:, 1] <= projection.y_max + 200.0)
            & (centroid[:, 2] >= projection.z_min - 200.0)
            & (centroid[:, 2] <= projection.z_max + 200.0))
    edge_a = vertices[:, 1] - vertices[:, 0]
    edge_b = vertices[:, 2] - vertices[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1)
    upness = np.where(lengths > 1e-9, normals[:, 1] / np.maximum(lengths, 1e-9), 0.0)
    for index in np.nonzero(keep)[0]:
        columns, rows = projection.to_pixels(vertices[index, :, 2], vertices[index, :, 1])
        if int(types[index]) == int(warp["surface_type"]):
            color, weight = (198, 64, 64), 0.85
        elif abs(float(upness[index])) > 0.7:
            color, weight = (96, 106, 126), 0.9
        else:
            color, weight = (44, 52, 68), 0.9
        fill_triangle(canvas, np.stack([columns, rows], axis=1), color, weight)

    goal_columns, goal_rows = projection.to_pixels(
        np.array([projection.z_min, projection.z_max]),
        np.array([header["goal"]["goal_y"], header["goal"]["goal_y"]]))
    row = float(goal_rows[0])
    for start in range(0, projection.width, 14):
        fill_rect(canvas, start, row - 1.0, start + 8, row + 1.0, COLOR_GOAL_LINE, 0.85)
    fill_rect(canvas, 0, 0, projection.width, 1, COLOR_PANEL_EDGE, 1.0)
    fill_rect(canvas, 0, projection.height - 1, projection.width, projection.height,
              COLOR_PANEL_EDGE, 1.0)
    fill_rect(canvas, 0, 0, 1, projection.height, COLOR_PANEL_EDGE, 1.0)
    fill_rect(canvas, projection.width - 1, 0, projection.width, projection.height,
              COLOR_PANEL_EDGE, 1.0)
    return canvas


def body_stencil(radius: float) -> np.ndarray:
    """Builds a soft disc the bodies are stamped with.

    Args:
        radius: Disc radius in pixels.

    Returns:
        A square float32 array of coverage in 0..1, with an odd side length.
    """
    reach = int(math.ceil(radius)) + 1
    axis = np.arange(-reach, reach + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis)
    distance = np.hypot(grid_x, grid_y)
    return np.clip(radius + 0.5 - distance, 0.0, 1.0).astype(np.float32)


def stamp(canvas: np.ndarray, stencil: np.ndarray, column: float, row: float,
          color: tuple[int, int, int], weight: float = 1.0) -> None:
    """Stamps one disc into a panel.

    Args:
        canvas: ``(height, width, 3)`` float32 panel, modified in place.
        stencil: Coverage from :func:`body_stencil`.
        column: Centre pixel column.
        row: Centre pixel row.
        color: RGB to blend toward.
        weight: Extra blend weight in 0..1.
    """
    height, width = canvas.shape[:2]
    reach = stencil.shape[0] // 2
    left, top = int(round(column)) - reach, int(round(row)) - reach
    x_lo, y_lo = max(left, 0), max(top, 0)
    x_hi, y_hi = min(left + stencil.shape[1], width), min(top + stencil.shape[0], height)
    if x_hi <= x_lo or y_hi <= y_lo:
        return
    cut = stencil[y_lo - top:y_hi - top, x_lo - left:x_hi - left, None] * weight
    patch = canvas[y_lo:y_hi, x_lo:x_hi]
    patch[...] = patch * (1.0 - cut) + np.asarray(color, dtype=np.float32) * cut


def stamp_trail(trail: np.ndarray, stencil: np.ndarray, column: float, row: float) -> None:
    """Stamps one disc into the single channel trail buffer.

    Args:
        trail: ``(height, width)`` float32 buffer, modified in place.
        stencil: Coverage from :func:`body_stencil`.
        column: Centre pixel column.
        row: Centre pixel row.
    """
    height, width = trail.shape
    reach = stencil.shape[0] // 2
    left, top = int(round(column)) - reach, int(round(row)) - reach
    x_lo, y_lo = max(left, 0), max(top, 0)
    x_hi, y_hi = min(left + stencil.shape[1], width), min(top + stencil.shape[0], height)
    if x_hi <= x_lo or y_hi <= y_lo:
        return
    cut = stencil[y_lo - top:y_hi - top, x_lo - left:x_hi - left]
    np.maximum(trail[y_lo:y_hi, x_lo:x_hi], cut, out=trail[y_lo:y_hi, x_lo:x_hi])


class Run:
    """One checkpoint's capture, decoded and ready to draw."""

    def __init__(self, header: dict[str, Any], payload: bytes, index: int) -> None:
        """Decodes one run's per frame streams.

        Args:
            header: Parsed container header.
            payload: Container payload bytes.
            index: Run index, matching the ``run<index>/`` stream prefix.
        """
        self.meta = header["checkpoints"][index]
        prefix = f"run{index}/"
        shape = header["streams"][prefix + "position"]["shape"]
        self.frames, self.population = int(shape[0]), int(shape[1])
        scale = float(header["position_scale"])
        position = decode_stream(header["streams"][prefix + "position"], payload)
        self.position = position.reshape(self.frames, self.population, 3) / scale
        status = decode_stream(header["streams"][prefix + "status"], payload)
        self.status = status.reshape(self.frames, self.population).astype(np.int64)

    @property
    def name(self) -> str:
        """Returns the checkpoint's name as the dump recorded it."""
        return str(self.meta["name"])

    @property
    def panel(self) -> str:
        """Returns the panel this run belongs to, from the part of the name before the separator."""
        return self.name.split(NAME_SEPARATOR)[0]

    @property
    def steps(self) -> int:
        """Returns the checkpoint's training step count."""
        return int(self.meta["steps"])


def order_panels(found: set[str], requested: str) -> list[str]:
    """Places panels in the order a figure wants to be read in.

    Sorting by name puts ``height`` first, which is the rung that never finds the exploit, and a
    figure that opens on the negative result reads backwards. So the caller names the order and
    anything it forgot is appended rather than dropped.

    Args:
        found: Panel names the container holds.
        requested: Comma separated names, in placement order, possibly empty.

    Returns:
        The panel names in placement order.

    Raises:
        ValueError: If a requested name is not in the container.
    """
    wanted = [name.strip() for name in requested.split(",") if name.strip()]
    missing = [name for name in wanted if name not in found]
    if missing:
        raise ValueError(f"container has no panel named {missing}, it has {sorted(found)}")
    return wanted + sorted(found - set(wanted))


def group_runs(runs: list[Run], layout: str,
               panel_order: str = "") -> tuple[list[str], list[list[Run]]]:
    """Arranges runs into panels and a shared timeline.

    In ``grid`` layout the name before the separator names the panel and the step count orders the
    timeline, so four rungs captured on the same step grid advance together and the reader is
    comparing like with like. In ``single`` layout every run is its own panel and the timeline is
    one step long, which is how a converged-behaviour figure is drawn.

    Args:
        runs: Every run in the container.
        layout: Either ``grid`` or ``single``.
        panel_order: Comma separated panel names, in placement order, possibly empty.

    Returns:
        A pair of panel names and, per timeline slot, the run to draw in each panel.

    Raises:
        ValueError: If a grid layout's panels do not share one step grid.
    """
    if layout == "single":
        return [run.name for run in runs], [runs]
    panels = order_panels({run.panel for run in runs}, panel_order)
    by_panel = {panel: sorted((r for r in runs if r.panel == panel), key=lambda r: r.steps)
                for panel in panels}
    grids = {panel: tuple(run.steps for run in group) for panel, group in by_panel.items()}
    if len({grid for grid in grids.values()}) != 1:
        raise ValueError(f"panels do not share a step grid: "
                         f"{ {panel: len(grid) for panel, grid in grids.items()} }")
    slots = list(zip(*(by_panel[panel] for panel in panels), strict=True))
    return panels, [list(slot) for slot in slots]


def sample_frames(available: int, wanted: int, mode: str) -> list[int]:
    """Chooses which of a run's frames the output shows.

    A fourteen checkpoint time lapse cannot afford every frame of every capture -- four panels of
    three hundred frames is a hundred and forty seconds of video for a figure nobody watches to
    the end. The prefix is the wrong economy, though: an episode spends its first second walking
    away from the spawn, so a panel cut to its first ninety frames shows four rungs that all look
    identical and none of them doing anything. Spreading the samples across the whole capture
    keeps the part where the policies differ.

    Args:
        available: Frames the capture holds.
        wanted: Frames the output shows.
        mode: Either ``spread`` or ``prefix``.

    Returns:
        ``wanted`` frame indices into the capture, non decreasing.
    """
    if wanted >= available or mode == "prefix":
        return [min(index, available - 1) for index in range(wanted)]
    step = (available - 1) / float(wanted - 1) if wanted > 1 else 0.0
    return [int(round(index * step)) for index in range(wanted)]


def escape_ass(text: str) -> str:
    """Escapes text for an ASS dialogue line.

    Args:
        text: Raw label text.

    Returns:
        The text with the characters libass reads as markup neutralised.
    """
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def write_labels(path: str, panels: list[str], timeline: list[list[Run]], per_run: int,
                 fps: int, panel_width: int, panel_height: int, columns: int,
                 width: int, height: int, font: str, labels: list[str] | None = None,
                 caption: str = "", strip: int = 0) -> None:
    """Writes the ASS track that burns each panel's label and counters onto the video.

    Args:
        path: ASS file to write.
        panels: Panel names, in placement order.
        timeline: Per timeline slot, the run drawn in each panel.
        per_run: Frames each timeline slot occupies.
        fps: Output frame rate.
        panel_width: Panel width in pixels.
        panel_height: Panel height in pixels.
        columns: Panels per row.
        width: Output width in pixels.
        height: Output height in pixels.
        font: Font file, whose basename names the ASS font.
        labels: Display labels parallel to ``panels``, or None to use the panel names.
        caption: Line burned along the bottom of the whole video, possibly empty.
        strip: Pixels of strip below the panels that the caption sits in.
    """
    family = os.path.splitext(os.path.basename(font))[0]
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: head,{family},20,&H00FFFFFF,&H00FFFFFF,&HC0140E0A,&HC0140E0A,"
        "1,0,0,0,100,100,0,0,3,6,0,7,0,0,0,1",
        f"Style: body,{family},15,&H00D8E4F4,&H00D8E4F4,&HC0140E0A,&HC0140E0A,"
        "0,0,0,0,100,100,0,0,3,6,0,7,0,0,0,1",
        f"Style: hit,{family},15,&H0080E860,&H0080E860,&HC0140E0A,&HC0140E0A,"
        "0,0,0,0,100,100,0,0,3,6,0,7,0,0,0,1",
        f"Style: legend,{family},15,&H00B0BCCC,&H00B0BCCC,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,0,0,2,0,0,{max(strip // 4, 4)},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def stamp_time(frame: int) -> str:
        total = frame / float(fps)
        hours, rest = divmod(total, 3600.0)
        minutes, seconds = divmod(rest, 60.0)
        return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"

    for slot, runs in enumerate(timeline):
        start, end = stamp_time(slot * per_run), stamp_time((slot + 1) * per_run)
        for index, run in enumerate(runs):
            column, row = index % columns, index // columns
            left, top = column * panel_width, row * panel_height
            successes = int((run.status & STATUS_SUCCESS).astype(bool).any(axis=0).sum())
            warps = int((run.status & STATUS_WARPED).astype(bool).sum())
            outcome = run.meta.get("outcome") or {}
            shown = (labels[index] if labels and len(labels) > index
                     else (panels[index] if len(panels) > index else run.name))
            head = escape_ass(shown)
            steps = f"{run.steps / 1e6:.1f}M steps"
            body = (f"{steps}\\N{outcome.get('episodes', 0)} episodes, "
                    f"{outcome.get('successes', 0)} reached the landing\\N"
                    f"peak backward {outcome.get('best_peak_backward', 0.0):.0f}\\N"
                    f"warps this clip {warps}")
            lines.append(f"Dialogue: 0,{start},{end},head,,0,0,0,,"
                         f"{{\\pos({left + 14},{top + 60})}}{head}")
            style = "hit" if successes else "body"
            lines.append(f"Dialogue: 0,{start},{end},{style},,0,0,0,,"
                         f"{{\\pos({left + 14},{top + 88})}}{body}")
    if caption:
        lines.append(f"Dialogue: 0,{stamp_time(0)},{stamp_time(len(timeline) * per_run)},"
                     f"legend,,0,0,0,,{escape_ass(caption)}")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def render(args: argparse.Namespace) -> dict[str, Any]:
    """Renders a container to a video.

    Args:
        args: Parsed command line.

    Returns:
        A dict of what was drawn, for the caller to log.

    Raises:
        RuntimeError: If ffmpeg exits non zero.
    """
    header, payload = read_container(args.swarm)
    check_status_bits(header)
    runs = [Run(header, payload, index) for index in range(len(header["checkpoints"]))]
    panels, timeline = group_runs(runs, args.layout, args.panel_order)
    columns = min(args.columns, len(panels))
    rows = int(math.ceil(len(panels) / columns))
    strip = args.caption_height if args.caption else 0
    width = columns * args.panel_width
    height = rows * args.panel_height + strip

    fitted_z, fitted_y = fit_ranges(runs, header, args.margin, args.fit_percentile)
    z_range = tuple(args.z_range) if args.z_range else fitted_z
    y_range = tuple(args.y_range) if args.y_range else fitted_y
    projection = Projection(z_range, y_range, args.panel_width, args.panel_height)
    offscreen = count_offscreen(runs, projection)
    x_cull = tuple(args.x_cull)
    backdrop = draw_backdrop(header, payload, projection, x_cull)
    stencil = body_stencil(args.body_radius)
    trail_stencil = body_stencil(max(args.body_radius - 1.2, 1.0))

    per_run = args.frames_per_run or min(run.frames for run in runs)
    shown = [name.strip() for name in args.panel_labels.split(",") if name.strip()]
    labels = os.path.splitext(args.out)[0] + ".ass"
    write_labels(labels, panels, timeline, per_run, args.fps, args.panel_width,
                 args.panel_height, columns, width, height, args.font, shown or None,
                 args.caption, strip)

    command = [
        args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-framerate", str(args.fps), "-i", "pipe:0",
        "-vf", f"ass={labels}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
        "-movflags", "+faststart", args.out,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    frame_buffer = np.zeros((height, width, 3), dtype=np.float32)
    drawn = 0
    try:
        for runs_in_slot in timeline:
            trails = [np.zeros((args.panel_height, args.panel_width), dtype=np.float32)
                      for _ in runs_in_slot]
            picks = [sample_frames(run.frames, per_run, args.sample) for run in runs_in_slot]
            for frame in range(per_run):
                for index, run in enumerate(runs_in_slot):
                    column, row = index % columns, index // columns
                    left, top = column * args.panel_width, row * args.panel_height
                    source = picks[index][frame]
                    status = run.status[source]
                    columns_px, rows_px = projection.to_pixels(run.position[source, :, 2],
                                                              run.position[source, :, 1])
                    trail = trails[index]
                    trail *= args.trail_decay
                    live = (status & STATUS_RESET) == 0
                    for slot in np.nonzero(live)[0]:
                        stamp_trail(trail, trail_stencil, columns_px[slot], rows_px[slot])
                    panel = backdrop.copy()
                    faded = np.clip(trail, 0.0, 1.0)[:, :, None] * 0.65
                    panel[...] = (panel * (1.0 - faded)
                                  + np.asarray(COLOR_TRAIL, dtype=np.float32) * faded)
                    for slot in range(run.population):
                        bits = int(status[slot])
                        if bits & STATUS_SUCCESS:
                            color, weight = COLOR_SUCCESS, 1.0
                        elif bits & STATUS_WARPED:
                            color, weight = COLOR_WARPED, 1.0
                        elif bits & STATUS_RECORD_SPEED:
                            color, weight = COLOR_RECORD, 1.0
                        elif bits & STATUS_AIRBORNE:
                            color, weight = COLOR_AIRBORNE, 0.95
                        else:
                            color, weight = COLOR_GROUNDED, 0.9
                        stamp(panel, stencil, columns_px[slot], rows_px[slot], color, weight)
                    frame_buffer[top:top + args.panel_height,
                                 left:left + args.panel_width] = panel
                process.stdin.write(np.clip(frame_buffer, 0.0, 255.0)
                                    .astype(np.uint8).tobytes())
                drawn += 1
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code}")
    return {
        "out": args.out,
        "panels": panels,
        "timeline": len(timeline),
        "frames": drawn,
        "seconds": round(drawn / float(args.fps), 2),
        "size": f"{width}x{height}",
        "bytes": os.path.getsize(args.out),
        "z_range": [round(value, 1) for value in z_range],
        "y_range": [round(value, 1) for value in y_range],
        "offscreen_samples": offscreen,
        "body_samples": sum(run.frames * run.population for run in runs),
    }


def main(argv: list[str]) -> int:
    """Renders a swarm container.

    Args:
        argv: Arguments without the program name.

    Returns:
        A process exit code.
    """
    args = parse_args(argv)
    result = render(args)
    print(f"{result['out']}  {result['size']}  {result['frames']} frames  "
          f"{result['seconds']}s  {result['bytes'] // 1024} KB")
    print(f"panels: {', '.join(result['panels'])}  timeline slots: {result['timeline']}")
    print(f"z {result['z_range']}  y {result['y_range']}")
    share = 100.0 * result["offscreen_samples"] / max(result["body_samples"], 1)
    print(f"bodies outside the panel: {result['offscreen_samples']} of "
          f"{result['body_samples']} samples ({share:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
