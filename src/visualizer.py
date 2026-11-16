"""
Tasks 1, 3, 5 & 6 -- Publication-quality visualisation
======================================================

All figures are rendered with the non-interactive ``Agg`` backend, so the
project runs headless (e.g. on a lab server or in CI) while still producing
300 DPI PNG files inside ``output/``:

1. ``01_field_environment.png``           -- colour-coded field + legend
2. ``02_algorithm_paths_comparison.png``  -- BFS / DFS / Greedy / A* (2x2)
3. ``03_crop_safe_cost_effect.png``       -- shortest-step vs crop-safe route
4. ``04_dynamic_replanning_simulation.png`` -- initial plan, detections, replans, executed path
"""

from __future__ import annotations

import textwrap
from pathlib import Path as FilePath
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless rendering -- must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .algorithms import SearchResult
from .environment import (
    CAUTION,
    CROP,
    DYNAMIC,
    GOAL,
    SAFE,
    START,
    STATIC,
    CropFieldEnvironment,
)
from .problem import Cell, CropFieldProblem

# --------------------------------------------------------------------------- #
# Shared styling
# --------------------------------------------------------------------------- #
COLORS: Dict[int, str] = {
    SAFE: "#fbfbf5",
    CROP: "#2e7d32",
    CAUTION: "#ffe082",
    STATIC: "#455a64",
    DYNAMIC: "#c62828",
    START: "#1565c0",
    GOAL: "#6a1b9a",
}

LEGEND_LABELS: Dict[int, str] = {
    SAFE: "Safe furrow / lane (cost 1)",
    CROP: "Potato row - protected core (blocked)",
    CAUTION: "Crop-proximity buffer (cost 5)",
    STATIC: "Static obstacle (blocked)",
    DYNAMIC: "Hidden dynamic obstacle",
    START: "Start S",
    GOAL: "Goal G",
}

PATH_COLOR = "#d81b60"
EXPLORED_COLOR = "#90a4ae"
DPI = 300


def _rgb_from_grid(grid: np.ndarray) -> np.ndarray:
    """Convert the integer grid into an RGB image array for ``imshow``."""
    rgb = np.zeros((*grid.shape, 3), dtype=float)
    for value, color in COLORS.items():
        rgb[grid == value] = matplotlib.colors.to_rgb(color)
    return rgb


def _draw_field(
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    show_hidden: bool = True,
) -> None:
    """Draw the cell colours, the grid lines, the coordinate ticks and S/G."""
    rows, cols = grid.shape
    display = grid.copy()
    if not show_hidden:
        display[display == DYNAMIC] = SAFE

    ax.imshow(
        _rgb_from_grid(display),
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, cols - 0.5, rows - 0.5, -0.5),
        zorder=0,
    )

    # grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#9e9e9e", linewidth=0.35, alpha=0.7)
    ax.tick_params(which="minor", length=0)

    # coordinate ticks (every cell, small font)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(axis="both", labelsize=5.5, length=2)
    ax.set_xlabel("Column index (c)", fontsize=8)
    ax.set_ylabel("Row index (r)", fontsize=8)

    # S and G glyphs
    positions = {
        START: np.argwhere(grid == START),
        GOAL: np.argwhere(grid == GOAL),
    }
    for value, coords in positions.items():
        for row, col in coords:
            ax.text(
                col, row, "S" if value == START else "G",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white",
            )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#616161")


def _legend_handles(include_hidden: bool = True) -> List[Patch]:
    """Legend proxies for the seven cell types."""
    keys = [SAFE, CROP, CAUTION, STATIC]
    if include_hidden:
        keys.append(DYNAMIC)
    keys += [START, GOAL]
    return [
        Patch(facecolor=COLORS[key], edgecolor="#424242", label=LEGEND_LABELS[key])
        for key in keys
    ]


def _plot_path(
    ax: plt.Axes,
    path: Sequence[Cell],
    color: str = PATH_COLOR,
    label: str = "Planned path",
    linestyle: str = "-",
    linewidth: float = 2.0,
    zorder: int = 3,
) -> None:
    """Draw a path using (row, col) -> (x=col, y=row) coordinates."""
    if not path:
        return
    cols = [cell[1] for cell in path]
    rows = [cell[0] for cell in path]
    ax.plot(
        cols, rows, linestyle=linestyle, color=color, linewidth=linewidth,
        marker="o", markersize=2.2, markerfacecolor="white",
        markeredgewidth=0.5, label=label, zorder=zorder,
    )


def _save(fig: plt.Figure, out_path: str | FilePath) -> str:
    out = FilePath(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Figure 1 -- Task 1 field environment
# --------------------------------------------------------------------------- #
def plot_environment(
    env: CropFieldEnvironment,
    out_path: str | FilePath = "output/01_field_environment.png",
) -> str:
    """Colour-coded field with legend, coordinates and cell statistics."""
    fig, ax = plt.subplots(figsize=(9.6, 8.6))
    _draw_field(
        ax,
        env.ground_truth,
        "Task 1 -- Potato Field Environment (20 x 20, ground truth)",
    )
    ax.legend(
        handles=_legend_handles(include_hidden=True),
        loc="upper center", bbox_to_anchor=(0.5, -0.09),
        ncol=4, fontsize=7.5, frameon=True, framealpha=0.95,
    )

    # mark the planned hidden hazards with a "?" glyph
    for row, col in env.hidden_obstacles:
        ax.text(
            col, row, "?", ha="center", va="center", fontsize=7,
            fontweight="bold", color="white",
        )

    stats = textwrap.fill(env.describe(), width=96)
    fig.text(0.5, 1.01, stats, ha="center", va="bottom", fontsize=8,
             color="#424242")
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 2 -- Task 3 algorithm comparison
# --------------------------------------------------------------------------- #
def plot_algorithm_comparison(
    env: CropFieldEnvironment,
    problem: CropFieldProblem,
    results: Sequence[SearchResult],
    out_path: str | FilePath = "output/02_algorithm_paths_comparison.png",
) -> str:
    """2 x 2 comparison of the four algorithms (expanded nodes + path)."""
    order = ["BFS", "DFS", "Greedy BFS", "A*"]
    by_name = {result.algorithm_name: result for result in results}

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 12.2))
    for ax, name in zip(axes.flat, order):
        result = by_name[name]
        _draw_field(
            ax,
            env.robot_map,
            (
                f"{name}\n"
                f"steps = {result.path_length_steps}   |   "
                f"crop-safe cost = {result.total_path_cost:.1f}   |   "
                f"expanded = {result.expanded_nodes_count}"
            ),
            show_hidden=False,
        )

        if result.expanded_order:
            exp_cols = [cell[1] for cell in result.expanded_order]
            exp_rows = [cell[0] for cell in result.expanded_order]
            ax.scatter(
                exp_cols, exp_rows, s=26, marker="s", color="#78909c",
                alpha=0.55, linewidths=0, label="Expanded nodes", zorder=2,
            )
        _plot_path(ax, result.path, label="Solution path")
        ax.legend(
            handles=[
                Patch(facecolor=COLORS[SAFE], edgecolor="#424242", label="Safe lane"),
                Patch(facecolor=COLORS[CAUTION], edgecolor="#424242", label="Caution (cost 5)"),
                Patch(facecolor=COLORS[CROP], edgecolor="#424242", label="Potato row (blocked)"),
                Patch(facecolor=COLORS[STATIC], edgecolor="#424242", label="Static obstacle"),
                Patch(facecolor=EXPLORED_COLOR, label="Expanded nodes"),
                Patch(facecolor=PATH_COLOR, label="Solution path"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.10),
            ncol=2, fontsize=6.5, framealpha=0.95,
        )

    fig.suptitle(
        "Task 3 -- Search Algorithm Comparison on the Robot's Known Map",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 3 -- Task 4 crop-safe cost effect
# --------------------------------------------------------------------------- #
def plot_cost_effect(
    env: CropFieldEnvironment,
    problem: CropFieldProblem,
    shortest_step_result: SearchResult,
    crop_safe_result: SearchResult,
    out_path: str | FilePath = "output/03_crop_safe_cost_effect.png",
) -> str:
    """Shortest-step route vs crop-safe A* route, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
    panels = (
        (
            shortest_step_result,
            "Standard shortest-step path (BFS: ignores crop penalty)",
            PATH_COLOR,
        ),
        (
            crop_safe_result,
            "Crop-safe A* path (cost-aware, keeps away from plants)",
            "#1b5e20",
        ),
    )

    for ax, (result, title, color) in zip(axes, panels):
        stats = problem.count_cells(result.path)
        _draw_field(
            ax,
            env.robot_map,
            (
                f"{title}\n"
                f"steps = {result.path_length_steps}   |   "
                f"crop-safe cost = {result.total_path_cost:.1f}   |   "
                f"caution cells entered = {stats['caution']}"
            ),
            show_hidden=False,
        )
        _plot_path(ax, result.path, color=color, label="Route", linewidth=2.4)

        # highlight every crop-proximity cell the robot actually enters
        caution_entries = [cell for cell in result.path[1:] if int(env.robot_map[cell]) == CAUTION]
        if caution_entries:
            ax.scatter(
                [cell[1] for cell in caution_entries],
                [cell[0] for cell in caution_entries],
                s=95, marker="o", facecolor="none", edgecolor="#e65100",
                linewidth=1.6, label="Caution cell entered (cost 5)", zorder=4,
            )

        ax.legend(
            handles=[
                Patch(facecolor=COLORS[SAFE], edgecolor="#424242", label="Safe lane (cost 1)"),
                Patch(facecolor=COLORS[CAUTION], edgecolor="#424242", label="Caution (cost 5)"),
                Patch(facecolor=COLORS[CROP], edgecolor="#424242", label="Potato row (blocked)"),
                Patch(facecolor=color, label="Route"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.10),
            ncol=2, fontsize=7, framealpha=0.95,
        )

    saved_cost = (
        f"cost saving = {shortest_step_result.total_path_cost - crop_safe_result.total_path_cost:.1f} "
        f"({(1 - crop_safe_result.total_path_cost / max(shortest_step_result.total_path_cost, 1e-9)) * 100:.1f}%)"
    )
    fig.suptitle(
        f"Task 4 -- Effect of the Crop-Safe Cost Function   ({saved_cost})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Figure 4 -- Task 5 dynamic replanning
# --------------------------------------------------------------------------- #
def plot_dynamic_replanning(
    env: CropFieldEnvironment,
    dynamic_result,
    out_path: str | FilePath = "output/04_dynamic_replanning_simulation.png",
) -> str:
    """Initial plan, detected hazards, replanning triggers, executed trajectory."""
    fig, ax = plt.subplots(figsize=(9.8, 9.0))
    _draw_field(
        ax,
        env.ground_truth,
        (
            "Task 5 -- Online Obstacle Detection & Dynamic Replanning\n"
            f"replans = {dynamic_result.total_replans}   |   "
            f"executed steps = {dynamic_result.traversed_steps}   |   "
            f"reached goal = {'Yes' if dynamic_result.reached_goal else 'No'}"
        ),
    )

    _plot_path(
        ax, dynamic_result.initial_planned_path,
        color="#1565c0", label="Initial planned path (A*)",
        linestyle="--", linewidth=1.9, zorder=3,
    )
    _plot_path(
        ax, dynamic_result.actual_traversed_path,
        color="#1b5e20", label="Final executed trajectory",
        linestyle="-", linewidth=2.6, zorder=5,
    )

    if dynamic_result.detected_obstacles:
        ax.scatter(
            [cell[1] for cell in dynamic_result.detected_obstacles],
            [cell[0] for cell in dynamic_result.detected_obstacles],
            s=210, marker="X", color="#c62828", edgecolor="white",
            linewidth=1.2, label="Detected unknown obstacle", zorder=6,
        )
        for row, col in dynamic_result.detected_obstacles:
            # keep the label inside the axes: shift it right, above for the
            # upper markers and below for the lower ones
            above = row < dynamic_result.detected_obstacles[-1][0]
            offset = (14, 12) if above else (14, -20)
            ax.annotate(
                "detected obstacle", (col, row), textcoords="offset points",
                xytext=offset, fontsize=7, color="#b71c1c",
                arrowprops=dict(arrowstyle="-", color="#b71c1c", lw=0.7),
            )

    if dynamic_result.trigger_points:
        ax.scatter(
            [cell[1] for cell in dynamic_result.trigger_points],
            [cell[0] for cell in dynamic_result.trigger_points],
            s=240, marker="*", color="#f9a825", edgecolor="#4e342e",
            linewidth=0.9, label="Replanning trigger point", zorder=7,
        )
        for index, (row, col) in enumerate(dynamic_result.trigger_points, start=1):
            # first trigger sits on the left border -> label above the star,
            # later triggers label below to avoid the detection annotation
            near_border = col <= 2
            offset = (8, 18) if near_border else (0, -22)
            ax.annotate(
                f"replan #{index}", (col, row), textcoords="offset points",
                xytext=offset, ha="left" if near_border else "center",
                fontsize=7, fontweight="bold",
                color="#4e342e",
                arrowprops=dict(arrowstyle="-", color="#4e342e", lw=0.7),
            )

    handles = _legend_handles(include_hidden=True)
    fig.legend(
        handles=handles, loc="lower center", ncol=4, fontsize=7.5,
        frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 0.0),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
              fontsize=7, framealpha=0.95)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, out_path)
