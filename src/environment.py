"""
Task 1 -- Potato Field Environment
==================================

This module models a 20 x 20 precision-agriculture field as a 2-D NumPy grid.

Cell encoding
-------------
======  =========================================================  ==========
Value   Meaning                                                    Traversable
======  =========================================================  ==========
0       Safe travelling cell (furrow / lane between crop rows)      yes (cost 1)
1       Potato crop row / strict crop-protection core                NO
2       Crop-proximity caution zone (adjacent to a crop row)        yes (cost 5)
3       Static known obstacle (stone, farm tool, fence post)         NO
4       Unknown / dynamic obstacle (hidden until detected)          NO (once known)
5       Start position S = (1, 1)                                   yes (cost 1)
6       Goal / monitoring destination G = (18, 18)                  yes (cost 1)
======  =========================================================  ==========

Two distinct crop-protection zones are modelled, as required by Task 4:

* Value ``1`` -- the potato row itself.  Driving over it would crush plants, so
  it is *strictly impassable* (an infinite-cost cell).
* Value ``2`` -- the crop-proximity caution zone, i.e. every passable cell that
  is 4-adjacent to a potato row.  It remains passable, but costs ``5`` instead of
  ``1``, because the robot's wheels / spray boom risk damaging sensitive
  seedlings.  A cost-minimising planner therefore hugs the middle of a lane
  whenever a slightly longer but safer route exists.

The field contains four vertical potato rows with 2-3 cell wide travelling lanes
between them.  Each crop row is interrupted by narrow "furrow openings" (gaps)
that act as the only legal crossings -- this forces genuine routing decisions
instead of a trivial diagonal-free straight line.

Two maps are maintained:

``robot_map``       What the robot believes: hidden obstacles (4) appear as safe
                    cells (0).
``ground_truth``    Reality: hidden obstacles are present as value 4.  The robot
                    can only learn about them through its on-board sensor
                    (Task 5).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Cell encoding constants
# --------------------------------------------------------------------------- #
SAFE: int = 0
CROP: int = 1
CAUTION: int = 2
STATIC: int = 3
DYNAMIC: int = 4
START: int = 5
GOAL: int = 6

#: Cells the robot can never enter (crushing / collision).
BLOCKED_VALUES = frozenset({CROP, STATIC, DYNAMIC})
#: Cells the robot may enter.
PASSABLE_VALUES = frozenset({SAFE, CAUTION, START, GOAL})

#: Crop-safe step-cost table (Task 2, cost function g).  Blocked cells are given
#: an infinite cost so that the table is total.
STEP_COSTS: Dict[int, float] = {
    SAFE: 1.0,
    START: 1.0,
    GOAL: 1.0,
    CAUTION: 5.0,
    CROP: float("inf"),
    STATIC: float("inf"),
    DYNAMIC: float("inf"),
}

Cell = Tuple[int, int]


class CropFieldEnvironment:
    """A 2-D potato field with crop-safe lanes, obstacles and hidden hazards.

    Parameters
    ----------
    rows, cols:
        Grid dimensions (default 20 x 20, as required by the assignment).
    """

    ROWS: int = 20
    COLS: int = 20
    START_POS: Cell = (1, 1)
    GOAL_POS: Cell = (18, 18)

    #: Column indices that carry a potato row (interrupted by furrow openings).
    CROP_COLUMNS: Sequence[int] = (3, 7, 11, 15)

    #: Furrow openings per crop column: (first_row, last_row) inclusive.
    #: Every pair is a legal crossing of that crop row.
    GAPS: Dict[int, Sequence[Tuple[int, int]]] = {
        3: ((5, 6), (14, 15)),
        7: ((2, 3), (10, 11), (17, 18)),
        11: ((5, 6), (13, 14)),
        15: ((2, 3), (9, 10), (17, 18)),
    }

    #: Static obstacles (stones, tools) placed inside travelling lanes so that
    #: BFS / DFS / Greedy / A* are forced to make different decisions.
    STATIC_OBSTACLES: Sequence[Cell] = (
        (9, 1), (10, 1),          # lane A wall (left of the first crop row)
        (7, 5), (8, 5),           # lane B pinch point
        (4, 9), (5, 9), (6, 9),   # lane C long wall -> detour required
        (11, 13), (12, 13),       # lane D pinch point
        (13, 17), (14, 17),       # lane E wall in front of the goal
    )

    def __init__(self, rows: int = ROWS, cols: int = COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.start: Cell = self.START_POS
        self.goal: Cell = self.GOAL_POS

        # 1) Static field layout, as known to the robot.
        self.robot_map: np.ndarray = self._design_field()
        # 2) Ground truth starts as an identical copy; hidden hazards are added
        #    later by ``seed_hidden_obstacles``.
        self.ground_truth: np.ndarray = self.robot_map.copy()
        self.hidden_obstacles: List[Cell] = []

    # ------------------------------------------------------------------ #
    # Field construction
    # ------------------------------------------------------------------ #
    def _design_field(self) -> np.ndarray:
        """Build the static (known) field map.

        Construction order matters:

        1. everything starts as a safe cell (0),
        2. potato rows (1) are drawn, leaving the furrow openings free,
        3. every passable cell 4-adjacent to a potato row becomes a caution
           zone (2) -- derived automatically, never hand-placed,
        4. static obstacles (3) are stamped on top,
        5. start (5) and goal (6) markers are written last.
        """
        grid = np.zeros((self.rows, self.cols), dtype=int)

        # -- 2) vertical potato rows with furrow openings --------------------
        for col in self.CROP_COLUMNS:
            if not 0 <= col < self.cols:
                continue
            for row in range(self.rows):
                grid[row, col] = CROP
            for first, last in self.GAPS.get(col, ()):
                for row in range(first, min(last, self.rows - 1) + 1):
                    grid[row, col] = SAFE

        # -- 3) crop-proximity caution zone (cost 5) ------------------------
        grid = self._mark_caution_zone(grid)

        # -- 4) static known obstacles --------------------------------------
        for row, col in self.STATIC_OBSTACLES:
            if self._in_bounds_static((row, col)):
                grid[row, col] = STATIC

        # -- 5) start and goal markers --------------------------------------
        grid[self.start] = START
        grid[self.goal] = GOAL
        return grid

    def _mark_caution_zone(self, grid: np.ndarray) -> np.ndarray:
        """Turn every passable cell adjacent to a crop row into a caution cell."""
        result = grid.copy()
        for row in range(self.rows):
            for col in range(self.cols):
                if grid[row, col] != SAFE:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if grid[nr, nc] == CROP:
                            result[row, col] = CAUTION
                            break
        return result

    def _in_bounds_static(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ------------------------------------------------------------------ #
    # Hidden / dynamic obstacles (Task 5 precondition)
    # ------------------------------------------------------------------ #
    def seed_hidden_obstacles(self, route: Sequence[Cell], count: int = 2) -> List[Cell]:
        """Hide ``count`` unknown obstacles directly on a planned route.

        The assignment requires at least two hidden hazards placed *on the
        initial optimal A\\* route* so that online detection and replanning is
        guaranteed to trigger.  Candidates are taken from the interior of the
        supplied route, must be plain safe cells of the static map (never a
        crop, caution or start/goal cell) and must stay at least four cells
        apart so that two separate detections occur.

        Parameters
        ----------
        route:
            Ordered list of cells of the route the robot is about to follow.
        count:
            Number of hidden hazards to place.

        Returns
        -------
        list of (row, col) coordinates that were made into hidden obstacles.
        """
        interior = [
            cell for cell in route[1:-1]
            if self.robot_map[cell] == SAFE
        ]
        if not interior:
            return []

        step = max(1, len(interior) // (count + 1))
        index = min(step, len(interior) - 1)

        placed: List[Cell] = []
        attempts = 0
        while len(placed) < count and attempts < len(interior):
            cell = interior[index % len(interior)]
            far_enough = all(
                abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 4
                for other in placed
            )
            if far_enough:
                placed.append(cell)
            attempts += 1
            index += step

        # Keep only placements that leave a solvable field.
        for cell in list(placed):
            self.ground_truth[cell] = DYNAMIC
            if not self.is_reachable(self.start, self.goal, self.ground_truth):
                self.ground_truth[cell] = SAFE
                placed.remove(cell)

        self.hidden_obstacles = list(placed)
        return self.hidden_obstacles

    def detect_adjacent_hazards(self, position: Cell, robot_map: np.ndarray) -> List[Cell]:
        """Return hidden hazards in the 4-neighbourhood that the robot has not
        discovered yet (sensor range = 1 cell)."""
        found: List[Cell] = []
        for nr, nc in self.neighbour_coords(position):
            if (
                self.ground_truth[nr, nc] == DYNAMIC
                and robot_map[nr, nc] != DYNAMIC
            ):
                found.append((nr, nc))
        return found

    def neighbour_coords(self, cell: Cell) -> List[Cell]:
        """In-bounds 4-connected neighbours of ``cell``."""
        row, col = cell
        out: List[Cell] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                out.append((nr, nc))
        return out

    def is_reachable(
        self,
        start: Cell,
        goal: Cell,
        grid: np.ndarray,
    ) -> bool:
        """Flood fill used to guarantee the field stays solvable."""
        seen = {start}
        queue: deque[Cell] = deque([start])
        while queue:
            row, col = queue.popleft()
            if (row, col) == goal:
                return True
            for nr, nc in self.neighbour_coords((row, col)):
                if (nr, nc) not in seen and int(grid[nr, nc]) not in BLOCKED_VALUES:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_blocked_value(value: int) -> bool:
        """``True`` for cells that are impassable (crop core, static, dynamic)."""
        return int(value) in BLOCKED_VALUES

    @staticmethod
    def step_cost(value: int) -> float:
        """Crop-safe entry cost of a cell (Task 2 cost function)."""
        return STEP_COSTS[int(value)]

    def reset_robot_map(self) -> np.ndarray:
        """Reset the robot's belief map to the static design map."""
        self.robot_map = self._design_field()
        return self.robot_map

    # ------------------------------------------------------------------ #
    # Debug / documentation helpers
    # ------------------------------------------------------------------ #
    SYMBOLS = {
        SAFE: ".",
        CROP: "#",
        CAUTION: "~",
        STATIC: "X",
        DYNAMIC: "?",
        START: "S",
        GOAL: "G",
    }

    def ascii_map(self, grid: np.ndarray | None = None) -> str:
        """Render a map as text (handy for debugging and README snippets)."""
        grid = self.robot_map if grid is None else grid
        header = "    " + "".join(str(col % 10) for col in range(self.cols))
        lines = [header]
        for row in range(self.rows):
            body = "".join(
                self.SYMBOLS[int(grid[row, col])] for col in range(self.cols)
            )
            lines.append(f"{row:3d} {body}")
        return "\n".join(lines)

    def describe(self) -> str:
        """Human-readable summary of cell statistics."""
        values, counts = np.unique(self.robot_map, return_counts=True)
        names = {
            SAFE: "safe lanes",
            CROP: "potato crop cells",
            CAUTION: "caution cells",
            STATIC: "static obstacles",
            DYNAMIC: "dynamic obstacles",
            START: "start",
            GOAL: "goal",
        }
        parts = [f"{names[int(v)]}={int(c)}" for v, c in zip(values, counts)]
        return (
            f"CropFieldEnvironment {self.rows}x{self.cols} | "
            f"start={self.start}, goal={self.goal} | " + ", ".join(parts)
        )
