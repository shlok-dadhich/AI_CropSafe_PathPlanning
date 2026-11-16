"""
Tasks 2 & 4 -- Formal Problem Formulation
=========================================

Defines the classical search problem for the crop-safe robot as a 5-tuple
``P = (S, s0, A, Result, c, IsGoal)`` and exposes the crop-safe path cost that
A* minimises.

Problem elements
----------------
* **State space S**: every cell ``(r, c)`` of the 20 x 20 field.
* **Initial state s0**: ``(1, 1)``.
* **Action space A(s)**: four cardinal moves UP / DOWN / LEFT / RIGHT.
  Diagonal moves are deliberately forbidden -- a diagonal step between two
  potato plants would clip the plants' canopy and is physically unsafe.
* **Transition model Result(s, a)**: ``s' = (r + dr, c + dc)`` if ``s'`` is
  inside the field and its cell value is not in {1, 3, 4}.
* **Goal test IsGoal(s)**: ``s == (18, 18)``.
* **Path cost g(n)**: sum of cell-entry costs, 1 for safe furrow cells, 5 for
  crop-proximity caution cells and infinity for blocked cells.

``print_formulation`` renders all of this (including the piecewise cost
function) to the console for the project report.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .environment import (
    BLOCKED_VALUES,
    CAUTION,
    CROP,
    GOAL,
    SAFE,
    START,
    STATIC,
    CropFieldEnvironment,
)

Cell = Tuple[int, int]


class CropFieldProblem:
    """A single-agent path-planning problem on a crop field grid.

    Parameters
    ----------
    grid:
        Map used for planning (usually the robot's known map; for Task 5 a
        freshly updated copy with detected hazards stamped in as value 4).
    start, goal:
        Initial state and goal state.
    environment:
        Optional reference to the owning environment (informational only).
    """

    ACTIONS: Sequence[Tuple[str, int, int]] = (
        ("UP", -1, 0),
        ("DOWN", +1, 0),
        ("LEFT", 0, -1),
        ("RIGHT", 0, +1),
    )

    def __init__(
        self,
        grid: np.ndarray,
        start: Cell,
        goal: Cell,
        environment: CropFieldEnvironment | None = None,
    ) -> None:
        self.grid = np.asarray(grid, dtype=int)
        self.start: Cell = start
        self.goal: Cell = goal
        self.environment = environment
        self.rows, self.cols = self.grid.shape

    # ------------------------------------------------------------------ #
    # Basic problem interface
    # ------------------------------------------------------------------ #
    def in_bounds(self, state: Cell) -> bool:
        row, col = state
        return 0 <= row < self.rows and 0 <= col < self.cols

    def passable(self, state: Cell) -> bool:
        """A cell is passable iff its value is not a crop core / obstacle."""
        return self.in_bounds(state) and int(self.grid[state]) not in BLOCKED_VALUES

    def transition(self, state: Cell, action_index: int) -> Cell | None:
        """Deterministic transition model ``Result(s, a)``.

        Returns ``None`` (the move is illegal) when the target cell leaves the
        field or is impassable.
        """
        _, dr, dc = self.ACTIONS[action_index]
        successor = (state[0] + dr, state[1] + dc)
        return successor if self.passable(successor) else None

    def successors(self, state: Cell) -> List[Tuple[str, Cell, float]]:
        """All legal ``(action_name, next_state, step_cost)`` triples."""
        result: List[Tuple[str, Cell, float]] = []
        for name, dr, dc in self.ACTIONS:
            successor = (state[0] + dr, state[1] + dc)
            if self.passable(successor):
                result.append((name, successor, self.step_cost(successor)))
        return result

    def neighbours(self, state: Cell) -> List[Cell]:
        """Legal successor states only (convenience for the search code)."""
        return [nxt for _, nxt, _ in self.successors(state)]

    def step_cost(self, state: Cell) -> float:
        """Cost of *entering* ``state`` (the crop-safe cost function)."""
        value = int(self.grid[state])
        if value in BLOCKED_VALUES:
            return float("inf")
        return 5.0 if value == CAUTION else 1.0

    def is_goal(self, state: Cell) -> bool:
        return state == self.goal

    # ------------------------------------------------------------------ #
    # Path utilities
    # ------------------------------------------------------------------ #
    def path_cost(self, path: Sequence[Cell]) -> float:
        """Total crop-safe cost of a path (edges = cell entries)."""
        if len(path) < 2:
            return 0.0
        return float(sum(self.step_cost(state) for state in path[1:]))

    def path_is_valid(self, path: Sequence[Cell]) -> bool:
        """Check that a path is contiguous and never enters a blocked cell."""
        if not path:
            return False
        if path[0] != self.start or path[-1] != self.goal:
            return False
        for previous, current in zip(path, path[1:]):
            if abs(previous[0] - current[0]) + abs(previous[1] - current[1]) != 1:
                return False
            if not self.passable(current):
                return False
        return True

    def count_cells(self, path: Sequence[Cell]) -> dict:
        """Count crop-proximity (2), crop core (1) and obstacle (3/4) entries."""
        stats = {"safe": 0, "caution": 0, "crop": 0, "obstacle": 0}
        for state in path[1:]:
            value = int(self.grid[state])
            if value == CAUTION:
                stats["caution"] += 1
            elif value == CROP:
                stats["crop"] += 1
            elif value in BLOCKED_VALUES:
                stats["obstacle"] += 1
            else:
                stats["safe"] += 1
        return stats

    # ------------------------------------------------------------------ #
    # Task 2 -- console formulation
    # ------------------------------------------------------------------ #
    def print_formulation(self) -> None:
        """Print the formal problem formulation used in the report."""
        line = "=" * 78
        print(line)
        print("TASK 2 -- FORMAL PROBLEM FORMULATION".center(78))
        print("AI-Based Crop-Safe Path Planning for an Autonomous Potato-Field Robot")
        print(line)
        print(f"Field: {self.rows} x {self.cols} grid of integer cell encodings")
        print("  0 = safe furrow/lane (cost 1)   1 = potato row / protected core (BLOCKED)")
        print("  2 = crop-proximity caution (cost 5)   3 = static obstacle (BLOCKED)")
        print("  4 = hidden dynamic obstacle (BLOCKED once detected)")
        print("  5 = start S   6 = goal G")
        print()

        print("1. INITIAL STATE (s0)")
        print(f"     s0 = {self.start}   (top-left corner of the field)")
        print()

        print("2. ACTION SPACE A(s)")
        print("     A(s) = { UP (-1, 0), DOWN (+1, 0), LEFT (0, -1), RIGHT (0, +1) }")
        print("     4-connected only: diagonal motion is forbidden so the robot never")
        print("     clips the canopy / roots of a potato plant.")
        print()

        print("3. TRANSITION MODEL  Result(s, a) = s'")
        print("     s' = (r + dr, c + dc)")
        print("     legal iff  0 <= r' < R  and  0 <= c' < C  and  grid[r'][c'] not in {1, 3, 4}")
        print()

        print("4. GOAL TEST  IsGoal(s)")
        print(f"     IsGoal(s) = True  <=>  s == {self.goal}")
        print()

        print("5. PATH-COST FUNCTION  g(n) = sum of c(s, a, s') along the path")
        print("            /  1   if s' is a normal safe cell        (values 0, 5, 6)")
        print("     c(s,a,s') =  5   if s' is a crop-proximity cell  (value 2)")
        print("            \\  inf  if s' is crop core / obstacle     (values 1, 3, 4)")
        print("     The penalty of 5 > any 4-step detour makes the planner prefer a")
        print("     slightly longer, safer centre lane over hugging the crop wall.")
        print()

        print("6. SOLUTION")
        print(f"     A path pi = <s0, s1, ..., sk> with sk = {self.goal}, every transition")
        print("     legal, minimising g(pi).  A* with an admissible Manhattan heuristic")
        print("     returns such an optimal crop-safe path.")
        print(line)
        print()
