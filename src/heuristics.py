"""
Task 3 -- Heuristics
====================

Two heuristics are provided for the 4-connected crop-field grid:

1. ``manhattan_distance`` -- the standard ``L1`` distance to the goal.  It is
   **admissible** and **consistent** for this problem (proof printed by
   ``explain_heuristic``).
2. ``crop_aware_heuristic`` -- Manhattan plus a grid-derived lower bound on the
   extra cost forced by the crop-proximity caution zones.  It is admissible but
   not necessarily consistent; it is used only for an empirical comparison, the
   four search algorithms use the Manhattan heuristic.

Why Manhattan is admissible (never over-estimates)
--------------------------------------------------
Minimum step cost is ``c_min = 1`` (a safe furrow cell).  With only cardinal
moves every step changes exactly one coordinate by one, hence at least
``h(n) = |r_n - r_g| + |c_n - c_g|`` steps are required to reach the goal.  The
cheapest possible completion therefore costs at least ``h(n) * 1 = h(n)``:
``h(n) <= h*(n)``.  It is also *consistent*:
``h(n) <= c(n, n') + h(n')`` because moving one cell changes ``h`` by at most 1
and every edge costs at least 1.  Consistency implies admissibility and means
A* never needs to re-open a closed node.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .environment import CAUTION, PASSABLE_VALUES

Cell = Tuple[int, int]


def manhattan_distance(state: Cell, goal: Cell) -> float:
    """``h(n) = |r_n - r_goal| + |c_n - c_goal|`` -- admissible and consistent."""
    return float(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))


def crop_aware_heuristic(state: Cell, goal: Cell, grid: np.ndarray) -> float:
    """Manhattan distance plus unavoidable crop-proximity penalties.

    Every crop column strictly between the current column and the goal column
    must be crossed, i.e. the robot must enter at least one cell of that column.
    If the cheapest passable cell of such a column is a caution cell (cost 5),
    the crossing costs at least 4 more than the "1 per step" already counted by
    the Manhattan term.  Summing those unavoidable surpluses keeps the estimate
    below the true optimal cost (admissible), because it only counts penalties
    that no route can avoid.
    """
    base = manhattan_distance(state, goal)
    if grid is None or grid.size == 0:
        return base

    rows, cols = grid.shape
    col_from, col_to = sorted((state[1], goal[1]))
    extra = 0.0
    for col in range(col_from + 1, col_to):
        column = grid[:, col]
        costs = [
            (5.0 if int(value) == CAUTION else 1.0)
            for value in column
            if int(value) in PASSABLE_VALUES
        ]
        if costs and min(costs) > 1.0:
            extra += min(costs) - 1.0
    return base + extra


def explain_heuristic() -> None:
    """Print the admissibility / consistency argument for the report."""
    line = "=" * 78
    print(line)
    print("TASK 3 -- HEURISTIC: MANHATTAN DISTANCE".center(78))
    print(line)
    print("h(n) = |r_n - r_goal| + |c_n - c_goal|")
    print()
    print("ADMISSIBLE  (h(n) <= h*(n) for every node n):")
    print("  * The action set is 4-connected, so one step changes exactly one")
    print("    coordinate by 1 -> at least h(n) steps are needed.")
    print("  * The minimum possible step cost is 1 (safe furrow cell), so the")
    print("    cheapest completion of any node costs at least h(n) * 1 = h(n).")
    print("  * Therefore the true remaining cost h*(n) >= h(n): the heuristic")
    print("    never over-estimates and A* is guaranteed to be optimal.")
    print()
    print("CONSISTENT  (h(n) <= c(n, n') + h(n') for every edge n -> n'):")
    print("  * Moving one cell changes h by at most 1 (triangle inequality).")
    print("  * Every legal edge costs at least 1, so h(n) - h(n') <= 1 <= c(n, n').")
    print("  * Consistency implies admissibility and allows A* to expand each")
    print("    node at most once (closed set), which is what our implementation")
    print("    does.")
    print()
    print("The crop-aware heuristic adds only penalties that are physically")
    print("unavoidable (caution cells that must be crossed in intermediate crop")
    print("columns), so it stays admissible as well -- it is used as a cross-check")
    print("and produces the same optimal cost as plain Manhattan.")
    print(line)
    print()
