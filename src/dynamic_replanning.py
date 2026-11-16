"""
Task 5 -- Online Obstacle Detection & Dynamic Replanning
=======================================================

The robot never sees the whole field: it carries a *belief map* (``robot_map``)
in which the hidden hazards of the ground truth simply do not exist yet.  While
driving, an on-board sensor with a range of **1 cell** (4-neighbourhood) scans
the surrounding cells and, whenever a hidden obstacle is discovered, the belief
map is corrected.

If (and only if) the newly discovered obstacle blocks the *remaining* planned
route, the robot stops following the stale plan and re-runs A* from its current
position to the goal on the updated map.  This is the classical
"plan -> execute -> detect -> replan" loop of online search.

The simulation records everything that happened so that Task 6 can draw the
initial plan, the detection points and the actually executed trajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .algorithms import SearchResult, a_star_search
from .environment import DYNAMIC, CropFieldEnvironment
from .problem import Cell, CropFieldProblem

Path = List[Cell]


@dataclass
class ReplanningEvent:
    """One "obstacle found on the route -> new sub-path" event."""

    robot_position: Cell
    detected_obstacle: Cell
    new_subpath: Path


@dataclass
class DynamicReplanningResult:
    """Everything Task 5 / Task 6 needs to know about the online run."""

    initial_planned_path: Path = field(default_factory=list)
    actual_traversed_path: Path = field(default_factory=list)
    replanning_events: List[ReplanningEvent] = field(default_factory=list)
    total_replans: int = 0
    total_expanded_nodes: int = 0
    total_execution_time_ms: float = 0.0
    reached_goal: bool = False
    traversed_steps: int = 0
    total_path_cost: float = 0.0
    detected_obstacles: List[Cell] = field(default_factory=list)
    trigger_points: List[Cell] = field(default_factory=list)
    final_robot_map: np.ndarray | None = None
    #: Statistics of the very first (offline) A* plan, used by Table 2.
    initial_expanded_nodes: int = 0
    initial_execution_time_ms: float = 0.0
    initial_steps: int = 0

    # Alias kept for readability in the report.
    @property
    def replanning_count(self) -> int:
        return self.total_replans


def run_dynamic_replanning(
    env: CropFieldEnvironment,
    max_steps: int = 1000,
    verbose: bool = True,
) -> DynamicReplanningResult:
    """Simulate the robot driving from ``env.start`` to ``env.goal``.

    Parameters
    ----------
    env:
        Environment whose ``ground_truth`` contains hidden obstacles (value 4)
        and whose ``robot_map`` is the initial belief map.
    max_steps:
        Safety guard against an endless loop in a pathological field.
    verbose:
        Print detection / replanning log lines to the console.

    Returns
    -------
    DynamicReplanningResult
        Initial plan, executed trajectory, detection log, replan count and the
        aggregated search effort (expanded nodes and wall-clock time).
    """
    result = DynamicReplanningResult()
    robot_map = env.robot_map.copy()          # belief map: hidden cells look safe
    ground_truth = env.ground_truth           # reality

    agent_problem = CropFieldProblem(robot_map, env.start, env.goal, env)

    # ---------------------------------------------------------------- #
    # 1) Initial offline plan on the (incomplete) robot map
    # ---------------------------------------------------------------- #
    initial_search: SearchResult = a_star_search(agent_problem)
    result.initial_planned_path = list(initial_search.path)
    result.total_expanded_nodes += initial_search.expanded_nodes_count
    result.total_execution_time_ms += initial_search.execution_time_ms
    result.initial_expanded_nodes = initial_search.expanded_nodes_count
    result.initial_execution_time_ms = initial_search.execution_time_ms
    result.initial_steps = initial_search.path_length_steps

    if not initial_search.reached_goal:
        result.final_robot_map = robot_map
        if verbose:
            print("[Task 5] No initial route exists -- aborting simulation.")
        return result

    #: Invariant: ``product`` always starts at the robot's current position.
    product: Path = list(result.initial_planned_path)
    position: Cell = env.start
    result.actual_traversed_path = [position]

    if verbose:
        print("[Task 5] Initial A* plan computed on the robot's known map:")
        print(f"         steps={initial_search.path_length_steps}, "
              f"cost={initial_search.total_path_cost:.1f}, "
              f"expanded={initial_search.expanded_nodes_count}")

    steps = 0
    while position != env.goal and steps < max_steps:
        steps += 1

        # ------------------------------------------------------------ #
        # 2) Local sensor scan (range = 1 cell, 4-neighbourhood)
        # ------------------------------------------------------------ #
        newly_detected = env.detect_adjacent_hazards(position, robot_map)
        for hazard in newly_detected:
            robot_map[hazard] = DYNAMIC              # update belief map
            result.detected_obstacles.append(hazard)
            if verbose:
                print(f"         [SENSOR] unknown obstacle detected at {hazard} "
                      f"while robot is at {position}")

        # ------------------------------------------------------------ #
        # 3) Replan only if the discovery invalidates the current plan
        # ------------------------------------------------------------ #
        if newly_detected:
            remaining = product[1:]                  # cells still ahead
            blocking = [hazard for hazard in newly_detected if hazard in remaining]
            if blocking:
                agent_problem = CropFieldProblem(robot_map, position, env.goal, env)
                replanned = a_star_search(agent_problem)
                result.total_expanded_nodes += replanned.expanded_nodes_count
                result.total_execution_time_ms += replanned.execution_time_ms
                if not replanned.reached_goal:
                    if verbose:
                        print("         [REPLAN] no route left -- robot stops.")
                    break
                product = list(replanned.path)
                result.total_replans += 1
                result.trigger_points.append(position)
                result.replanning_events.append(
                    ReplanningEvent(
                        robot_position=position,
                        detected_obstacle=blocking[0],
                        new_subpath=list(replanned.path),
                    )
                )
                if verbose:
                    print(f"         [REPLAN #{result.total_replans}] obstacle "
                          f"{blocking[0]} blocks the plan -> new A* route from "
                          f"{position} (steps={replanned.path_length_steps}, "
                          f"cost={replanned.total_path_cost:.1f})")

        # ------------------------------------------------------------ #
        # 4) Execute one step of the (possibly fresh) plan
        # ------------------------------------------------------------ #
        if len(product) < 2:
            break
        next_cell = product[1]
        if int(robot_map[next_cell]) in (1, 3, 4):    # safety net, should not fire
            if verbose:
                print(f"         [WARN] next cell {next_cell} is blocked; stopping.")
            break
        product = product[1:]
        position = next_cell
        result.actual_traversed_path.append(position)

    # ---------------------------------------------------------------- #
    # 5) Aggregate statistics
    # ---------------------------------------------------------------- #
    result.reached_goal = position == env.goal
    result.traversed_steps = max(0, len(result.actual_traversed_path) - 1)
    # Costs are evaluated on the ground truth so the penalty of caution cells is
    # counted with the same rule as in Task 3.
    cost_problem = CropFieldProblem(ground_truth, env.start, env.goal, env)
    result.total_path_cost = cost_problem.path_cost(result.actual_traversed_path)
    result.final_robot_map = robot_map

    if verbose:
        print()
        print(f"[Task 5] Finished: reached_goal={result.reached_goal}, "
              f"steps={result.traversed_steps}, "
              f"cost={result.total_path_cost:.1f}, "
              f"replans={result.total_replans}, "
              f"expanded={result.total_expanded_nodes}, "
              f"time={result.total_execution_time_ms:.3f} ms")
        print()
    return result
