"""
Task 3 -- Uninformed and Informed Search Algorithms
===================================================

Four graph-search algorithms are implemented for the crop-field problem:

===========  =========================  =====================================
Algorithm    Frontier                   Objective
===========  =========================  =====================================
BFS          FIFO ``collections.deque`` shallowest node first -> fewest steps
DFS          LIFO ``list``              deepest node first -> any path
Greedy BFS   ``heapq`` on ``h(n)``      pure goal-directed, ignores cost
A*           ``heapq`` on ``g(n)+h(n)``optimal crop-safe path
===========  =========================  =====================================

Every algorithm returns the same :class:`SearchResult` object so that the
comparison tables and the visualisations can treat the algorithms uniformly.

Implementation details
----------------------
* Nodes are expanded at most once (``closed`` set / ``visited`` order), which is
  safe for BFS, DFS, Greedy and for A* with a *consistent* heuristic.
* The insertion ``counter`` in the priority queues makes the ordering stable
  when two nodes share the same priority key (tuples are never compared).
* Execution time is measured with ``time.perf_counter()`` around the whole
  search (excluding path reconstruction bookkeeping noise is negligible).
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from .heuristics import manhattan_distance
from .problem import Cell, CropFieldProblem

Path = List[Cell]


@dataclass
class SearchResult:
    """Standardised result object returned by every algorithm."""

    algorithm_name: str
    path: Path = field(default_factory=list)
    reached_goal: bool = False
    path_length_steps: int = 0
    total_path_cost: float = 0.0
    expanded_nodes_count: int = 0
    expanded_order: List[Cell] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def summary_row(self) -> Tuple[str, str, int, float, int, float]:
        """Row used by the performance comparison table."""
        return (
            self.algorithm_name,
            "Yes" if self.reached_goal else "No",
            self.path_length_steps,
            round(self.total_path_cost, 2),
            self.expanded_nodes_count,
            round(self.execution_time_ms, 3),
        )


def _reconstruct_path(parent: Dict[Cell, Cell], node: Cell) -> Path:
    """Walk the parent pointers back from ``node`` to the start node."""
    path: Path = [node]
    while node in parent:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path


def _make_result(
    *,
    name: str,
    problem: CropFieldProblem,
    parent: Dict[Cell, Cell],
    goal_node: Cell | None,
    expanded: List[Cell],
    elapsed_ms: float,
) -> SearchResult:
    """Assemble a :class:`SearchResult` from the raw search bookkeeping."""
    if goal_node is None:
        return SearchResult(
            algorithm_name=name,
            path=[],
            reached_goal=False,
            path_length_steps=0,
            total_path_cost=float("inf"),
            expanded_nodes_count=len(expanded),
            expanded_order=expanded,
            execution_time_ms=elapsed_ms,
        )
    path = _reconstruct_path(parent, goal_node)
    return SearchResult(
        algorithm_name=name,
        path=path,
        reached_goal=True,
        path_length_steps=max(0, len(path) - 1),
        total_path_cost=problem.path_cost(path),
        expanded_nodes_count=len(expanded),
        expanded_order=expanded,
        execution_time_ms=elapsed_ms,
    )


# --------------------------------------------------------------------------- #
# 1) Breadth-First Search
# --------------------------------------------------------------------------- #
def breadth_first_search(problem: CropFieldProblem) -> SearchResult:
    """FIFO exploration -- guarantees the minimum number of *steps*.

    BFS is blind to the crop-proximity penalty: it returns a shortest-step path
    even if that path hugs the potato rows and accumulates a high crop-safe cost.
    """
    start_time = time.perf_counter()
    name = "BFS"

    if problem.is_goal(problem.start):
        return SearchResult(name, [problem.start], True, 0, 0.0, 0, [], 0.0)

    frontier: deque[Cell] = deque([problem.start])
    reached = {problem.start}
    parent: Dict[Cell, Cell] = {}
    expanded: List[Cell] = []
    goal_node: Cell | None = None

    while frontier:
        node = frontier.popleft()
        expanded.append(node)
        for successor in problem.neighbours(node):
            if successor in reached:
                continue
            reached.add(successor)
            parent[successor] = node
            if problem.is_goal(successor):
                goal_node = successor
                frontier.clear()
                break
            frontier.append(successor)

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return _make_result(
        name=name,
        problem=problem,
        parent=parent,
        goal_node=goal_node,
        expanded=expanded,
        elapsed_ms=elapsed,
    )


# --------------------------------------------------------------------------- #
# 2) Depth-First Search
# --------------------------------------------------------------------------- #
def depth_first_search(problem: CropFieldProblem) -> SearchResult:
    """LIFO exploration with cycle checking.

    DFS finds *a* path, not a good one: it plunges into the first direction it
    can and only backtracks on dead ends, so its path is typically long and
    mechanically wasteful for a field robot.
    """
    start_time = time.perf_counter()
    name = "DFS"

    stack: List[Cell] = [problem.start]
    parent: Dict[Cell, Cell] = {}
    visited = {problem.start}
    expanded: List[Cell] = []
    goal_node: Cell | None = None

    while stack:
        node = stack.pop()
        expanded.append(node)
        if problem.is_goal(node):
            goal_node = node
            break
        # Push neighbours in reverse so that the natural order UP, DOWN, LEFT,
        # RIGHT is explored first (purely cosmetic, keeps runs reproducible).
        for successor in reversed(problem.neighbours(node)):
            if successor in visited:
                continue
            visited.add(successor)
            parent[successor] = node
            stack.append(successor)

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return _make_result(
        name=name,
        problem=problem,
        parent=parent,
        goal_node=goal_node,
        expanded=expanded,
        elapsed_ms=elapsed,
    )


# --------------------------------------------------------------------------- #
# 3) Greedy Best-First Search
# --------------------------------------------------------------------------- #
def greedy_best_first_search(
    problem: CropFieldProblem,
    heuristic: Callable[[Cell, Cell], float] = manhattan_distance,
) -> SearchResult:
    """Priority queue ordered purely by ``f(n) = h(n)``.

    Greedy search sprints towards the goal and therefore happily cuts through
    high-penalty crop-proximity cells whenever that looks shorter.
    """
    start_time = time.perf_counter()
    name = "Greedy BFS"

    counter = itertools.count()
    frontier: List[Tuple[float, int, Cell]] = [
        (heuristic(problem.start, problem.goal), next(counter), problem.start)
    ]
    parent: Dict[Cell, Cell] = {}
    visited = {problem.start}
    expanded: List[Cell] = []
    goal_node: Cell | None = None

    while frontier:
        _, _, node = heapq.heappop(frontier)
        expanded.append(node)
        if problem.is_goal(node):
            goal_node = node
            break
        for successor in problem.neighbours(node):
            if successor in visited:
                continue
            visited.add(successor)
            parent[successor] = node
            priority = heuristic(successor, problem.goal)
            heapq.heappush(frontier, (priority, next(counter), successor))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return _make_result(
        name=name,
        problem=problem,
        parent=parent,
        goal_node=goal_node,
        expanded=expanded,
        elapsed_ms=elapsed,
    )


# --------------------------------------------------------------------------- #
# 4) A* Search
# --------------------------------------------------------------------------- #
def a_star_search(
    problem: CropFieldProblem,
    heuristic: Callable[[Cell, Cell], float] = manhattan_distance,
) -> SearchResult:
    """``f(n) = g(n) + h(n)`` -- optimal for an admissible heuristic.

    ``g`` stores the cheapest known crop-safe cost to reach a node.  With the
    consistent Manhattan heuristic each node is expanded once, so A* returns the
    minimum-cost path, i.e. the path that best balances length and crop safety.
    """
    start_time = time.perf_counter()
    name = "A*"

    counter = itertools.count()
    frontier: List[Tuple[float, int, Cell]] = [
        (heuristic(problem.start, problem.goal), next(counter), problem.start)
    ]
    g_score: Dict[Cell, float] = {problem.start: 0.0}
    parent: Dict[Cell, Cell] = {}
    closed = set()
    expanded: List[Cell] = []
    goal_node: Cell | None = None

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expanded.append(node)
        if problem.is_goal(node):
            goal_node = node
            break
        for _, successor, cost in problem.successors(node):
            if successor in closed:
                continue
            tentative_g = g_score[node] + cost
            if tentative_g < g_score.get(successor, float("inf")):
                g_score[successor] = tentative_g
                parent[successor] = node
                priority = tentative_g + heuristic(successor, problem.goal)
                heapq.heappush(frontier, (priority, next(counter), successor))

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return _make_result(
        name=name,
        problem=problem,
        parent=parent,
        goal_node=goal_node,
        expanded=expanded,
        elapsed_ms=elapsed,
    )


#: Convenience registry used by ``main.py`` and the tests.
ALGORITHMS: Dict[str, Callable[[CropFieldProblem], SearchResult]] = {
    "BFS": breadth_first_search,
    "DFS": depth_first_search,
    "Greedy BFS": greedy_best_first_search,
    "A*": a_star_search,
}


def run_all_algorithms(problem: CropFieldProblem) -> List[SearchResult]:
    """Run all four algorithms on the same problem, in report order."""
    return [search(problem) for search in ALGORITHMS.values()]
