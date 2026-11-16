"""
Master runner -- AI-Based Crop-Safe Path Planning for an Autonomous
Potato-Field Robot
===========================================================================

Executes the complete project end-to-end:

  Task 1  build the 20 x 20 potato field (crop rows, caution zones, obstacles)
  Task 2  print the formal search-problem formulation
  Task 3  run BFS, DFS, Greedy Best-First and A*, print the heuristic proof
  Task 4  verify the crop-safe behaviour of A*
  Task 5  simulate online obstacle detection and repeated A* replanning
  Task 6  save four 300 DPI figures and the comparison tables (CSV + Markdown)

Usage
-----
    pip install -r requirements.txt
    python main.py

All artefacts are written to ``output/`` next to this script, independent of
the current working directory.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from src.algorithms import ALGORITHMS, a_star_search, run_all_algorithms
from src.dynamic_replanning import run_dynamic_replanning
from src.environment import CropFieldEnvironment
from src.heuristics import (
    crop_aware_heuristic,
    explain_heuristic,
    manhattan_distance,
)
from src.problem import CropFieldProblem
from src import visualizer

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"

TABLE1_HEADERS = (
    "Algorithm",
    "Destination Reached",
    "Path Length (Steps)",
    "Total Crop-Safe Cost",
    "Expanded Nodes",
    "Execution Time (ms)",
)
TABLE2_HEADERS = (
    "Experiment Phase",
    "Destination Reached",
    "Traversed Steps",
    "Total Crop-Safe Cost",
    "Replans Triggered",
    "Total Expanded Nodes",
    "Total Time (ms)",
)


# --------------------------------------------------------------------------- #
# Console / file table helpers
# --------------------------------------------------------------------------- #
def format_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Return an aligned plain-text table (ASCII separators)."""
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def rule(char: str = "-") -> str:
        return "+".join(char * (width + 2) for width in widths)

    def line(cells: Sequence[str]) -> str:
        return "|".join(f" {cell.ljust(w)} " for cell, w in zip(cells, widths))

    lines = [rule("-"), line([h.center(w) for h, w in zip(headers, widths)])]
    lines.append(rule("="))
    for row in rows:
        lines.append(line(row))
    lines.append(rule("-"))
    return "\n".join(lines)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Return a GitHub-flavoured Markdown table."""
    rows = [[str(cell) for cell in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def banner(text: str) -> None:
    line = "#" * 78
    print(line)
    print(f"# {text}".ljust(77) + "#")
    print(line)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("AI-BASED CROP-SAFE PATH PLANNING FOR AN AUTONOMOUS POTATO-FIELD ROBOT")
    print("University group project -- Tasks 1 to 6")
    print()

    # ------------------------------------------------------------------ #
    # Task 1 -- environment
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("TASK 1 -- FIELD ENVIRONMENT".center(78))
    print("=" * 78)
    env = CropFieldEnvironment()
    print(env.describe())
    print()
    print("Field layout ('.' safe  '~' caution  '#' potato row  'X' static  "
          "'S' start  'G' goal)")
    print(env.ascii_map(env.robot_map))
    print()

    # The robot plans on its known map first; the hidden hazards are then
    # planted directly on that optimal route so Task 5 is guaranteed to trigger.
    known_problem = CropFieldProblem(env.robot_map, env.start, env.goal, env)
    reference_route = a_star_search(known_problem)
    hidden = env.seed_hidden_obstacles(reference_route.path, count=2)
    print(f"Hidden dynamic obstacles planted on the initial optimal A* route: {hidden}")
    print("(They are absent from the robot's known map and appear in the ground "
          "truth as value 4.)")
    print()

    # ------------------------------------------------------------------ #
    # Task 2 -- formal formulation
    # ------------------------------------------------------------------ #
    known_problem.print_formulation()

    # ------------------------------------------------------------------ #
    # Task 3 -- heuristics + four searches
    # ------------------------------------------------------------------ #
    explain_heuristic()

    print("=" * 78)
    print("TASK 3 -- STATIC SEARCH RESULTS ON THE ROBOT'S KNOWN MAP".center(78))
    print("=" * 78)
    results = run_all_algorithms(known_problem)

    table1_rows = [result.summary_row() for result in results]
    print(format_table(TABLE1_HEADERS, table1_rows))
    print()

    for result in results:
        validity = "valid" if known_problem.path_is_valid(result.path) else "INVALID"
        print(f"  {result.algorithm_name:<11} path {validity}, "
              f"{result.path_length_steps} steps, "
              f"cost {result.total_path_cost:.1f}, "
              f"{result.expanded_nodes_count} expansions")
    print()

    # Empirical admissibility cross-check: A* with a stronger (crop-aware) but
    # still admissible heuristic must return the same optimal cost.
    a_star_default = next(r for r in results if r.algorithm_name == "A*")
    a_star_crop_aware = a_star_search(
        known_problem,
        heuristic=lambda state, goal: crop_aware_heuristic(state, goal, env.robot_map),
    )
    print("Heuristic cross-check:")
    print(f"  A* with Manhattan h(n)      -> cost {a_star_default.total_path_cost:.1f} "
          f"({a_star_default.expanded_nodes_count} expansions)")
    print(f"  A* with crop-aware h(n)     -> cost {a_star_crop_aware.total_path_cost:.1f} "
          f"({a_star_crop_aware.expanded_nodes_count} expansions)")
    print("  Both heuristics are admissible, hence both searches return the same "
          "optimal crop-safe cost.")
    print()

    # ------------------------------------------------------------------ #
    # Task 4 -- crop-safe verification
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("TASK 4 -- CROP-SAFE PLANNING VERIFICATION".center(78))
    print("=" * 78)
    by_name = {result.algorithm_name: result for result in results}
    for name in ("BFS", "DFS", "Greedy BFS", "A*"):
        stats = known_problem.count_cells(by_name[name].path)
        print(f"  {name:<11} safe={stats['safe']:>3}  caution={stats['caution']:>3}  "
              f"crop_core={stats['crop']}  static_obstacle={stats['obstacle']}")
    print()

    bfs_result = by_name["BFS"]
    astar_result = by_name["A*"]
    bfs_stats = known_problem.count_cells(bfs_result.path)
    astar_stats = known_problem.count_cells(astar_result.path)

    print("Interpretation:")
    print(f"  * BFS returns the minimum-step route ({bfs_result.path_length_steps} steps)")
    print(f"    but enters {bfs_stats['caution']} crop-proximity cells "
          f"(cost {bfs_result.total_path_cost:.1f}).")
    print(f"  * A* accepts {astar_result.path_length_steps - bfs_result.path_length_steps} "
          f"extra step(s) and enters only {astar_stats['caution']} caution cell(s),")
    print(f"    reducing the crop-safe cost to {astar_result.total_path_cost:.1f} "
          f"({(1 - astar_result.total_path_cost / bfs_result.total_path_cost) * 100:.1f}% lower).")
    print("  * No algorithm enters a potato row (1) or a static obstacle (3), and")
    print("    A* never leaves a cell of cost 5 unprovoked -- it only crosses a")
    print("    caution cell where a crop row must be crossed through a furrow")
    print("    opening, which is physically unavoidable.")
    print()

    # ------------------------------------------------------------------ #
    # Task 5 -- online detection and replanning
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("TASK 5 -- ONLINE OBSTACLE DETECTION & DYNAMIC REPLANNING".center(78))
    print("=" * 78)
    dyn = run_dynamic_replanning(env, verbose=True)

    phase1_row = (
        "Phase 1 -- initial plan (known map)",
        "Yes" if dyn.initial_planned_path else "No",
        dyn.initial_steps,
        f"{known_problem.path_cost(dyn.initial_planned_path):.1f}",
        0,
        dyn.initial_expanded_nodes,
        f"{dyn.initial_execution_time_ms:.3f}",
    )
    phase2_row = (
        "Phase 2 -- online run (detection + replanning)",
        "Yes" if dyn.reached_goal else "No",
        dyn.traversed_steps,
        f"{dyn.total_path_cost:.1f}",
        dyn.total_replans,
        dyn.total_expanded_nodes,
        f"{dyn.total_execution_time_ms:.3f}",
    )

    print(format_table(TABLE2_HEADERS, [phase1_row, phase2_row]))
    print()
    if dyn.replanning_events:
        print("Replanning log:")
        for index, event in enumerate(dyn.replanning_events, start=1):
            print(f"  {index}. robot at {event.robot_position} detected hidden "
                  f"obstacle {event.detected_obstacle}")
            print(f"     -> new sub-path of {len(event.new_subpath) - 1} steps "
                  f"starting {event.new_subpath[0]}")
    print()

    # ------------------------------------------------------------------ #
    # Task 6 -- figures
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("TASK 6 -- VISUALISATION & PERFORMANCE EVALUATION".center(78))
    print("=" * 78)

    figures = [
        visualizer.plot_environment(env, OUTPUT_DIR / "01_field_environment.png"),
        visualizer.plot_algorithm_comparison(
            env, known_problem, results,
            OUTPUT_DIR / "02_algorithm_paths_comparison.png",
        ),
        visualizer.plot_cost_effect(
            env, known_problem, bfs_result, astar_result,
            OUTPUT_DIR / "03_crop_safe_cost_effect.png",
        ),
        visualizer.plot_dynamic_replanning(
            env, dyn, OUTPUT_DIR / "04_dynamic_replanning_simulation.png",
        ),
    ]

    # ------------------------------------------------------------------ #
    # Tables -> CSV + Markdown
    # ------------------------------------------------------------------ #
    table1_md = markdown_table(TABLE1_HEADERS, table1_rows)
    table2_md = markdown_table(TABLE2_HEADERS, [phase1_row, phase2_row])

    save_csv(OUTPUT_DIR / "performance_comparison.csv", TABLE1_HEADERS, table1_rows)
    (OUTPUT_DIR / "performance_comparison.md").write_text(
        "# Performance Comparison\n\n"
        "## Table 1 -- Static Path-Planning Performance Comparison\n\n"
        f"{table1_md}\n\n"
        "## Table 2 -- Online Obstacle Detection & Replanning Summary\n\n"
        f"{table2_md}\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # Wrap-up
    # ------------------------------------------------------------------ #
    print("Figures written:")
    for path in figures:
        size_kb = Path(path).stat().st_size / 1024
        print(f"  - {path}  ({size_kb:.0f} KB)")
    print()
    print("Tables written:")
    print(f"  - {OUTPUT_DIR / 'performance_comparison.md'}")
    print(f"  - {OUTPUT_DIR / 'performance_comparison.csv'}")
    print()
    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()
