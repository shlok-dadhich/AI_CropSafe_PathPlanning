"""AI-Based Crop-Safe Path Planning for an Autonomous Potato-Field Robot.

Package layout
--------------
environment          Task 1 -- 2-D potato-field grid and crop-safe buffers
problem              Tasks 2 & 4 -- state space, transitions, crop-safe cost
heuristics           Task 3 -- admissible Manhattan / crop-aware heuristics
algorithms           Task 3 -- BFS, DFS, Greedy Best-First, A*
dynamic_replanning   Task 5 -- online detection and repeated A* replanning
visualizer           Tasks 1, 3, 5, 6 -- publication-quality figures
"""

from .environment import CropFieldEnvironment
from .problem import CropFieldProblem

__all__ = ["CropFieldEnvironment", "CropFieldProblem"]
