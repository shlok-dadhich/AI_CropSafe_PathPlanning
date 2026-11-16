# Performance Comparison

## Table 1 -- Static Path-Planning Performance Comparison

| Algorithm | Destination Reached | Path Length (Steps) | Total Crop-Safe Cost | Expanded Nodes | Execution Time (ms) |
| --- | --- | --- | --- | --- | --- |
| BFS | Yes | 34 | 98.0 | 310 | 0.598 |
| DFS | Yes | 174 | 550.0 | 203 | 0.44 |
| Greedy BFS | Yes | 42 | 126.0 | 57 | 0.147 |
| A* | Yes | 42 | 58.0 | 213 | 0.586 |

## Table 2 -- Online Obstacle Detection & Replanning Summary

| Experiment Phase | Destination Reached | Traversed Steps | Total Crop-Safe Cost | Replans Triggered | Total Expanded Nodes | Total Time (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 1 -- initial plan (known map) | Yes | 42 | 58.0 | 0 | 213 | 0.562 |
| Phase 2 -- online run (detection + replanning) | Yes | 44 | 68.0 | 2 | 453 | 1.186 |
