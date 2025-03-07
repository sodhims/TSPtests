# TSP Genetic Algorithm Implementation

This repository contains an implementation of the Traveling Salesman Problem (TSP) using genetic algorithms, including special crossover operators designed for TSP.

## Problem Description
The Traveling Salesman Problem involves finding the shortest possible route that visits each city exactly once and returns to the starting city. This implementation uses genetic algorithms with specialized crossover operators to find near-optimal solutions.

## Features
- Multiple crossover operators including:
  - Matrix crossover
  - Edge crossover
- Customizable GA parameters
- Comparison with exact solution using Gurobi
- Visualization of convergence
- Performance analysis tools

## Dependencies
```python
gurobipy    # For exact solution comparison
numpy       # For numerical operations
matplotlib  # For plotting results
