import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple

def solve_tsp_gurobi(distance_matrix: np.ndarray, time_limit: int = None) -> Tuple[List[int], float]:
    """
    Solve TSP using Gurobi.
    
    Args:
        distance_matrix: Square matrix of distances between cities
        time_limit: Maximum solution time in seconds (None for no limit)
    
    Returns:
        tour: List of cities in optimal tour order
        optimal_distance: Length of optimal tour
    """
    n = len(distance_matrix)
    
    # Create a new model
    m = gp.Model("tsp")
    
    # Set time limit if specified
    if time_limit is not None:
        m.setParam('TimeLimit', time_limit)
    
    # Create variables
    vars = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    
    # Set objective
    m.setObjective(
        gp.quicksum(distance_matrix[i,j] * vars[i,j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )
    
    # Add degree-2 constraints
    for i in range(n):
        # Each city must be visited exactly once
        m.addConstr(gp.quicksum(vars[i,j] for j in range(n)) == 1)
        # Each city must be departed from exactly once
        m.addConstr(gp.quicksum(vars[j,i] for j in range(n)) == 1)
    
    # Prevent self-loops
    for i in range(n):
        vars[i,i].UB = 0
    
    # Subtour elimination constraints using lazy constraints
    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            # Create adjacency matrix from solution
            selected = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if vals[i, j] > 0.5:
                        selected[i,j] = 1
            # Find subtours
            tour = find_subtour(selected)
            if len(tour) < n:
                # Add subtour elimination constraint
                expr = 0
                for i, j in tour:
                    expr += model._vars[i,j]
                model.cbLazy(expr <= len(tour)-1)
    
    # Set callback
    m._vars = vars
    m.Params.lazyConstraints = 1
    
    # Optimize
    m.optimize(subtour_callback)
    
    # Get solution
    if m.status == GRB.OPTIMAL:
        solution = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if vars[i,j].X > 0.5:
                    solution[i,j] = 1
                    
        tour = extract_tour(solution)
        tour_distance = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
        
        # Print the optimal tour
        print("\nOptimal Tour Found:")
        print("Tour order:", ' -> '.join(str(city) for city in tour + [tour[0]]))
        print(f"Total distance: {tour_distance:.2f}")
        print(f"Gurobi optimal distance: {m.objVal:.2f}")
        
        return tour, tour_distance
    else:
        print("\nNo optimal solution found")
        return None, None

def find_subtour(solution: np.ndarray) -> List[Tuple[int, int]]:
    """Find the smallest subtour in the current solution."""
    n = solution.shape[0]
    visited = [False] * n
    shortest_tour = None
    
    for start in range(n):
        if visited[start]:
            continue
            
        # Find tour starting at the current node
        current_tour = []
        current = start
        while True:
            visited[current] = True
            # Find the next city in the tour
            next_cities = [j for j in range(n) if solution[current,j] > 0.5]
            if not next_cities:
                break
            next_city = next_cities[0]
            current_tour.append((current, next_city))
            current = next_city
            if current == start:  # Complete tour
                break
            
        if current_tour:  # If we found a tour
            if shortest_tour is None or len(current_tour) < len(shortest_tour):
                shortest_tour = current_tour
                
    return shortest_tour if shortest_tour is not None else []

def extract_tour(solution: np.ndarray) -> List[int]:
    """Extract the complete tour from the solution matrix."""
    n = solution.shape[0]
    visited = set()
    tour = [0]  # Start from city 0
    visited.add(0)
    
    while len(tour) < n:
        current = tour[-1]
        # Find the unvisited city that we travel to from the current city
        for j in range(n):
            if solution[current,j] > 0.5 and j not in visited:
                tour.append(j)
                visited.add(j)
                break
    
    return tour