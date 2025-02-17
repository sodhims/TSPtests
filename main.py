import numpy as np
from src.tsp_ga import TSPGeneticAlgorithm, GAParameters, CrossoverType, plot_convergence
from src.tsp_gurobi import solve_tsp_gurobi
import time

# Create a small test problem
num_cities = 20
np.random.seed(42)  # Same seed for both methods
distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) // 2  # Make symmetric

# Run Genetic Algorithm
print("\nRunning Genetic Algorithm...")
start_time = time.time()

params = GAParameters(
    population_size=100,
    generations=200,
    mutation_rate=0.02,
    elitism_rate=0.1,
    crossover_type=CrossoverType.EDGE,
    random_seed=42
)

ga = TSPGeneticAlgorithm(distance_matrix, params)
best_solution, stats = ga.evolve()

ga_time = time.time() - start_time
ga_distance = 1/best_solution.fitness
ga_route = best_solution.route

# Run Gurobi
print("\nRunning Gurobi Solver...")
start_time = time.time()
gurobi_route, gurobi_distance = solve_tsp_gurobi(distance_matrix, time_limit=300)  # 5 minute limit
gurobi_time = time.time() - start_time

# Print comparison
print("\nResults Comparison:")
print("-" * 50)
print(f"Genetic Algorithm:")
print(f"Distance: {ga_distance:.2f}")
print(f"Time: {ga_time:.2f} seconds")
print(f"Route: {ga_route}")
print("\nGurobi (Exact):")
print(f"Distance: {gurobi_distance:.2f}")
print(f"Time: {gurobi_time:.2f} seconds")
print(f"Route: {gurobi_route}")
print("\nGA vs Optimal Gap:")
gap = ((ga_distance - gurobi_distance) / gurobi_distance) * 100
print(f"Gap: {gap:.2f}%")

# Show convergence plot
plot_convergence(stats, "TSP Convergence Over Generations")