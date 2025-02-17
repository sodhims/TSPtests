import numpy as np
from src.tsp_ga import TSPGeneticAlgorithm, GAParameters, CrossoverType

# Create a small test problem
num_cities = 10
np.random.seed(42)
distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) // 2  # Make symmetric

# Set up parameters
params = GAParameters(
    population_size=50,
    generations=100,
    mutation_rate=0.02,
    elitism_rate=0.1,
    crossover_type=CrossoverType.EDGE,
    random_seed=42
)

# Create and run GA
ga = TSPGeneticAlgorithm(distance_matrix, params)
best_solution, stats = ga.evolve()

# Print results
print(f"Best distance found: {1/best_solution.fitness:.2f}")
print(f"Best route: {best_solution.route}")