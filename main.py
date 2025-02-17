# main.py
import numpy as np
from src.tsp_ga import (
    TSPGeneticAlgorithm, 
    GAParameters, 
    CrossoverType, 
    plot_convergence,
    run_parameter_optimization,
    plot_parameter_comparison
)

def main():
    # Create sample distance matrix
    num_cities = 20
    np.random.seed(42)
    distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) // 2  # Make symmetric
    
    # Run single GA instance
    print("\nRunning single GA instance...")
    params = GAParameters(random_seed=42)
    ga = TSPGeneticAlgorithm(distance_