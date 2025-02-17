import numpy as np
from src.tsp_ga import TSPGeneticAlgorithm, GAParameters, CrossoverType, plot_convergence
from src.tsp_gurobi import solve_tsp_gurobi
import time
import pandas as pd
from itertools import product

# Create test problem (keep this constant for all experiments)
num_cities = 20
np.random.seed(42)
distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) // 2

# Run Gurobi once for optimal solution
print("\nSolving with Gurobi...")
start_time = time.time()
gurobi_route, gurobi_distance = solve_tsp_gurobi(distance_matrix, time_limit=300)
gurobi_time = time.time() - start_time
print(f"Gurobi optimal distance: {gurobi_distance:.2f}")
print(f"Gurobi solution time: {gurobi_time:.2f} seconds")

# Define parameter levels for DOE
param_levels = {
    'population_size': [100, 200],
    'generations': [200, 500],
    'mutation_rate': [0.01, 0.05],
    'elitism_rate': [0.05, 0.15],
    'crossover_type': [CrossoverType.EDGE, CrossoverType.ORDER, CrossoverType.PMX]
}

# Generate all combinations
experiments = list(product(
    param_levels['population_size'],
    param_levels['generations'],
    param_levels['mutation_rate'],
    param_levels['elitism_rate'],
    param_levels['crossover_type']
))

# Store results
results = []

# Run experiments
total_experiments = len(experiments)
for i, (pop_size, gens, mut_rate, elit_rate, cross_type) in enumerate(experiments, 1):
    print(f"\nRunning experiment {i}/{total_experiments}")
    print(f"Parameters: pop={pop_size}, gen={gens}, mut={mut_rate}, elit={elit_rate}, cross={cross_type.value}")
    
    # Set up parameters
    params = GAParameters(
        population_size=pop_size,
        generations=gens,
        mutation_rate=mut_rate,
        elitism_rate=elit_rate,
        crossover_type=cross_type,
        random_seed=42
    )
    
    # Run GA
    start_time = time.time()
    ga = TSPGeneticAlgorithm(distance_matrix, params)
    best_solution, stats = ga.evolve()
    solution_time = time.time() - start_time
    
    # Calculate metrics
    final_distance = 1/best_solution.fitness
    gap = ((final_distance - gurobi_distance) / gurobi_distance) * 100
    
    # Store results
    results.append({
        'Population Size': pop_size,
        'Generations': gens,
        'Mutation Rate': mut_rate,
        'Elitism Rate': elit_rate,
        'Crossover Type': cross_type.value,
        'Final Distance': final_distance,
        'Solution Time': solution_time,
        'Gap to Optimal (%)': gap,
        'Best Route': best_solution.route
    })
    
    # Plot convergence for this experiment
    plot_convergence(stats, 
                    f"Convergence for pop={pop_size}, gen={gens}, mut={mut_rate}, "
                    f"elit={elit_rate}, cross={cross_type.value}")

# Create results DataFrame
df_results = pd.DataFrame(results)

# Sort by Gap to Optimal
df_results = df_results.sort_values('Gap to Optimal (%)')

# Display summary (excluding route)
summary_cols = ['Population Size', 'Generations', 'Mutation Rate', 'Elitism Rate', 
                'Crossover Type', 'Final Distance', 'Solution Time', 'Gap to Optimal (%)']
print("\nResults Summary (sorted by Gap to Optimal):")
print("=" * 100)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 2)
print(df_results[summary_cols])

# Save results to CSV
df_results.to_csv('experiment_results.csv', index=False)
print("\nDetailed results saved to 'experiment_results.csv'")

# Print best configuration
best_result = df_results.iloc[0]
print("\nBest Configuration:")
print("=" * 50)
print(f"Population Size: {best_result['Population Size']}")
print(f"Generations: {best_result['Generations']}")
print(f"Mutation Rate: {best_result['Mutation Rate']}")
print(f"Elitism Rate: {best_result['Elitism Rate']}")
print(f"Crossover Type: {best_result['Crossover Type']}")
print(f"Final Distance: {best_result['Final Distance']:.2f}")
print(f"Gap to Optimal: {best_result['Gap to Optimal (%)']:.2f}%")
print(f"Solution Time: {best_result['Solution Time']:.2f} seconds")