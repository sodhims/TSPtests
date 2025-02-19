import numpy as np
from src.tsp_ga import TSPGeneticAlgorithm, GAParameters, CrossoverType, plot_convergence
from src.tsp_gurobi import solve_tsp_gurobi
import time
import pandas as pd
from itertools import product
import json
import os


def load_cities_and_create_matrix(filename: str):
    """
    Load cities from JSON file and create distance matrix
    
    Args:
        filename: Path to JSON file containing city coordinates
        
    Returns:
        tuple: (distance_matrix, cities)
    """
    try:
        # Read JSON file
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Extract city coordinates
        cities = [(city['x'], city['y']) for city in data['cities']]
        num_cities = len(cities)
        for i in range(5):
            x, y = cities[i]
            print(f"City {i}: (x={x:.2f}, y={y:.2f})")
        
        # Create distance matrix
        distance_matrix = np.zeros((num_cities, num_cities))
        
        # Calculate distances
        for i in range(num_cities):
            for j in range(i+1, num_cities):
                # Get coordinates
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                
                # Calculate Euclidean distance
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Make matrix symmetric
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        print(f"Successfully created distance matrix for {num_cities} cities")
        return distance_matrix, cities
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filename}")
        return None, None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

if __name__ == "__main__":
    # Load and create distance matrix
    current_dir = os.getcwd()

# Method 1: Join paths using os.path.join
    filename = os.path.join(current_dir, "cities.json")

    distance_matrix, cities = load_cities_and_create_matrix(filename)
    
    if distance_matrix is not None:
        # Print first 5x5 of distance matrix
        print("\nFirst 5x5 of distance matrix:")
        print(distance_matrix[:5, :5])
        
        # Print some statistics
        print(f"\nDistance matrix statistics:")
        print(f"Min distance: {np.min(distance_matrix[distance_matrix > 0]):.2f}")
        print(f"Max distance: {np.max(distance_matrix):.2f}")
        print(f"Average distance: {np.mean(distance_matrix[distance_matrix > 0]):.2f}")    
# Create test problem (keep this constant for all experiments)
""" num_cities = 20
np.random.seed(42)
distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) // 2 """


# Run Gurobi once for optimal solution
print("\nSolving with Gurobi...")
start_time = time.time()
gurobi_route, gurobi_distance = solve_tsp_gurobi(distance_matrix, time_limit=3000)
gurobi_time = time.time() - start_time
print(f"Gurobi optimal distance: {gurobi_distance:.2f}")
print(f"Gurobi solution time: {gurobi_time:.2f} seconds")

# Define parameter levels for DOE
# In your parameter levels, change to use EAX
param_levels = {
    'population_size': [200, 500],
    'generations': [1000, 2000],
    'mutation_rate': [0.05, 0.1],
    'elitism_rate': [0.05, 0.1],
    'crossover_type': [CrossoverType.EDGE,CrossoverType.ERX]  
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
    
    # Plot convergence for this experiment  - removed by MSS to speed run
#    plot_convergence(stats, 
#                    f"Convergence for pop={pop_size}, gen={gens}, mut={mut_rate}, "
#                    f"elit={elit_rate}, cross={cross_type.value}")

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
df_results.to_csv(os.path.join(current_dir,"experiment_results.csv"), index=False)
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