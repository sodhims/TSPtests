import numpy as np
from src.tsp_ga import TSPGeneticAlgorithm, GAParameters, CrossoverType, plot_convergence
import time
import pandas as pd
from itertools import product
import json
import os

def generate_cities(num_cities: int, seed: int):
    """
    Generate random city coordinates and save to JSON
    
    Args:
        num_cities: Number of cities to generate
        seed: Random seed for reproducibility
        
    Returns:
        str: Filename where cities were saved
    """
    np.random.seed(seed)
    
    # Generate random coordinates between 0 and 100
    cities = []
    for i in range(num_cities):
        cities.append({
            'x': float(np.random.uniform(0, 100)),
            'y': float(np.random.uniform(0, 100))
        })
    
    # Create filename with format cities{num_cities}_{seed}.json
    filename = f"cities{num_cities:03d}_{seed:02d}.json"
    
    # Save to JSON
    with open(filename, 'w') as f:
        json.dump({'cities': cities}, f)
    
    print(f"Generated {num_cities} cities and saved to {filename}")
    return filename

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
        
        print(f"Successfully loaded data from {filename}")
        
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
    # Set parameters
    num_cities = 150  # You can modify this
    random_seed = 42  # You can modify this
    
    # Generate and save cities
    filename = generate_cities(num_cities, random_seed)
    
    # Load and create distance matrix
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

        # Define parameter levels for DOE
        param_levels = {
            'population_size': [500],
            'generations': [2000],
            'mutation_rate': [0.1],
            'elitism_rate': [0.1],
            'crossover_type': [CrossoverType.EDGE, CrossoverType.ERX]  
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
                random_seed=random_seed
            )
            
            # Run GA
            start_time = time.time()
            ga = TSPGeneticAlgorithm(distance_matrix, params)
            best_solution, stats = ga.evolve()
            solution_time = time.time() - start_time
            
            # Calculate metrics
            final_distance = 1/best_solution.fitness
            
            # Store results
            results.append({
                'Population Size': pop_size,
                'Generations': gens,
                'Mutation Rate': mut_rate,
                'Elitism Rate': elit_rate,
                'Crossover Type': cross_type.value,
                'Final Distance': final_distance,
                'Solution Time': solution_time,
                'Best Route': best_solution.route
            })

        # Create results DataFrame
        df_results = pd.DataFrame(results)

        # Sort by Final Distance
        df_results = df_results.sort_values('Final Distance')

        # Display summary (excluding route)
        summary_cols = ['Population Size', 'Generations', 'Mutation Rate', 'Elitism Rate', 
                       'Crossover Type', 'Final Distance', 'Solution Time']
        print("\nResults Summary (sorted by Final Distance):")
        print("=" * 100)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.precision', 2)
        print(df_results[summary_cols])

        # Save results to Excel with new naming convention
        results_filename = f"experiment_results_{num_cities:03d}_{random_seed:02d}.xlsx"
        df_results.to_excel(results_filename, index=False)
        print(f"\nDetailed results saved to '{results_filename}'")

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
        print(f"Solution Time: {best_result['Solution Time']:.2f} seconds")