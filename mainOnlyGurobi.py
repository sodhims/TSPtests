import numpy as np
from src.tsp_gurobi import solve_tsp_gurobi
import json
import os
import time

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
    # Get list of JSON files in current directory
    json_files = [f for f in os.listdir('.') if f.startswith('cities') and f.endswith('.json')]
    
    if not json_files:
        print("No cities*.json files found in current directory")
        exit()
    
    # Sort files to process them in order
    json_files.sort()
    
    # Process each file
    for filename in json_files:
        print(f"\nProcessing {filename}...")
        
        # Load and create distance matrix
        distance_matrix, cities = load_cities_and_create_matrix(filename)
        
        if distance_matrix is not None:
            # Print matrix statistics
            print(f"\nDistance matrix statistics:")
            print(f"Min distance: {np.min(distance_matrix[distance_matrix > 0]):.2f}")
            print(f"Max distance: {np.max(distance_matrix):.2f}")
            print(f"Average distance: {np.mean(distance_matrix[distance_matrix > 0]):.2f}")
            
            # Solve with Gurobi
            print("\nSolving with Gurobi...")
            start_time = time.time()
            gurobi_route, gurobi_distance = solve_tsp_gurobi(distance_matrix, time_limit=3000)
            solution_time = time.time() - start_time
            
            # Print results
            print("\nGurobi Solution:")
            print(f"Optimal distance: {gurobi_distance:.2f}")
            print(f"Solution time: {solution_time:.2f} seconds")
            print(f"Optimal route: {gurobi_route}")
            
            print("\n" + "="*50)