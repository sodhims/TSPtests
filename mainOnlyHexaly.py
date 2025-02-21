
import numpy as np
import json
import os
import time
import hexaly.optimizer as hexaly

def load_cities_and_create_matrix(filename: str):
    """
    Load cities from JSON file and create distance matrix.
    
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
            for j in range(i + 1, num_cities):
                # Get coordinates
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                
                # Calculate Euclidean distance
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # Make matrix symmetric
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        print(f"Successfully created distance matrix for {num_cities} cities")
        return distance_matrix, cities
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

def solve_tsp_hexaly(distance_matrix):
    """
    Solves the TSP using Hexaly's optimizer.

    Args:
        distance_matrix: Numpy array representing distances between cities.

    Returns:
        tuple: (optimal_route, optimal_distance)
    """
    # Initialize optimizer with parameters
    params = {
        'mip_gap': 0.01,  # 1% optimality gap
        'time_limit': 3600  # 1 hour time limit
    }

    with hexaly.HexalyOptimizer() as optimizer:
        # Create a model
        model = optimizer.model
        num_cities = len(distance_matrix)

        # Declare list variable representing the tour
        cities = model.list(num_cities)

        # All cities must be visited exactly once
        model.constraint(model.count(cities) == num_cities)

        # Create a Hexaly array for the distance matrix
        dist_matrix = model.array(distance_matrix.tolist())

        # Objective function: Minimize the total distance
        dist_lambda = model.lambda_function(lambda i:
                                            model.at(dist_matrix, cities[i - 1], cities[i]))
        obj = model.sum(model.range(1, num_cities), dist_lambda) + \
              model.at(dist_matrix, cities[num_cities - 1], cities[0])
        model.minimize(obj)

        # Close the model (Hexaly-specific)
        model.close()



        # Solve the model
        optimizer.solve()

        # Extract the optimal distance and route
        optimal_distance = obj.value
        optimal_route = [0]
        for c in cities.value:
            optimal_route.append(c+1)

        return optimal_route, optimal_distance

if __name__ == "__main__":
    # Specify parameters
    num_cities = 200  # Change this to your desired number
    random_seed = 63  # Change this to your desired seed
    
    # Construct the filename
    filename = f"cities{num_cities:03d}_{random_seed:02d}.json"
    
    print(f"Looking for file: {filename}")
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        exit()
    
    print(f"\nProcessing {filename}...")
    
    # Load and create distance matrix
    distance_matrix, cities = load_cities_and_create_matrix(filename)

        
    if distance_matrix is not None:
        # Print matrix statistics
        print(f"\nDistance matrix statistics:")
        print(f"Min distance: {np.min(distance_matrix[distance_matrix > 0]):.2f}")
        print(f"Max distance: {np.max(distance_matrix):.2f}")
        print(f"Average distance: {np.mean(distance_matrix[distance_matrix > 0]):.2f}")
        
        # Solve with Hexaly
        print("\nSolving with Hexaly...")
        start_time = time.time()
        optimal_route, optimal_distance = solve_tsp_hexaly(distance_matrix)
        solution_time = time.time() - start_time
        
        # Print results
        print("\nHexaly Solution:")
        print(f"Optimal distance: {optimal_distance:.2f}")
        print(f"Solution time: {solution_time:.2f} seconds")
        print(f"Optimal route: {optimal_route}")
        
        print("\n" + "=" * 50)


