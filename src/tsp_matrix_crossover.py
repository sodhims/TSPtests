import numpy as np
import random
import time
from typing import List, Tuple, Dict
import json

class TSPGA:
    def __init__(self, cities: List[Tuple[float, float]], 
                 population_size: int = 100,
                 mutation_rate: float = 0.02,
                 elite_size: int = 2):
        
        # Set random seed
        np.random.seed(42)
        random.seed(42)
        
        # Initialize parameters
        self.num_cities = len(cities)
        self.cities = np.array(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Initialize population with random tours
        self.population = self._initialize_population()
        self.best_distance = float('inf')
        self.best_tour = None
        self.generation = 0
        self.history = {'best': [], 'avg': [], 'worst': []}

    def _initialize_population(self) -> List[np.ndarray]:
        """Create initial population of random adjacency matrices"""
        population = []
        for _ in range(self.population_size):
            # Create random tour
            tour = list(range(self.num_cities))
            random.shuffle(tour)
            
            # Convert to adjacency matrix
            matrix = np.zeros((self.num_cities, self.num_cities), dtype=int)
            for i in range(self.num_cities):
                matrix[tour[i]][tour[(i + 1) % self.num_cities]] = 1
                
            population.append(matrix)
        return population

    def calculate_fitness(self, matrix: np.ndarray) -> float:
        """Calculate fitness (1/distance) for a tour matrix"""
        tour = self.matrix_to_tour(matrix)
        distance = self.calculate_tour_distance(tour)
        return 1.0 / distance if distance > 0 else float('inf')

    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of a tour"""
        total_distance = 0
        for i in range(len(tour)):
            city1 = self.cities[tour[i]]
            city2 = self.cities[tour[(i + 1) % len(tour)]]
            distance = np.sqrt(np.sum((city1 - city2) ** 2))
            total_distance += distance
        return float(total_distance)

    def matrix_to_tour(self, matrix: np.ndarray) -> List[int]:
        """Convert adjacency matrix to tour"""
        tour = [0]  # Start from city 0
        visited = {0}
        current = 0
        
        while len(tour) < self.num_cities:
            # Find next unvisited city
            next_cities = np.where(matrix[current] == 1)[0]
            if len(next_cities) == 0:
                # If no valid next city, fix the matrix
                unvisited = list(set(range(self.num_cities)) - visited)
                if unvisited:
                    next_city = random.choice(unvisited)
                    matrix[current, :] = 0
                    matrix[current, next_city] = 1
                    tour.append(next_city)
                    visited.add(next_city)
                    current = next_city
                else:
                    # Complete the tour back to start
                    matrix[current, 0] = 1
                    break
            else:
                next_city = next_cities[0]
                if next_city not in visited:
                    tour.append(next_city)
                    visited.add(next_city)
                    current = next_city
                else:
                    # Fix invalid next city
                    unvisited = list(set(range(self.num_cities)) - visited)
                    if unvisited:
                        next_city = random.choice(unvisited)
                        matrix[current, :] = 0
                        matrix[current, next_city] = 1
                        tour.append(next_city)
                        visited.add(next_city)
                        current = next_city
                    else:
                        # Complete the tour back to start
                        matrix[current, 0] = 1
                        break
        
        return tour

    def repair_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix represents a valid tour"""
        repaired = matrix.copy()
        
        # Ensure one outgoing and one incoming edge per city
        for i in range(self.num_cities):
            # Fix outgoing edges
            if np.sum(repaired[i]) != 1:
                repaired[i] = 0
                available = [j for j in range(self.num_cities)
                           if np.sum(repaired[:, j]) == 0 and i != j]
                if not available:
                    available = [j for j in range(self.num_cities) if i != j]
                next_city = random.choice(available)
                repaired[i, next_city] = 1
            
            # Fix incoming edges
            if np.sum(repaired[:, i]) != 1:
                repaired[:, i] = 0
                available = [j for j in range(self.num_cities)
                           if np.sum(repaired[j]) == 0 and i != j]
                if not available:
                    available = [j for j in range(self.num_cities) if i != j]
                prev_city = random.choice(available)
                repaired[prev_city, i] = 1
        
        return repaired
    
    # Add this method to your TSPGA class:

    def select_parent(self, fitnesses: List[float]) -> int:
        """Tournament selection for parent selection
        
        Args:
            fitnesses: List of fitness values for the population
            
        Returns:
            Index of the selected parent
        """
        # Tournament selection with size 3
        tournament_size = 3
        # Randomly select tournament_size individuals
        tournament = random.sample(range(len(fitnesses)), tournament_size)
        # Return the best one
        return max(tournament, key=lambda i: fitnesses[i])
    

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform union crossover with repair"""
        # Create child using union
        child = np.logical_or(parent1, parent2).astype(int)
        
        # Repair the child to ensure valid tour
        child = self.repair_matrix(child)
        
        return child

    def mutate(self, matrix: np.ndarray) -> np.ndarray:
        """Swap mutation on matrix representation"""
        if random.random() < self.mutation_rate:
            # Convert to tour for easier mutation
            tour = self.matrix_to_tour(matrix)
            
            # Perform swap mutation
            i, j = random.sample(range(self.num_cities), 2)
            tour[i], tour[j] = tour[j], tour[i]
            
            # Convert back to matrix
            mutated = np.zeros_like(matrix)
            for i in range(len(tour)):
                mutated[tour[i]][tour[(i + 1) % len(tour)]] = 1
            return mutated
        return matrix

    def evolve(self, generations: int = 1000) -> Dict:
        """Main evolution loop"""
        start_time = time.time()
        
        for gen in range(generations):
            self.generation = gen
            
            # Calculate fitness for all individuals
            fitnesses = [self.calculate_fitness(matrix) for matrix in self.population]
            
            # Find best solution in current generation
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            best_tour = self.matrix_to_tour(self.population[best_idx])
            current_best_distance = self.calculate_tour_distance(best_tour)
            
            # Update best solution found
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_tour = best_tour
            
            # Store statistics
            distances = [1.0/f for f in fitnesses]
            self.history['best'].append(min(distances))
            self.history['avg'].append(sum(distances)/len(distances))
            self.history['worst'].append(max(distances))
            
            # Create new population
            new_population = []
            
            # Elitism - keep best solutions
            sorted_indices = sorted(range(len(fitnesses)), 
                                 key=lambda i: fitnesses[i],
                                 reverse=True)
            for i in range(self.elite_size):
                new_population.append(self.population[sorted_indices[i]].copy())
            
            # Create rest of new population
            while len(new_population) < self.population_size:
                # Select parents
                parent1_idx = self.select_parent(fitnesses)
                parent2_idx = self.select_parent(fitnesses)
                
                # Create child
                child = self.crossover(self.population[parent1_idx],
                                     self.population[parent2_idx])
                
                # Mutate child
                child = self.mutate(child)
                
                # Ensure child is valid
                child = self.repair_matrix(child)
                
                new_population.append(child)
            
            self.population = new_population
            
            # Print progress
            if gen % 100 == 0:
                print(f"Generation {gen}: Best Distance = {self.best_distance:.2f}")
        
        execution_time = time.time() - start_time
        
        return {
            'best_tour': self.best_tour,
            'best_distance': self.best_distance,
            'history': self.history,
            'execution_time': execution_time
        }

def save_cities_to_json(cities: List[Tuple[float, float]], filename: str):
    """
    Save city coordinates to a JSON file
    
    Args:
        cities: List of (x,y) coordinates
        filename: Name of output JSON file
    """
    # Create dictionary with city data
    city_data = {
        'num_cities': len(cities),
        'cities': [{'x': x, 'y': y} for x, y in cities]
    }
    
    try:
        # Write to JSON file with nice formatting
        with open(filename, 'w') as f:
            json.dump(city_data, f, indent=4)
        print(f"Successfully saved {len(cities)} cities to {filename}")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")

def load_cities_from_json(filename: str) -> List[Tuple[float, float]]:
    """
    Load city coordinates from a JSON file
    
    Args:
        filename: Name of JSON file to load
        
    Returns:
        List of (x,y) coordinates
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            cities = [(city['x'], city['y']) for city in data['cities']]
            print(f"Successfully loaded {len(cities)} cities from {filename}")
            print("the number of cities is ",num_cities)
            print("\nFirst 5 of cities*********")
            for i in range(5):
                x, y = cities[i]
                print(f"City {i}: (x={x:.2f}, y={y:.2f})")
            return cities
    except Exception as e:
        print(f"Error loading from {filename}: {e}")
        return []
    
if __name__ == "__main__":
    # Create random cities
    num_cities = 20
    random.seed(42)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) 
              for _ in range(num_cities)]
    save_cities_to_json(cities, "cities.json")

# Load cities for GA
    cities = load_cities_from_json("cities.json")
    # Create and run GA
    ga = TSPGA(cities, population_size=100, mutation_rate=0.02, elite_size=2)
    results = ga.evolve(generations=1000)
    
    print("\nResults:")
    print(f"Best Distance: {results['best_distance']:.2f}")
    print(f"Best Tour: {results['best_tour']}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    
    # Plot best tour
    try:
        import matplotlib.pyplot as plt
        
        best_tour = results['best_tour']
        coords = np.array([cities[i] for i in best_tour + [best_tour[0]]])
        
        plt.figure(figsize=(10, 10))
        plt.plot(coords[:, 0], coords[:, 1], 'b-')
        plt.plot(coords[:, 0], coords[:, 1], 'r.')
        plt.title(f'Best Tour (Distance: {results["best_distance"]:.2f})')
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")