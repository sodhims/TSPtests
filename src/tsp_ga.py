import numpy as np
from typing import List, Tuple, Dict
import random
from enum import Enum
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class CrossoverType(Enum):
    EDGE = "edge"
    ORDER = "order"
    PMX = "pmx"
    EAX = "eax"
    ERX = "erx"

class GAParameters:
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 1000,
        mutation_rate: float = 0.02,
        elitism_rate: float = 0.1,
        crossover_type: CrossoverType = CrossoverType.EDGE,
        random_seed: int = 42
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_type = crossover_type
        self.random_seed = random_seed

class Individual:
    def __init__(self, route: List[int], distance_matrix: np.ndarray):
        self.route = route
        self.fitness = self._calculate_fitness(distance_matrix)
    
    def _calculate_fitness(self, distance_matrix: np.ndarray) -> float:
        total_distance = 0
        for i in range(len(self.route)):
            from_city = self.route[i]
            to_city = self.route[(i + 1) % len(self.route)]
            total_distance += distance_matrix[from_city][to_city]
        return 1 / total_distance

class TSPGeneticAlgorithm:
    def __init__(self, distance_matrix: np.ndarray, params: GAParameters):
        self.distance_matrix = distance_matrix
        self.params = params
        self.num_cities = len(distance_matrix)
        
        # Set random seeds for reproducibility
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)
        
        self.population = self._initialize_population()
        self.best_individual = None
        self.generation_stats = []

    def _initialize_population(self) -> List[Individual]:
        population = []
        for _ in range(self.params.population_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(Individual(route, self.distance_matrix))
        return population

    def _create_edge_table(self, route: List[int]) -> Dict[int, List[int]]:
        edge_table = {i: [] for i in range(self.num_cities)}
        for i in range(len(route)):
            curr_city = route[i]
            prev_city = route[i - 1]
            next_city = route[(i + 1) % len(route)]
            edge_table[curr_city].extend([prev_city, next_city])
        return edge_table

    def _edge_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        # Create edge tables for both parents
        edge_table1 = self._create_edge_table(parent1)
        edge_table2 = self._create_edge_table(parent2)
        
        # Combine edge tables
        combined_edges = {}
        for city in range(self.num_cities):
            edges = list(set(edge_table1[city] + edge_table2[city]))
            combined_edges[city] = edges
            
        # Generate offspring
        offspring = []
        current_city = random.choice(range(self.num_cities))
        
        while len(offspring) < self.num_cities:
            offspring.append(current_city)
            
            # Remove current city from all neighbor lists
            for neighbors in combined_edges.values():
                if current_city in neighbors:
                    neighbors.remove(current_city)
            
            if not combined_edges[current_city]:
                # If no neighbors left, choose random unvisited city
                unvisited = list(set(range(self.num_cities)) - set(offspring))
                current_city = random.choice(unvisited) if unvisited else None
            else:
                # Choose next city with fewest neighbors
                neighbors = combined_edges[current_city]
                current_city = min(neighbors, key=lambda x: len(combined_edges[x]) if x not in offspring else float('inf'))
                
        return offspring

    def _eax_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Edge Assembly Crossover (EAX) operator.
        """
        def create_edge_list(tour):
            edges = set()
            for i in range(len(tour)):
                edge = tuple(sorted([tour[i], tour[(i + 1) % len(tour)]]))
                edges.add(edge)
            return edges

        def find_cycles(p1_edges, p2_edges):
            # Create union graph
            all_edges = p1_edges.union(p2_edges)
            cycles = []
            used_edges = set()
            
            while len(used_edges) < len(all_edges):
                # Find an unused edge to start a new cycle
                start_edge = next(iter(e for e in all_edges if e not in used_edges))
                cycle = []
                current_vertex = start_edge[0]
                current_edge = start_edge
                
                while True:
                    cycle.append(current_edge)
                    used_edges.add(current_edge)
                    next_vertex = current_edge[1] if current_edge[0] == current_vertex else current_edge[0]
                    
                    # Find next edge
                    next_edges = [e for e in all_edges if e not in used_edges and 
                                (e[0] == next_vertex or e[1] == next_vertex)]
                    
                    if not next_edges:
                        break
                        
                    current_edge = next_edges[0]
                    current_vertex = next_vertex
                    
                    if current_edge == start_edge:
                        break
                        
                if cycle:
                    cycles.append(cycle)
                    
            return cycles

        def create_offspring_from_edges(edges, n):
            # Convert edges to adjacency list
            adj = {i: [] for i in range(n)}
            for e in edges:
                adj[e[0]].append(e[1])
                adj[e[1]].append(e[0])
                
            # Create tour
            tour = []
            current = 0  # Start from vertex 0
            used = set()
            
            while len(tour) < n:
                tour.append(current)
                used.add(current)
                
                # Choose next vertex
                next_vertices = [v for v in adj[current] if v not in used]
                if next_vertices:
                    current = next_vertices[0]
                else:
                    # If no adjacent unused vertices, find any unused vertex
                    unused = [v for v in range(n) if v not in used]
                    if unused:
                        current = unused[0]
                        
            return tour

        # Create edge sets from parents
        p1_edges = create_edge_list(parent1)
        p2_edges = create_edge_list(parent2)
        
        # Find AB-cycles
        cycles = find_cycles(p1_edges, p2_edges)
        
        # Create intermediate solutions by combining different cycle combinations
        best_offspring = None
        best_fitness = float('-inf')
        
        # Try different combinations of cycles
        for i in range(min(len(cycles), 8)):  # Limit number of combinations to try
            # Randomly select cycles
            selected_cycles = random.sample(cycles, random.randint(1, min(len(cycles), 4)))
            
            # Create edge set for offspring
            offspring_edges = set(p1_edges)  # Start with parent1 edges
            
            # Apply selected cycles
            for cycle in selected_cycles:
                for edge in cycle:
                    if edge in offspring_edges:
                        offspring_edges.remove(edge)
                    else:
                        offspring_edges.add(edge)
                        
            # Create offspring tour
            offspring_tour = create_offspring_from_edges(offspring_edges, len(parent1))
            
            # Evaluate offspring
            offspring = Individual(offspring_tour, self.distance_matrix)
            
            if offspring.fitness > best_fitness:
                best_fitness = offspring.fitness
                best_offspring = offspring_tour
                
        return best_offspring if best_offspring else parent1  # Fallback to parent1 if no valid offspring

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create offspring with inherited segment from parent1
        offspring = [-1] * size
        offspring[start:end] = parent1[start:end]
        
        # Fill the remaining positions with cities from parent2
        remaining_cities = [city for city in parent2 if city not in offspring[start:end]]
        j = 0
        for i in range(size):
            if offspring[i] == -1:
                offspring[i] = remaining_cities[j]
                j += 1
                
        return offspring

    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Initialize offspring
        offspring = [-1] * size
        
        # Copy the segment from parent1
        offspring[start:end] = parent1[start:end]
        
        # Create mapping of values
        mapping = dict(zip(parent1[start:end], parent2[start:end]))
        
        # Fill in remaining positions
        for i in range(size):
            if i < start or i >= end:
                value = parent2[i]
                while value in offspring[start:end]:
                    value = mapping[value]
                offspring[i] = value
                
        return offspring

    def _erx_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform edge recombination crossover between two parent tours.
        
        Args:
            parent1: First parent tour
            parent2: Second parent tour
            
        Returns:
            Offspring tour
        """
        # Create edge tables for both parents
        edge_table1 = self._create_edge_table(parent1)
        edge_table2 = self._create_edge_table(parent2)
        
        # Create combined edge map
        edge_map = {}
        for city in range(self.num_cities):
            # Combine edges from both parents and remove duplicates
            neighbors = list(set(edge_table1[city] + edge_table2[city]))
            edge_map[city] = neighbors
        
        # Initialize offspring
        offspring = []
        
        # Choose starting city (randomly from either parent's endpoints)
        possible_starts = [parent1[0], parent1[-1], parent2[0], parent2[-1]]
        current_city = random.choice(possible_starts)
        
        # Build offspring
        while len(offspring) < self.num_cities:
            # Add current city to offspring
            offspring.append(current_city)
            
            # Remove current city from all neighbor lists
            for neighbors in edge_map.values():
                if current_city in neighbors:
                    neighbors.remove(current_city)
            
            # Get neighbors of current city
            neighbors = edge_map[current_city]
            
            if not neighbors:
                # If no neighbors left, choose random unvisited city
                unvisited = list(set(range(self.num_cities)) - set(offspring))
                if unvisited:
                    current_city = random.choice(unvisited)
                else:
                    break
            else:
                # Choose neighbor with fewest remaining neighbors
                # If tie, prefer common edges (edges in both parents)
                min_neighbors = float('inf')
                candidates = []
                
                for neighbor in neighbors:
                    if neighbor in offspring:
                        continue
                        
                    num_neighbors = len(edge_map[neighbor])
                    
                    if num_neighbors < min_neighbors:
                        min_neighbors = num_neighbors
                        candidates = [neighbor]
                    elif num_neighbors == min_neighbors:
                        candidates.append(neighbor)
                
                if candidates:
                    # If multiple candidates, prefer edges present in both parents
                    common_edges = [c for c in candidates if 
                                (current_city in edge_table1[c] and current_city in edge_table2[c])]
                    if common_edges:
                        current_city = random.choice(common_edges)
                    else:
                        current_city = random.choice(candidates)
                else:
                    # If no valid neighbors, choose random unvisited city
                    unvisited = list(set(range(self.num_cities)) - set(offspring))
                    if unvisited:
                        current_city = random.choice(unvisited)
                    else:
                        break
        
        return offspring
    
    def _mutate(self, route: List[int]) -> List[int]:
        if random.random() < self.params.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        if self.params.crossover_type == CrossoverType.EDGE:
            offspring_route = self._edge_crossover(parent1.route, parent2.route)
        elif self.params.crossover_type == CrossoverType.ORDER:
            offspring_route = self._order_crossover(parent1.route, parent2.route)
        elif self.params.crossover_type == CrossoverType.EAX:
            offspring_route = self._eax_crossover(parent1.route, parent2.route)
        elif self.params.crossover_type == CrossoverType.ERX:
            offspring_route = self._erx_crossover(parent1.route, parent2.route)            
        else:  # PMX
            offspring_route = self._pmx_crossover(parent1.route, parent2.route)
            
        offspring_route = self._mutate(offspring_route)
        return Individual(offspring_route, self.distance_matrix)

    def _select_parent(self) -> Individual:
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def evolve(self, show_progress: bool = True) -> Tuple[Individual, List[dict]]:
        iterator = range(self.params.generations)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Evolution ({self.params.crossover_type.value})")
        
        for generation in iterator:
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Store best individual
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0]
                if show_progress:
                    iterator.set_postfix({"Best Distance": f"{1/self.best_individual.fitness:.2f}"})
            
            # Store generation statistics
            self.generation_stats.append({
                'generation': generation,
                'best_fitness': self.population[0].fitness,
                'avg_fitness': sum(ind.fitness for ind in self.population) / len(self.population),
                'best_distance': 1 / self.population[0].fitness
            })
            
            # Create new population
            elite_size = int(self.params.population_size * self.params.elitism_rate)
            new_population = self.population[:elite_size]
            
            # Fill rest of population with offspring
            while len(new_population) < self.params.population_size:
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                offspring = self._crossover(parent1, parent2)
                new_population.append(offspring)
                
            self.population = new_population
            
        return self.best_individual, self.generation_stats
def _create_edge_table(self, tour: List[int]) -> Dict[int, List[int]]:
    """
    Create an edge table for a given tour.
    
    Args:
        tour: List of cities representing a tour
        
    Returns:
        Dictionary mapping each city to its neighbors in the tour
    """
    edge_table = {city: [] for city in range(self.num_cities)}
    n = len(tour)
    
    for i in range(n):
        city = tour[i]
        # Add connections to previous and next cities in tour
        prev_city = tour[(i - 1) % n]
        next_city = tour[(i + 1) % n]
        edge_table[city].extend([prev_city, next_city])
    
    return edge_table



def plot_convergence(stats: List[dict], title: str = "Convergence Plot"):
    """Plot the convergence of the GA over generations."""
    generations = [s['generation'] for s in stats]
    best_distances = [s['best_distance'] for s in stats]
    avg_distances = [1/s['avg_fitness'] for s in stats]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_distances, label='Best Distance')
    plt.plot(generations, avg_distances, label='Average Distance')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()