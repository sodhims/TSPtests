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
    ERX = "erx"

class GAParameters:
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 1000,
        mutation_rate: float = 0.02,
        elitism_rate: float = 0.1,
        crossover_type: CrossoverType = CrossoverType.EDGE,
        tournament_size: int = 3,
        random_seed: int = 42
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_type = crossover_type
        self.tournament_size = tournament_size
        self.random_seed = random_seed

class Individual:
    __slots__ = ['route', 'distance_matrix', 'fitness']
    
    def __init__(self, route: np.ndarray, distance_matrix: np.ndarray):
        self.route = route
        self.distance_matrix = distance_matrix
        self.fitness = self._calculate_fitness()
    
    def _calculate_fitness(self) -> float:
        route_shifted = np.roll(self.route, -1)
        total_distance = np.sum(self.distance_matrix[self.route, route_shifted])
        return 1 / total_distance

class TSPGeneticAlgorithm:
    def __init__(self, distance_matrix: np.ndarray, params: GAParameters):
        self.distance_matrix = distance_matrix
        self.params = params
        self.num_cities = len(distance_matrix)
        
        # Set random seeds
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)
        
        # Pre-compute frequently used values
        self.elite_size = int(self.params.population_size * self.params.elitism_rate)
        
        # Initialize arrays for efficient computation
        self.population = self._initialize_population()
        self.best_individual = None
        self.generation_stats = []

    def _initialize_population(self) -> List[Individual]:
        routes = [np.random.permutation(self.num_cities) 
                 for _ in range(self.params.population_size)]
        return [Individual(route, self.distance_matrix) for route in routes]

    def _order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        size = len(parent1)
        points = random.sample(range(size), 2)
        start, end = min(points), max(points)
        
        offspring = np.full(size, -1)
        offspring[start:end] = parent1[start:end]
        
        remaining = parent2[~np.isin(parent2, offspring[start:end])]
        offspring[offspring == -1] = remaining
        
        return offspring

    def _edge_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        edge_table = np.zeros((self.num_cities, 4), dtype=np.int32) - 1
        
        # Build edge table efficiently
        for parent in [parent1, parent2]:
            for i in range(self.num_cities):
                curr = parent[i]
                prev = parent[i - 1]
                next_city = parent[(i + 1) % self.num_cities]
                
                idx = np.where(edge_table[curr] == -1)[0][0]
                edge_table[curr, idx] = prev
                
                idx = np.where(edge_table[curr] == -1)[0][0]
                edge_table[curr, idx] = next_city
        
        offspring = np.zeros(self.num_cities, dtype=np.int32)
        current = random.randrange(self.num_cities)
        used = np.zeros(self.num_cities, dtype=bool)
        
        for i in range(self.num_cities):
            offspring[i] = current
            used[current] = True
            
            neighbors = edge_table[current]
            valid_neighbors = neighbors[(neighbors != -1) & ~used[neighbors]]
            
            if len(valid_neighbors) == 0:
                unused = np.where(~used)[0]
                current = unused[0] if len(unused) > 0 else offspring[0]
                continue
            
            neighbor_counts = np.array([
                np.sum((edge_table[n] != -1) & ~used[edge_table[n]])
                for n in valid_neighbors
            ])
            current = valid_neighbors[np.argmin(neighbor_counts)]
        
        return offspring

    def _pmx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        size = len(parent1)
        points = random.sample(range(size), 2)
        start, end = min(points), max(points)
        
        offspring = np.full(size, -1)
        offspring[start:end] = parent1[start:end]
        
        # Create mapping
        p1_segment = parent1[start:end]
        p2_segment = parent2[start:end]
        mapping = dict(zip(p2_segment, p1_segment))
        
        # Fill remaining positions
        for i in range(size):
            if i < start or i >= end:
                value = parent2[i]
                while value in p1_segment:
                    value = mapping[value]
                offspring[i] = value
        
        return offspring

    def _erx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # Simplified ERX implementation using numpy operations
        edge_table = np.zeros((self.num_cities, 4), dtype=np.int32) - 1
        
        for parent in [parent1, parent2]:
            for i in range(self.num_cities):
                curr = parent[i]
                prev = parent[i - 1]
                next_city = parent[(i + 1) % self.num_cities]
                
                idx = np.where(edge_table[curr] == -1)[0][0]
                edge_table[curr, idx] = prev
                idx = np.where(edge_table[curr] == -1)[0][0]
                edge_table[curr, idx] = next_city
        
        offspring = np.zeros(self.num_cities, dtype=np.int32)
        current = random.choice([parent1[0], parent1[-1], parent2[0], parent2[-1]])
        used = np.zeros(self.num_cities, dtype=bool)
        
        for i in range(self.num_cities):
            offspring[i] = current
            used[current] = True
            
            neighbors = edge_table[current]
            valid_neighbors = neighbors[(neighbors != -1) & ~used[neighbors]]
            
            if len(valid_neighbors) == 0:
                unused = np.where(~used)[0]
                current = unused[0] if len(unused) > 0 else offspring[0]
                continue
            
            current = valid_neighbors[np.argmin([
                np.sum((edge_table[n] != -1) & ~used[edge_table[n]])
                for n in valid_neighbors
            ])]
        
        return offspring

    def _mutate(self, route: np.ndarray) -> np.ndarray:
        if random.random() < self.params.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        crossover_funcs = {
            CrossoverType.EDGE: self._edge_crossover,
            CrossoverType.ORDER: self._order_crossover,
            CrossoverType.PMX: self._pmx_crossover,
            CrossoverType.ERX: self._erx_crossover
        }
        
        crossover_func = crossover_funcs.get(self.params.crossover_type)
        if not crossover_func:
            raise ValueError(f"Unknown crossover type: {self.params.crossover_type}")
            
        offspring_route = crossover_func(parent1.route, parent2.route)
        offspring_route = self._mutate(offspring_route)
        return Individual(offspring_route, self.distance_matrix)

    def _select_parent(self) -> Individual:
        tournament = random.sample(self.population, self.params.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def evolve(self, show_progress: bool = True) -> Tuple[Individual, List[dict]]:
        iterator = range(self.params.generations)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Evolution ({self.params.crossover_type.value})")
        
        for generation in iterator:
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best individual
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0]
                if show_progress:
                    iterator.set_postfix({"Best Distance": f"{1/self.best_individual.fitness:.2f}"})
            
            # Store generation statistics
            self.generation_stats.append({
                'generation': generation,
                'best_fitness': self.population[0].fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population]),
                'best_distance': 1 / self.population[0].fitness
            })
            
            # Create new population
            new_population = self.population[:self.elite_size]
            
            # Generate offspring
            while len(new_population) < self.params.population_size:
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                offspring = self._crossover(parent1, parent2)
                new_population.append(offspring)
                
            self.population = new_population
            
        return self.best_individual, self.generation_stats

def plot_convergence(stats: List[dict], title: str = "Convergence Plot"):
    data = pd.DataFrame(stats)
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['generation'], data['best_distance'], label='Best Distance')
    plt.plot(data['generation'], 1/data['avg_fitness'], label='Average Distance')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()