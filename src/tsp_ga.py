# src/tsp_ga.py
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
    PMX = "pmx"  # Partially Mapped Crossover

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

def run_parameter_optimization(
    distance_matrix: np.ndarray,
    mutation_rates: List[float],
    elitism_rates: List[float],
    crossover_types: List[CrossoverType],
    num_runs: int = 5,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    Run parameter optimization experiments.
    Returns DataFrame with results for each parameter combination.
    """
    results = []
    total_combinations = len(mutation_rates) * len(elitism_rates) * len(crossover_types)
    
    with tqdm(total=total_combinations, desc="Parameter Optimization") as pbar:
        for mutation_rate in mutation_rates:
            for elitism_rate in elitism_rates:
                for crossover_type in crossover_types:
                    run_results = []
                    convergence_stats = []
                    
                    for run in range(num_runs):
                        run_seed = base_seed + run
                        params = GAParameters(
                            mutation_rate=mutation_rate,
                            elitism_rate=elitism_rate,
                            crossover_type=crossover_type,
                            random_seed=run_seed
                        )
                        
                        ga = TSPGeneticAlgorithm(distance_matrix, params)
                        best_individual, stats = ga.evolve(show_progress=False)
                        run_results.append(1 / best_individual.fitness)
                        convergence_stats.append(stats)
                    
                    avg_distance = sum(run_results) / len(run_results)
                    std_distance = np.std(run_results)
                    
                    # Calculate average convergence speed
                    convergence_gens = []
                    for stats in convergence_stats:
                        final_dist = stats[-1]['best_distance']
                        threshold = final_dist * 1.05
                        for gen_stat in stats:
                            if gen_stat['best_distance'] <= threshold:
                                convergence_gens.append(gen_stat['generation'])
                                break
                    
                    results.append({
                        'mutation_rate': mutation_rate,
                        'elitism_rate': elitism_rate,
                        'crossover_type': crossover_type.value,
                        'avg_best_distance': avg_distance,
                        'std_best_distance': std_distance,
                        'min_best_distance': min(run_results),
                        'max_best_distance': max(run_results),
                        'avg_convergence_gen': sum(convergence_gens) / len(convergence_gens)
                    })
                    
                    pbar.update(1)
    
    # Convert to DataFrame and sort by performance
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('avg_best_distance')
    
    return df_results

def plot_parameter_comparison(results_df: pd.DataFrame):
    """Create visualization of parameter optimization results."""
    # Plot 1: Boxplot of distances by crossover type
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    crossover_data = []
    crossover_labels = []
    for crossover in results_df['crossover_type'].unique():
        mask = results_df['crossover_type'] == crossover
        crossover_data.append([
            results_df.loc[mask, 'avg_best_distance'],
            results_df.loc[mask, 'min_best_distance'],
            results_df.loc[mask, 'max_best_distance']
        ])
        crossover_labels.append(crossover)
    plt.boxplot(crossover_data, labels=crossover_labels)
    plt.title('Performance by Crossover Type')
    plt.ylabel('Distance')
    
    # Plot 2: Scatter plot of mutation rate vs elitism rate
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        results_df['mutation_rate'],
        results_df['elitism_rate'],
        c=results_df['avg_best_distance'],
        cmap='viridis'
    )
    plt.colorbar(scatter, label='Average Distance')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Elitism Rate')
    plt.title('Parameter Space Exploration')
    
    plt.tight_layout()
    plt.show()