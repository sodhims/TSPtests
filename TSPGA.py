import numpy as np
from typing import List, Tuple, Dict
import random
from enum import Enum
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

    # [Previous methods remain the same until evolve()]

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
                    
                    # Calculate average convergence speed (generations to reach within 5% of final distance)
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

# Example usage:
if __name__ == "__main__":
    # Create sample distance matrix
    num_cities = 20
    np.random.seed(42)
    distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) // 2  # Make symmetric
    
    # Run single GA instance with fixed seed
    print("\nRunning single GA instance...")
    params = GAParameters(random_seed=42)
    ga = TSPGeneticAlgorithm(distance_matrix, params)
    best_solution, stats = ga.evolve()
    print(f"\nBest distance: {1/best_solution.fitness:.2f}")
    print(f"Best route: {best_solution.route}")
    
    # Plot convergence for single run
    plot_convergence(stats, "Single Run Convergence")
    
    # Run parameter optimization
    print("\nRunning parameter optimization...")
    optimization_results = run_parameter_optimization(
        distance_matrix,
        mutation_rates=[0.01, 0.02, 0.05],
        elitism_rates=[0.05, 0.1, 0.2],
        crossover_types=[CrossoverType.EDGE, CrossoverType.ORDER, CrossoverType.PMX],
        num_runs=3  # Reduced for demonstration
    )
    
    # Display results
    print("\nParameter Optimization Results:")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(optimization_results)
    
    # Plot parameter comparison
    plot_parameter_comparison(optimization_results)