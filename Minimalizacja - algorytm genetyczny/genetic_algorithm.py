from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, Union
import random


def func(x: float, y: float) -> float:
    return 1.5 - np.exp(-x**2 - y**2) - 0.5 * np.exp(-(x - 1)**2 - (y + 2)**2)


def min_max_norm(val: Union[int, float], min_val: Union[int, float],
                 max_val: Union[int, float], new_min: Union[int, float],
                 new_max: Union[int, float]) -> float:
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
    def __init__(self, length: int, array: Optional[list[int]] = None):
        self.length = length
        if array:
            self.genes = np.array(array)
        else:
            self.genes = np.array([random.randint(0, 1) for _ in range(length)])

    def decode(self, lower_bound: int, upper_bound: int, aoi: tuple[int, int]) -> float:
        """
        lower_bound: the starting index of the active genes (bits)
        upper_bound: the ending index of the active genes (bits)
        aoi: (Area of Interest) range of decoded value

        Returns value of the active part of Chromosome
        """
        active_genes = self.genes[lower_bound:upper_bound + 1]
        genes_value = int("".join([str(bit) for bit in active_genes]), 2)
        max_genes_value = int("1" * len(active_genes), 2)
        return min_max_norm(genes_value, 0, max_genes_value, aoi[0], aoi[1])

    def mutation(self, probability: float) -> bool:
        """
        probability: probability of a mutation occuring during the process

        Returns True if the mutation occured and False if not
        """
        random_index = random.randint(0, self.length - 1)
        mutation_chance = random.uniform(0.0, 1.0)
        if mutation_chance <= probability:
            self.genes[random_index] = 1 - self.genes[random_index]
            return True
        return False

    def crossover(self, other: Chromosome) -> Chromosome:
        """
        other: the other Chromosome to crossover with

        Returns a child, also Chromosome
        """
        assert self.length % other.length == 0, "Lengths of chromosomes should be the same"
        seperation_index = random.randint(0, self.length - 1)
        child_genes_part1 = self.genes[0:seperation_index + 1]
        child_genes_part2 = other.genes[seperation_index + 1:self.length]
        child_genes = list(np.concatenate((child_genes_part1, child_genes_part2)))
        return Chromosome(self.length, child_genes)


class GeneticAlgorithm:
    def __init__(self, chromosome_length: int, obj_func_num_args: int,
                 objective_function: Callable[..., float], aoi: tuple[int, int],
                 population_size: int = 1000, tournament_size: int = 2,
                 mutation_probability: float = 0.05,
                 crossover_probability: float = 0.8, num_steps: int = 30):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_lengths = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = chromosome_length // obj_func_num_args
        self.objective_function = objective_function
        self.aoi = aoi
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps

    def _get_arguments(self, chromosome: Chromosome) -> list[float]:
        """
        Takes values from genes of some chromosome depending on the number
        of arguments for the objective function (self.obj_func_num_args)

        Returns list of arguments for the objective function
        """
        args = [None] * self.obj_func_num_args
        current_index = 0
        for i in range(self.obj_func_num_args):
            args[i] = chromosome.decode(current_index, current_index + self.bits_per_arg - 1, self.aoi)
            current_index += self.bits_per_arg
        return args

    def eval_objective_func(self, chromosome: Chromosome) -> float:
        return self.objective_function(*self._get_arguments(chromosome))

    def tournament_selection(self, population: list[Chromosome]) -> list[tuple[Chromosome,
                                                                               Chromosome]]:
        """
        We assume that we select Chromsomes to a group with no repetitions, one tournament
        picks two parents

        population: list of all Chromosomes in the population

        Returns tuple of two Chromosemes, potencial parents
        """
        num_of_tournaments = len(population) // self.tournament_size
        future_parents = [None] * num_of_tournaments
        for i in range(num_of_tournaments):
            group1 = list(np.random.choice(population, self.tournament_size, replace=False))
            group2 = list(np.random.choice(population, self.tournament_size, replace=False))
            parent1 = self._get_best_chromosome(group1)
            parent2 = self._get_best_chromosome(group2)
            future_parents[i] = (parent1, parent2)
        return future_parents

    def _get_best_chromosome(self, chromosomes: list[Chromosome]) -> Chromosome:
        return min(chromosomes, key=lambda chrom: self.eval_objective_func(chrom))

    def reproduce(self, parents: list[tuple[Chromosome, Chromosome]]) -> list[Chromosome]:
        offspring = []
        for single_parents in parents:
            crossover_chance = random.uniform(0.0, 1.0)
            if crossover_chance <= self.crossover_probability:
                parent1 = single_parents[0]
                parent2 = single_parents[1]
                child = parent1.crossover(parent2)
                child.mutation(self.mutation_probability)
                offspring.append(child)
            else:
                offspring.extend(single_parents)
        return offspring

    def plot_func(self, trace: list[tuple[float, float]]) -> None:
        X = np.arange(-2, 3, 0.05)
        Y = np.arange(-4, 2, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = 1.5 - np.exp(-X ** (2) - Y ** (2)) - 0.5 * np.exp(-(X - 1) ** (2) - (Y + 2) ** (2))
        plt.figure()
        plt.contour(X, Y, Z, 10)
        cmaps = [[ii / len(trace), 0, 0] for ii in range(len(trace))]
        plt.scatter([x[0] for x in trace], [x[1] for x in trace], c=cmaps)
        plt.show()

    def run(self, *, if_generational_replacement: bool = False):
        trace = []
        population = self.initialize_population()
        best_chromosome, min_value = self.find_best_chromosome(population)
        trace.append(self._get_arguments(best_chromosome))
        print(f"0. iteration | MIN VALUE {min_value} at {self._get_arguments(best_chromosome)} |"
              f"Population size: {len(population)} |")
        for i in range(self.num_steps):
            if if_generational_replacement:
                population = self.make_new_population_alternative(population)
            else:
                population = self.make_new_population_custom(population)

            new_best_chromosome, new_best_min_value = self.find_best_chromosome(population)
            if new_best_min_value < min_value:
                min_value = new_best_min_value
                best_chromosome = new_best_chromosome
                trace.append(self._get_arguments(best_chromosome))
                print(f"{i + 1}. iteration | MIN VALUE {min_value} at "
                      f"{self._get_arguments(best_chromosome)} | Population size: {len(population)} |")
        self.plot_func(trace)

    def initialize_population(self) -> list[Chromosome]:
        population = [None] * self.population_size
        for i in range(self.population_size):
            population[i] = Chromosome(self.chromosome_lengths)
        return population

    def find_best_chromosome(self, population: list[Chromosome]) -> tuple[Chromosome, float]:
        best_chromosome = self._get_best_chromosome(population)
        return best_chromosome, self.eval_objective_func(best_chromosome)

    def make_new_population_custom(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Replaces the worst individuals with offspring, it replaces
        some part of the current population
        """
        future_parents = self.tournament_selection(population)
        offspring = self.reproduce(future_parents)
        population = sorted(population, key=lambda chrom: self.eval_objective_func(chrom))
        population = population[0:len(population) - len(offspring)]
        population.extend(offspring)

        return population

    def make_new_population_alternative(self, population: list[Chromosome]) -> list[Chromosome]:
        """
        Alternative way to make a new population - generational replacement, it replaces
        the entire population with the new generation
        """
        new_population = []
        while len(new_population) != self.population_size:
            # Single tournament selection
            group1 = list(np.random.choice(population, self.tournament_size, replace=False))
            group2 = list(np.random.choice(population, self.tournament_size, replace=False))
            parent1 = self._get_best_chromosome(group1)
            parent2 = self._get_best_chromosome(group2)

            # Crossover, mutation
            crossover_chance = random.uniform(0.0, 1.0)
            if crossover_chance <= self.crossover_probability:
                child = parent1.crossover(parent2)
                child.mutation(self.mutation_probability)
                new_population.append(child)
            else:
                mutation_change = self.mutation_probability
                if mutation_change <= self.mutation_probability:
                    parent1.mutation(self.mutation_probability)
                new_population.append(parent1)
                if len(new_population) != self.population_size:
                    if mutation_change <= self.mutation_probability:
                        parent2.mutation(self.mutation_probability)
                    new_population.append(parent2)
        return new_population


if __name__ == "__main__":
    # Test metody decode
    expected_value = 0.529
    chrom0 = Chromosome(8, [1, 0, 0, 0, 0, 1, 1, 1])
    print(f"Decoded expected values is {expected_value}, "
          f"calculated value {chrom0.decode(0, 7, (0, 1)):.3f}")

    # Test mutation
    chrom1 = Chromosome(8, [1, 1, 1, 1, 1, 1, 1, 1])
    chrom1.mutation(1)
    print(f"Mutated from [1 1 1 1 1 1 1 1] to {chrom1.genes}")

    # Test crossover
    chrom1 = Chromosome(8, [1, 1, 1, 1, 1, 1, 1, 1])
    chrom2 = Chromosome(8, [0, 0, 0, 0, 0, 0, 0, 0])
    print(f"Product of the crossover: {(chrom1.crossover(chrom2)).genes}")

    # Test decode
    chrom3 = Chromosome(9, [1, 0, 1, 0, 0, 0, 1, 1, 0])
    print(f"Decoded value of [1 0 1] to value from range (-1, 1) should be {5/7*2-1}, "
          f"it is {chrom3.decode(0, 2, (-1, 1))}")
    print("Decoded value of [0 0 0] to value from range (-1, 1) should be -1, "
          f"it is {chrom3.decode(3, 5, (-1, 1))}")
    print(f"Decoded value of [1 1 0] to value from range (-1, 1) should be {6/7*2-1}, "
          f"it is {chrom3.decode(6, 8, (-1, 1))}")
    print('\n')

    # Good parameters
    gen_alg = GeneticAlgorithm(
        chromosome_length=64,
        obj_func_num_args=2,
        objective_function=func,
        aoi=(-5, 5),
        population_size=256,
        tournament_size=2,
        mutation_probability=0.1,
        crossover_probability=0.75,
        num_steps=40
    )
    gen_alg.run(if_generational_replacement=True)
