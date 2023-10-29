from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import random


def func(x: float, y: float) -> float:
    return 1.5 - np.exp(-x**2 - y**2) - 0.5 * np.exp(-(x - 1)**2 - (y + 2)**2)


def min_max_norm(val, min_val, max_val, new_min, new_max):
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
        Is it assumed that this and the other Chromosome
        have genes of the same length

        other: the other Chromosome to crossover with

        Returns a child also Chromosome
        """
        seperation_index = random.randint(0, self.length - 1)
        child_genes_part1 = self.genes[0:seperation_index + 1]
        child_genes_part2 = other.genes[seperation_index + 1:self.length]
        child_genes = list(np.concatenate((child_genes_part1, child_genes_part2)))
        return Chromosome(self.length, child_genes)


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi, population_size=1000,
                 tournament_size=2, mutation_probability=0.05, crossover_probability=0.8, num_steps=30):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_lengths = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps

    def _get_arguments(self, chromosome: Chromosome) -> list[float]:
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
        We assume that we select Chromsomes to a group with no repetitions

        population: list of all Chromosomes in the population

        Returns tuple of two Chromosemes, potencial parents
        """
        num_of_groups = len(population) // 2
        future_parents = [None] * num_of_groups
        for i in range(num_of_groups):
            group1 = list(np.random.choice(population, 2, replace=False))
            group2 = list(np.random.choice(population, 2, replace=False))
            parent1 = self._get_best_chromosome(group1)
            parent2 = self._get_best_chromosome(group2)
            future_parents[i] = (parent1, parent2)
        return future_parents

    def _get_best_chromosome(self, chromosomes: list[Chromosome]) -> Chromosome:
        return max(chromosomes, key=lambda chrom: self.eval_objective_func(chrom))

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

    def plot_func(self, trace):
        X = np.arange(-2, 3, 0.05)
        Y = np.arange(-4, 2, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = 1.5 - np.exp(-X ** (2) - Y ** (2)) - 0.5 * np.exp(-(X - 1) ** (2) - (Y + 2) ** (2))
        plt.figure()
        plt.contour(X, Y, Z, 10)
        cmaps = [[ii / len(trace), 0, 0] for ii in range(len(trace))]
        plt.scatter([x[0] for x in trace], [x[1] for x in trace], c=cmaps)
        plt.show()

    def run(self):
        pass


if __name__ == "__main__":
    # Test metody decode
    expected_value = 0.529
    chrom = Chromosome(8, [1, 0, 0, 0, 0, 1, 1, 1])
    print(f"Decoded expected values is {expected_value}, "
          f"calculated value {chrom.decode(0, 7, (0, 1)):.3f}")

    # Test mutation
    chrom = Chromosome(8, [1, 1, 1, 1, 1, 1, 1, 1])
    # chrom.mutation(1)
    # print(chrom.genes)

    # Test crossover
    chrom1 = Chromosome(8, [0, 1, 1, 1, 0, 1, 0, 1])
    # print((chrom.crossover(chrom1)).genes)
    # print(chrom1.decode(0, 3, (-3, 3)))

    gen_alg = GeneticAlgorithm(8, 2, func, (-3, 3))
    print(gen_alg.eval_objective_func(chrom1))
