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

    def decode(self, lower_bound: int, upper_bound: int, aoi: tuple[int, int]):
        """
        lower_bound: the starting index of the active genes (bits)
        upper_bound: the ending index of the active genes (bits)
        aoi: range of decoded value
        """
        active_genes = self.genes[lower_bound:upper_bound + 1]
        genes_value = int("".join([str(bit) for bit in active_genes]), 2)
        max_genes_value = int("1" * len(active_genes), 2)
        return min_max_norm(genes_value, 0, max_genes_value, aoi[0], aoi[1])

    def mutation(self, probability):
        pass

    def crossover(self, other):
        pass


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi, population_size=1000,
                 tournament_size=2, mutation_probability=0.05, crossover_probability=0.8, num_steps=30):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_lengths = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps

    def eval_objective_func(self, chromosome):
        pass

    def tournament_selection(self):
        pass

    def reproduce(self, parents):
        pass

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
