import numpy as np
from itertools import product

weights = np.array([8, 3, 5, 2])
capacity = 9
profits = np.array([16, 8, 9, 6])


class KnapSack:
    def __init__(self, profits, weights, capacity):
        self.profits = profits
        self.weights = weights
        self.capacity = capacity

    def solve_knapsack_brute_force(self):
        num_of_items = len(self.weights)
        # 1 - item selected, 0 - item not selected
        selected_items_possibilites = list(product(range(2), repeat=num_of_items))

        best_selected_items = None
        best_profit = 0
        best_weight = 0
        for selected_items in selected_items_possibilites:
            selected_items_weight = 0
            selected_items_profit = 0
            for i in range(len(self.weights)):
                if selected_items[i] == 1:
                    selected_items_weight += self.weights[i]
                    selected_items_profit += self.profits[i]
                if selected_items_weight > self.capacity:
                    break

            if selected_items_weight <= self.capacity:
                if selected_items_profit > best_profit:
                    best_profit = selected_items_profit
                    best_weight = selected_items_weight
                    best_selected_items = selected_items

        indexes_of_best_selected_items = [
            i for i, item in enumerate(best_selected_items) if item == 1]

        return {'indexes': indexes_of_best_selected_items,
                'weight': best_weight,
                'profit': best_profit}

    def solve_knapsack_pw_ratio(self):
        indexes_with_profit_weight_ratios = sorted(
            [(i, self.profits[i] / self.weights[i]) for i in range(
                len(self.weights))], key=lambda x: x[1])[::-1]

        weight = 0
        profit = 0
        selected_items_indexes = []

        for index, _ in indexes_with_profit_weight_ratios:
            if weight + self.weights[index] < self.capacity:
                weight += self.weights[index]
                profit += self.profits[index]
                selected_items_indexes.append(index)

        return {'indexes': sorted(selected_items_indexes),
                'weight': weight,
                'profit': profit}


kp = KnapSack(profits, weights, 9)
print(kp.solve_knapsack_brute_force())
print(kp.solve_knapsack_pw_ratio())
