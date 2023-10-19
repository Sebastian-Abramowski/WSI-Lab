import numpy as np

import plot
from knapsack_problem import KnapSack
from random import randint
from process_times import get_time


def calc_times_brute_force(max_num_of_elements: int) -> dict[int, float]:
    """
    Calculates correlation between the time of execution
    and the number of elements in arrays with weights and
    profits in algorithm solving knapsack problem with
    brute force

    Returns dictionary:
    {number_of_elements: execution_time_in_sec...}
    """
    result = {}
    weights = np.array([], dtype=int)
    profits = np.array([], dtype=int)
    for i in range(1, max_num_of_elements + 1):
        weights = np.append(weights, [randint(0, 20)])
        profits = np.append(profits, [randint(0, 20)])
        knapsack = KnapSack(profits, weights, randint(10, 50))
        time_in_sec = get_time(knapsack.solve_knapsack_brute_force)
        result[i] = time_in_sec
    return result


if __name__ == "__main__":
    result = calc_times_brute_force(25)
    num_of_items = [key for key, _ in result.items()]
    times = [value for _, value in result.items()]

    plot.configurate_plot(x_title="Number of items in knapsack", y_title="Execution time [s]",
                          title="Solving knapsack problem with brute-force")
    plot.make_plot(x_values=num_of_items, y_values=times,
                   hex_color=plot.PLOT_GREEN, line_width=plot.LINE_WIDTH_MEDIUM)
    plot.save_plot("img/knapsack_brute_force2.png")
    plot.clear_after_making_plot()
