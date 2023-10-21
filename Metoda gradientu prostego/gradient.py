from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np


def f_func(x: float, y: float) -> float:
    return x**2 + y**2


def g_func(x: float, y: float) -> float:
    return 1.5 - np.exp(-x**2 - y**2) - 0.5 * np.exp(-(x - 1)**2 - (y + 2)**2)


def gradient(func: Callable[[float, float], float],
             x: float, y: float, h: float = 1e-6) -> Tuple[float, float]:
    df_dx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    df_dy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return df_dx, df_dy


class SimpleGradientDescent:
    X = np.arange(-2, 2, 0.05)
    Y = np.arange(-3, 2, 0.05)
    X, Y = np.meshgrid(X, Y)

    def __init__(self,
                 func: Callable[[float, float], float],
                 grad_func: Callable[[Callable[[float, float], float], float, float, float],
                                     Tuple[float, float]],
                 alpha: float = 0.1):
        self.alpha = alpha
        self.func = func
        self.grad_func = grad_func
        self.trace = None

    def _calc_Z_value(self) -> None:
        self.Z = self.func(self.X, self.Y)

    def plot_func(self) -> None:
        self._calc_Z_value()
        plt.figure()
        plt.contour(self.X, self.Y, self.Z, 50)
        if self.trace is not None and len(self.trace) > 0:
            plt.scatter(self.trace[:, 0], self.trace[:, 1], s=5, c='#ff33cc')
        plt.show()

    def calculate_func_value(self, x1: float, x2: float) -> float:
        return self.func(x1, x2)

    def calculate_func_grad(self, x1: float, x2: float) -> Tuple[float, float]:
        return self.grad_func(self.func, x1, x2)

    def gradient_descent_step(self, x1: float, x2: float) -> Tuple[float, float]:
        gradient = self.calculate_func_grad(x1, x2)
        next_x1 = x1 - self.alpha * gradient[0]
        next_x2 = x2 - self.alpha * gradient[1]

        return next_x1, next_x2

    def minimize(self, x1_init: float, x2_init: float, steps: int, *, if_verbose: bool = False,
                 if_plot: bool = False) -> float:
        x1, x2 = x1_init, x2_init
        initial_steps = steps
        self.trace = np.array([[x1, x2]])
        while steps > 0:
            new_x1, new_x2 = self.gradient_descent_step(x1, x2)
            print(self.alpha)
            if self.calculate_func_value(new_x1, new_x2) >= self.calculate_func_value(x1, x2):
                self.alpha /= 2
                continue
            x1, x2 = new_x1, new_x2
            steps -= 1
            self.trace = np.vstack((self.trace, [x1, x2]))
        min_found_value = self.calculate_func_value(x1, x2)
        if if_verbose:
            print(f"Trace of search after {initial_steps} iterations (rounded to 4 decimal places): "
                  "{list([list([round(value, 4) for value in point]) for point in self.trace])}")
            print(f"Min value of the function found is {min_found_value}\n")
        if if_plot:
            self.plot_func()
        return min_found_value


if __name__ == "__main__":
    # to powinno dać wynik (2, 4)
    print(gradient(f_func, 1, 2))

    # to powinno dać wynik (0.02699718792, 0.01361209378)
    print(gradient(g_func, 2, 1))

    smpl_gradient_func = SimpleGradientDescent(f_func, gradient)
    smpl_gradient_func.minimize(-1.5, -2, 25, if_verbose=True, if_plot=True)

# TODO: widoczne zmniejszanie ogarnij co mówi długość gradientu
# TODO: test gradient_descent_step - krok gradientu prostego
# TODO: przetestuj dlugie kroki
