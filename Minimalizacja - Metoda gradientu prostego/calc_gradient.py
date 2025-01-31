from typing import Callable, Tuple
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
