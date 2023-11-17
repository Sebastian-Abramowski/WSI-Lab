import time
import gc
from typing import Callable, Any


def get_time(func_name: Callable[[Any], Any], *args) -> float:
    """
    Calculates the execution time of the funciton
    and returns time in seconds
    """
    gc_old = gc.isenabled()
    gc.disable()
    time_start = time.process_time()

    func_name(*args)

    time_stop = time.process_time()
    time_count = time_stop - time_start
    if gc_old:
        gc.enable()
    return time_count
