from numpy import array, mean
from numba import njit


@njit(fastmath=True, parallel=True)
def mse(real: array, approximation: array) -> float:
    return mean((real - approximation) ** 2)
