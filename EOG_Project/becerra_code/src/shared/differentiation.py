from numba import njit, stencil
from numpy import array


@njit(fastmath=True, parallel=True)
def central_difference_3(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (f[1] - f[-1]) / (2 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def central_difference_5(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (f[-2] - 8 * f[-1] + 8 * f[1] - f[2]) / (12 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def central_difference_7(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (-f[-3] + 9 * f[-2] - 45 * f[-1] + 45 * f[1] - 9 * f[2] + f[3]) / (60 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def central_difference_9(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (3 * f[-4] - 32 * f[-3] + 168 * f[-2] - 672 * f[-1] + 672 * f[1] - 168 * f[2] + 32 * f[3] - 3 * f[4]) / (840 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def lanczos_5(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (f[1] - f[-1] + 2 * (f[2] - f[-2])) / (10 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def lanczos_7(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (f[1] - f[-1] + 2 * (f[2] - f[-2]) + 3 * (f[3] - f[-3])) / (28 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def lanczos_9(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (f[1] - f[-1] + 2 * (f[2] - f[-2]) + 3 * (f[3] - f[-3]) + 4 * (f[4] - f[-4])) / (60 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def lanczos_11(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (f[1] - f[-1] + 2 * (f[2] - f[-2]) + 3 * (f[3] - f[-3]) + 4 * (f[4] - f[-4]) + 5 * (f[5] - f[-5])) / (110 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def lanczos_13(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (f[1] - f[-1] + 2 * (f[2] - f[-2]) + 3 * (f[3] - f[-3]) + 4 * (f[4] - f[-4]) + 5 * (f[5] - f[-5])) + 6 * (f[6] - f[-6]) / (182 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def super_lanczos_7(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (58 * (f[1] - f[-1]) + 67 * (f[2] - f[-2]) - 22 * (f[3] - f[-3])) / (252 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def super_lanczos_9(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (126 * (f[1] - f[-1]) + 193 * (f[2] - f[-2]) + 142 * (f[3] - f[-3]) - 86 * (f[4] - f[-4])) / (1188 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def super_lanczos_11(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (296 * (f[1] - f[-1]) + 503 * (f[2] - f[-2]) + 532 * (f[3] - f[-3]) + 294 * (f[4] - f[-4]) - 300 * (f[5] - f[-5])) / (5148 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def smooth_noise_robust_5(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (2 * (f[1] - f[-1]) + f[2] - f[-2]) / (8 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def smooth_noise_robust_7(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (5 * (f[1] - f[-1]) + 4 * (f[2] - f[-2]) + f[3] - f[-3]) / (32 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def smooth_noise_robust_9(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (14 * (f[1] - f[-1]) + 14 * (f[2] - f[-2]) + 6 * (f[3] - f[-3]) + f[4] - f[-4]) / (128 * h)
    )(data, step)


@njit(fastmath=True, parallel=True)
def smooth_noise_robust_11(data: array, step: float) -> array:
    return stencil(
        lambda f, h: (42 * (f[1] - f[-1]) + 48 * (f[2] - f[-2]) + 27 * (f[3] - f[-3]) + 8 * (f[4] - f[-4]) + f[5] - f[-5]) / (512 * h)
    )(data, step)


METHODS = {
    'cd3': central_difference_3,
    'cd5': central_difference_5,
    'cd7': central_difference_7,
    'cd9': central_difference_9,
    'l5': lanczos_5,
    'l7': lanczos_7,
    'l9': lanczos_9,
    'l11': lanczos_11,
    'l13': lanczos_13,
    'sl7': super_lanczos_7,
    'sl9': super_lanczos_9,
    'sl11': super_lanczos_11,
    'snr5': smooth_noise_robust_5,
    'snr7': smooth_noise_robust_7,
    'snr9': smooth_noise_robust_9,
    'snr11': smooth_noise_robust_11,
}

def differentiate(data: array, step: float, method: str='l11') -> array:
    return METHODS[method](data, step)
