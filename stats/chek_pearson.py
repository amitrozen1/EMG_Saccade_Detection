import numpy as np
from scipy.stats import pearsonr


def get_random_number_between_minus_one_and_one():
    return np.random.uniform(-10, 10)

def create_random_vector_size_n(n):
    return np.array([get_random_number_between_minus_one_and_one() for i in range(n)])
x = np.linspace(0, 100, 50)
y = x + np.sin(create_random_vector_size_n(50))
# random number between -1 and 1


x2 = np.linspace(0, 100, 100)
y2 = x2 + np.sin(create_random_vector_size_n(100))

r, p = pearsonr(x, y)
print(r, p)

r, p = pearsonr(x2, y2)
print(r, p)