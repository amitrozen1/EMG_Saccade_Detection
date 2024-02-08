import numpy as np
from scipy.stats import entropy
from scipy.stats import zscore

vector1 = np.sin(np.linspace(0, 10, 100)) +2
vector2 = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100) + 1.5

print(min(vector1), max(vector1))
print(min(vector2), max(vector2))

# pearson correlation
correlation = np.corrcoef(vector1, vector2)[0, 1]

# Compute entropy
e = entropy(vector1, vector2)

#correlate
vector1 = zscore(vector1)
vector2 = zscore(vector2)
cor = np.correlate(vector1, vector2, "valid")/len(vector1)

print(f"Vector 1: {vector1}")
print(f"Vector 2: {vector2}")
print(f"Pearson Correlation: {correlation}")
print(f"1-Entropy: {1-e}")
print(f"Correlation: {cor}")

#plot
import matplotlib.pyplot as plt
plt.plot(vector1)

plt.plot(vector2)

plt.show()


