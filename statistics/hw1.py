import numpy as np
import matplotlib.pyplot as plt

a_true, b_true = 1, 4
n_values = range(10, 1000, 10)
a_errors, b_errors = [], []

for n in n_values:
    sample = np.random.uniform(a_true, b_true, n)
    a_hat = np.mean(sample) - np.sqrt(3 * (np.var(sample)))
    b_hat = np.mean(sample) + np.sqrt(3 * (np.var(sample)))
    a_errors.append((a_hat - a_true)**2)
    b_errors.append((b_hat - b_true)**2)

plt.plot(n_values, a_errors, label='a_error')
plt.plot(n_values, b_errors, label='b_error')
plt.legend()
plt.savefig('plot.png')
plt.show()
