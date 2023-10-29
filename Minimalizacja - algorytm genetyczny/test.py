import numpy as np

x = np.array([5, 6])
y = np.array([1, 2])
print(np.concatenate((x, y)))

z = np.array([1, 2, 3])
print(np.random.choice(z, 2, replace=False))

print(sorted([1, 2, 3]))
q = [3, 4, 5]
parent = (1, 2)
q.extend(parent)
print(q)
