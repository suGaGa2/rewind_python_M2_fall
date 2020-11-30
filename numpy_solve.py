import numpy as np

A = np.array([[2, -1],\
             [ 0, 1]])

b = np.array([0, 2])


X = np.linalg.solve(A, b)

print(X)

