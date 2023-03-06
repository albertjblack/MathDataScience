import numpy as np

A = np.array([[4, 2, 4], [5, 3, 7], [9, 3, 6]])
B = np.array([[44, 56, 72]]).transpose()
X = np.linalg.inv(A) @ B

print(X)
