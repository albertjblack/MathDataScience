""" 
Eigendecomposition
- break matrix into easier to work components
lambda; eigenvector; square matrices

Av = lambdaV
- original matrix A is composed of eigenvector and eigenvalue

"""

import numpy as np

A = np.array([[1, 2], [4, 5]])
eigenvals, eigenvecs = np.linalg.eig(A)

# Rebuild A from eigenvector and eigenvalues
# -> A*v = lambda*v

""" WE NEED TO TWEAK FORMULA TO RECONSTRUCT A """
# A = EIGENVE * EIGENVA * EIGENVE-1 = Q^R
Q = eigenvecs  # EIGENVECS
R = np.linalg.inv(Q)  # INV
L = np.diag(eigenvals)  # EIGENVAL = ^

B = Q @ L @ R
print(B)
