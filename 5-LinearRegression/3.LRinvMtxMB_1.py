# we can use transposed and inverse matrix to fit a linear regression .. matrix of coefficients b for m and b
""" b = inv(Xt * X) * Xt * y """

import pandas as pd
import numpy as np

df = pd.read_csv("https://bit.ly/3goOAnt", delimiter=",")

X = df.values[:, :-1].flatten()
Y = df.values[:, -1]

# add placeholder "1" column to generate intercept for each x value (transpose to go from 2*n to n*2)
X_1 = np.vstack([X, np.ones(len(X))]).transpose()

# coefficients for slope and intercept
b = np.linalg.inv(X_1.transpose() @ X_1) @ X_1.transpose() @ Y
print(b)

# predict against y value
y_pred = X_1 @ b
print(
    Y, y_pred
)  # since b has the coeff. for b1 and b0, when we @ with X_1 we predict y due to b1 and b0
