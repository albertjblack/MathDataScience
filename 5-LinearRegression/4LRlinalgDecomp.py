# X = Q @ R
# b = inv(R) @ Qt @ y

import pandas as pd
import numpy as np

df = pd.read_csv("https://bit.ly/3goOAnt", delimiter=",")

X = df.values[:, :-1].flatten()
Y = df.values[:, -1].flatten()

X_1 = np.vstack([X, np.ones(len(X))]).transpose()
Q, R = np.linalg.qr(X_1)
b = np.linalg.inv(R) @ Q.transpose() @ Y

print(Q, R, b)
