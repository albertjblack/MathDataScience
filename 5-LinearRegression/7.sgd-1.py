import pandas as pd
import numpy as np

data = pd.read_csv("https://bit.ly/2KF29Bd", header=0)
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

n = data.shape[0]  # rows
m = 0.0
b = 0.0
sample_size = 1  # greater than 1 perform mini batches
learning_rate = 0.0001
epochs = 1_000_000  # number of iterations to perform gradient descent

for i in range(epochs):
    idx = np.random.choice(
        n, sample_size, replace=False
    )  # n to generate idx from [0 to n-1]; sample size is the number of elements in each subarray

    x_sample = X[idx]
    y_sample = Y[idx]
    Y_pred = m * x_sample + b

    # derivative along m of loss function
    partial_of_m = (-2 / sample_size) * sum(x_sample * (y_sample - Y_pred))

    # derivative along b of loss function
    partial_of_b = (-2 / sample_size) * sum(y_sample - Y_pred)

    m -= learning_rate * partial_of_m
    b -= learning_rate * partial_of_b

    # print progress
    if i % 10000 == 0:
        print(i, m, b)

print("y = {0}x + {1}".format(m, b))
