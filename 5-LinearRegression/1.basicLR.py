import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://bit.ly/3goOAnt", delimiter=",")

X = df.values[:, :-1]
Y = df.values[:, -1]

# line to the points
fit = LinearRegression().fit(X, Y)

# m = 1.78, b = -16.5
m = fit.coef_.flatten()
b = fit.intercept_.flatten()

# chart
plt.plot(X, Y, "o")  # scatter
plt.plot(X, m * X + b)  # line
plt.show()

""" CALCULATING RESIDUALS """
points = pd.read_csv("https://bit.ly/3goOAnt", delimiter=",").itertuples()

# test with given line
m = 1.9393
b = 4.7333
squared_error_sum = 0

for p in points:
    y_actual = p.y
    y_predict = m * p.x + b
    residual = y_actual - y_predict
    squared_error_sum += residual**2

print(squared_error_sum)
