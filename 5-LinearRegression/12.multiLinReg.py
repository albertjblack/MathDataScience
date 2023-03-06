import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://bit.ly/2X1HWH7", delimiter=",")
X = df.values[:, :-1]
Y = df.values[:, -1]

fit = LinearRegression().fit(X, Y)

print(f"Coefficients = {fit.coef_}")
print(f"Intercept = {fit.intercept_}")
print(f"z = {fit.intercept_} + {fit.coef_[0]}x + {fit.coef_[1]}y")
