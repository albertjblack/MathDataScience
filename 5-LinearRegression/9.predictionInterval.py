import pandas as pd
from scipy.stats import t as T
from math import sqrt

points = list(pd.read_csv("https://bit.ly/2KF29Bd", delimiter=",").itertuples())
n = len(points)

m = 1.939
b = 4.733
x_0 = 8.5
x_mean = sum(p.x for p in points) / n

t_value = T(n - 2).ppf(0.975)  # ppf(area) -> value
standard_error = sqrt(sum((p.y - (m * p.x + b)) ** 2 for p in points) / (n - 2))
margin_of_error = (
    t_value
    * standard_error
    * sqrt(
        1
        + (1 / n)
        + (n * (x_0 - x_mean) ** 2)
        / (n * sum(p.x**2 for p in points) - sum(p.x for p in points) ** 2)
    )
)

y_pred = m * x_0 + b
print(y_pred - margin_of_error, y_pred + margin_of_error)
