""" find best m and b to produce minimum squared error (SIMPLE L.R.) """
import pandas as pd

points = list(pd.read_csv("https://bit.ly/2KF29Bd", delimiter=",").itertuples())
n = len(points)

m = (
    n * sum((p.x * p.y for p in points))
    - sum((p.x for p in points)) * sum(p.y for p in points)
) / (n * sum((p.x**2 for p in points)) - sum(p.x for p in points) ** 2)

b = (sum(p.y for p in points) / n) - m * sum((p.x for p in points)) / n

# eqn: derived from calculus ;; closed form do not scale well do to computational complexity (P vs NP)
print(m, b)
