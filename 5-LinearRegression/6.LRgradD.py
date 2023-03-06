""" 

Linear regression
- need to solve for m and b

Via GD
- take partiav derivative of each
.. trying to minimize loss -> take partial of loss function -> least squares

"""

import pandas as pd
import sympy

# getting data
points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())

m, b, i, n = sympy.symbols("m b i n")
x, y = sympy.symbols("x y", cls=sympy.Function)
sum_of_squares_f_of_x = sympy.Sum(((m * x(i)) + b - y(i)) ** 2, (i, 0, n))

# plotting loss function
sum_of_squares_plotting = (
    sympy.Sum(((m * x(i)) + b - y(i)) ** 2, (i, 0, n))
    .subs(n, len(points) - 1)
    .doit()
    .replace(x, lambda i: points[i].x)
    .replace(y, lambda i: points[i].y)
)
sympy.plotting.plot3d(sum_of_squares_plotting)

partial_along_m = (
    sympy.diff(sum_of_squares_f_of_x, m)
    .subs(n, len(points) - 1)
    .doit()
    .replace(x, lambda i: points[i].x)
    .replace(y, lambda i: points[i].y)
)
partial_along_b = (
    sympy.diff(sum_of_squares_f_of_x, b)
    .subs(n, len(points) - 1)
    .doit()
    .replace(x, lambda i: points[i].x)
    .replace(y, lambda i: points[i].y)
)

# compile using lambdify for faster computation
partial_along_m = sympy.lambdify([m, b], partial_along_m)
partial_along_b = sympy.lambdify([m, b], partial_along_b)

# build model
m = 0
b = 0
L = 0.001
iterations = 100_000
for i in range(iterations):
    m -= L * partial_along_m(m, b)
    b -= L * partial_along_b(m, b)

print("y = {}x + {}".format(m, b))
