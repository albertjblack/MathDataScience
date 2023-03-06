""" 
Gradient Descent:
    - Minimize parameter set against an objective
    - Steps in direction where slope goes downward

Machine learning
    - Minimize our loss function
    - Partial derivative 
        .. flashlight allowing us to see the slope for every parameter (B0, B1)
        .. we step in directions for B1 and B0 where the slope goes downward -> step size depending on slope size
            .. calculate length of step by taking a fraction of the slope A.K.A. learning rate (fraction of slope)
            * Choosing a learning rate (fration of slope) to take step is like choosing between an ant, human or a giant - precise but more time

"""

# Walk before we run
import random


def f(x):
    return (x - 3) ** 2 + 4


def f_x(x):
    return 2 * (x - 3)


learning_rate = 0.001
iterations = 100_000
start_random_x_value = random.randint(-15, 15)

for i in range(iterations):
    derivative_evaluated_at_x = f_x(start_random_x_value)

    # updated x by subtracting the (learning_rate) * (slope)
    start_random_x_value -= learning_rate * derivative_evaluated_at_x

# find x after iterations and to get the value where the f_x is minimized -> evaluate f(x)
print(start_random_x_value, f(start_random_x_value))
