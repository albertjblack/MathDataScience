# %%
import pandas as pd
import numpy as np

df = pd.read_csv("https://bit.ly/3C8JzrM")


""" 
1. find the best m and b that minimize the sum of squares
2. calculate the correlation coefficient and statistical significance of this data at 95%. is it useful?
3. If predict x=50, what is the 95% prediction interval
4. Start your regression over train/test, can use cross-validation or random-fold
"""
# %% 0. load data

X = df.values[:, :-1]
Y = df.values[:, -1]
n_rows = df.shape[0]


# %% 1 min m and b
m = 0.0
b = 0.0
sample_size = 1
learning_rate = 0.0001
epochs = 1_000_000

from sklearn.linear_model import LinearRegression

fit = LinearRegression().fit(X, Y)
m = fit.coef_.flatten()
b = fit.intercept_.flatten()

print(m, b)  # 1.58, 4.98

# %% 2 - correlation coefficient and statistical significance
from scipy.stats import t as T

correlation_coefficient = df.corr(method="pearson").values[0, -1]
print(f"Correlation coefficient: {correlation_coefficient}")

confidence_value = 0.95
lower_cv = T(n_rows - 1).ppf((1 - confidence_value) / 2)  # 0.025
upper_cv = T(n_rows - 1).ppf(confidence_value + (1 - confidence_value) / 2)  # 0.975
print(f"Confidence range [{lower_cv}, {upper_cv}]")

from math import sqrt

test_value = correlation_coefficient / sqrt(
    (1 - correlation_coefficient**2) / (n_rows - 2)
)
print(f"Test value: {test_value}")

if test_value < 0:
    p_value = T(n_rows - 1).cdf(test_value)
else:
    p_value = 1 - T(n_rows - 1).cdf(test_value)

p_value *= 2
print("P-value: ", p_value)
print(
    f"Significance: {((test_value<lower_cv or test_value>upper_cv) and p_value < 0.05)}"
)

# %% 3 x = 50; 95% confidence interval
points = list(df.itertuples())
x_0 = 50
t_value = T(n_rows - 2).ppf(0.975)  # ppf(area) -> value
x_mean = sum(p.x for p in points) / n_rows

standard_error = sqrt(sum((p.y - (m * p.x + b)) ** 2 for p in points) / (n_rows - 2))

margin_of_error = (
    t_value
    * standard_error
    * sqrt(
        1
        + (1 / n_rows)
        + (n_rows * (x_0 - x_mean) ** 2)
        / (n_rows * sum(p.x**2 for p in points) - sum(p.x for p in points) ** 2)
    )
)

y_pred = m * x_0 + b
print(y_pred - margin_of_error, y_pred + margin_of_error)

# %% 4
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print(results)
print("Mse: (mean -> %.3f) (stdev -> %.3f)" % (results.mean(), results.std()))

# %%
