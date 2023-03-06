import pandas as pd
from math import sqrt, fabs
import scipy

points = list(pd.read_csv("https://bit.ly/2KF29Bd").itertuples())
n = len(points)

numerator = n * sum(p.x * p.y for p in points) - sum(p.x for p in points) * sum(
    p.y for p in points
)

denominator = sqrt(
    n * sum(p.x**2 for p in points) - sum(p.x for p in points) ** 2
) * sqrt(n * sum(p.y**2 for p in points) - sum(p.y for p in points) ** 2)

corr = numerator / denominator
print(corr)

""" NOW THAT WE HAVE THE CORRELATION COEFFICIENT, WE CAN CALCULATE THE CONFIDENCE INTERVAL """

n = 10
lower_cv = scipy.stats.t(n - 1).ppf(0.025)
upper_cv = scipy.stats.t(n - 1).ppf(0.975)

# get range to know if we can reject null hypothesis
print(lower_cv, upper_cv)
r = corr

# to prove correlation -> r need to be outside the range of the confidence interval
# calculate the test value -> t = confidence/sqrt((1-conf**2)/n-2)
test_value = r / sqrt((1 - r**2) / (n - 2))

print(f"Test value: {test_value} | Confidence interval: {lower_cv} - {upper_cv}")
if test_value < lower_cv or test_value > upper_cv:
    print("Correlation Proven. Reject null hypothesis")
else:
    print("Coerrelation not Proven. Accept null hypothesis")

# calculate p-value -> whether there is evidence of a difference between the two groups <= 0.05 -> reject null hypothesis

if test_value > 0:
    p_value = 1.0 - scipy.stats.t(n - 1).cdf(test_value)
else:
    p_value = scipy.stats.t(n - 1).cdf(test_value)

# p_value = 1.0 - scipy.stats.t(n - 1).cdf(fabs(test_value))
p_value *= 2.0
print(f"p_value = {p_value} | p_value <= 0.05: {p_value <= 0.05}")

""" 

Test val = 9.39956
- outside confidence interval

Now we check for coincidence via the p-value treshold of 0.05
- we get 0.0000005976

Thus, there is virtually not coincidence between the two variables
It is highly unlikely that the variables are related by chance

Thus, there is a correlation between the two variables


"""
