import random
import plotly.express as px

sample_size: int = 31
sample_count: int = 1000

x_values: list(int) = [
    ((sum(random.uniform(0.0, 1.0)) for i in range(sample_size)) / sample_size)
    for _ in range(sample_count)
]  # 1000 averages of 31 randomly generated 0.0 to 1.0 values

y_values: list(int) = [1 for _ in range(sample_count)]

# HOW COME EQUALLY LIKELY NUMBERS FORM A NORMAL DISTRIBUTION INSTEAD OF A FLAT LINE (UNIFROM DISTRIBUTION)

px.histogram(x=x_values, y=y_values)

# INDIVIDUAL NUMBERS IN THE SAMPLE ALONE WILL NOT CREATE A NORMAL DISTRIBUTION, WHEN WE GROUP THEM AND AVERAGE THEM IS ANOTHER SCENARIO

""" central limit theorem """

"""

- interesting things happen when we take large samples of a population, calculate the mean and plot them as a distribution

1. x-bar = mu
2. population is normal, thus ssmample means will be normal
3. population is not normal, sample size big enough (>=31), then normal distr
4. stdev of sample mean equal population stdec divided by sqrt(n)

! These behaviors allows us to infer things about populations based on samples
- <31  samples -> rely on T-distributions
# larger samples is better when uncertain

"""
