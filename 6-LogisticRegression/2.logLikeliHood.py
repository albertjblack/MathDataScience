from math import log, exp
import pandas as pd
import scipy

patient_data = list(pd.read_csv("https://bit.ly/33ebs2R", delimiter=",").itertuples())

likelihood = sum(p.y for p in patient_data) / len(patient_data)
b0 = -3.17576395
b1 = 0.69267212

log_likelihood = 0.0
for p in patient_data:
    if p.y == 1.0:
        log_likelihood += log(likelihood)
    elif p.y == 0.0:
        log_likelihood += log(1.0 - likelihood)


def logistic_function(x):
    p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))
    return p


log_likelihood_fit = sum(
    log(logistic_function(p.x)) * p.y + log(1.0 - logistic_function(p.x)) * (1.0 - p.y)
    for p in patient_data
)

print(log_likelihood, log_likelihood_fit)
r2 = (
    log_likelihood - log_likelihood_fit
) / log_likelihood  # show power of causation of exposure hours (lo = 0) (hi = 1)
print(r2)

# p-value
chi2_input = 2 * (log_likelihood_fit - log_likelihood)
p_value = scipy.stats.chi2.pdf(chi2_input, 1)  # 1 deg f ->  "Nparams - 1"
print(p_value)

""" 
BASICALLY R2 AND P VALUES HELP TO VALIDATE INSTEAD OF TRAIN TEST SPLITS
"""
