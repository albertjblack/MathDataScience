import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit

# data
df = pd.read_csv("https://bit.ly/3cIH97A", delimiter=",")
X = df.values[:, :-1]
Y = df.values[:, -1]

# simple linear reg
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)

print(results)
print("MSE: mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))
print()

# random fold validation
kfold = ShuffleSplit(n_splits=10, test_size=1 / 3, random_state=7)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)

print(results)
print("MSE: mean=%.3f (stdev-%.3f)" % (results.mean(), results.std()))
