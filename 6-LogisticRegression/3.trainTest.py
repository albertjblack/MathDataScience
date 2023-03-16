import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("https://tinyurl.com/y6r7qjrp", delimiter=",")
X = df.values[:, :-1]
Y = df.values[:, -1]

# random state is the random seed which we fix to seven
kfold = KFold(n_splits=3, random_state=7, shuffle=True)
model = LogisticRegression(penalty="none")
results = cross_val_score(model, X, Y, cv=kfold)
print("Acc: mean=%.3f; stdev=%.3f " % (results.mean(), results.std()))

# AUC SCORING
results = cross_val_score(model, X, Y, cv=kfold, scoring="roc_auc")
print(results.mean(), results.std())
