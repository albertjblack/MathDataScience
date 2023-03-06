import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# initial data
df = pd.read_csv("https://bit.ly/3cIH97A", delimiter=",")
X = df.values[:, :-1]
Y = df.values[:, -1]

# separation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3)

# model
model = LinearRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)

""" r^2 is 1 - (sum(y-y_pred)**2)/(sum(y-y_avg)**2) """
print("r^2: %.3f" % result)
