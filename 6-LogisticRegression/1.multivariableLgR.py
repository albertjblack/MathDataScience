import pandas as pd
from sklearn.linear_model import LogisticRegression

employee_data = pd.read_csv("https://tinyurl.com/y6r7qjrp")
inputs = employee_data.iloc[:, :-1]
output = employee_data.iloc[:, -1]

fit = LogisticRegression(penalty="none").fit(inputs, output)
print("Coefficients: {0}".format(fit.coef_.flatten()))
print("Intercepts: {0}".format(fit.intercept_.flatten()))


# interact and test
def predict_employee_staying(sex: int, age: int, promotions: int, years_employed: int):
    prediction = fit.predict([[sex, age, promotions, years_employed]])
    probabilities = fit.predict_proba([[sex, age, promotions, years_employed]])
    if prediction == [[1]]:
        return "Will leave: {0}".format(probabilities)  # [[p(false), p(true)]]
    return "Will stay: {0}".format(probabilities)


while True:
    line = input(
        "Predict employee will stay or leave {sex}, {age}, {promotions}, {years_employed}: "
    )
    sex, age, promotions, years_employed = line.strip(" ").split(",")
    print(
        predict_employee_staying(
            int(sex), int(age), int(promotions), int(years_employed)
        )
    )
