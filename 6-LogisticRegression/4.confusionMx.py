# Accuracy is misleading in imbalanced data
"""  
Confusion matrix ! CAN BE SUMMARIZED (if multiple) by ROC receiver operator characteristic
^ compare different models

- breaks predictions against outcomes
- true positives
- true negatives
- false positives (err 1)
- false negatives (err 2)

! Precision: Tp/(Tp+Fp) // how accurate positive preds are
- by Bayes -> flip sensitivity -> what percentage of those who tested positive have the disease
! Sensitivity: Tp/(Tp+Fn) // rate of identified positives
- of patients that have disease, how many will be identified successfully
! Specificity: Tn/(Tn+Fp) 
! Accuracy: (Tp+Tn)/(Tp+Tn+Fp+Fn)
! F1 score: (2*Precision*Recall)/(Precision+Recall)

"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold

df = pd.read_csv("https://bit.ly/3cManTi", delimiter=",")
X = df.values[:, :-1]
Y = df.values[:, -1]

model = LogisticRegression(solver="liblinear")
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.33, random_state=10)
model.fit(Xtr, Ytr)
prediction = model.predict(Xte)

# confusion matrix evaluates accuracy within each category
# ! [tp, fn]
# ! [fp, tn]
matrix = confusion_matrix(y_true=Yte, y_pred=prediction)
print(matrix)

""" 
- USE BAYES
... 1% of population actually have the disease
>> account for proportion that actually has disease into our Conf. Matrix

P(risk | pos) = (P(pos | risk) * P(risk)) / (P(pos))
e.g. = (.99*.01)/(.99) = 0.0339

         [test post, test neg]
At risk  [198,          2    ]
Not risk [50,           750  ]

"""

# class imbalance -> data not equally represented across categories
# 1. get more data
# 2. duplicate samples in minority class
# stratify -> pass column (Y) to balance classes data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, stratify=Y)

# 3. algorithms SMOTE -> synthetic samples of minority
