"""  
Feed forward neural network
- Input: color .. seeking what it outputs
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

# extract RGB colors [0,1,2] -> scale down by 255x -> [0, 1] range
all_inputs = all_data.iloc[:, 0:3].values / 255.0
all_outputs = all_data.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    all_inputs, all_outputs, test_size=1 / 3
)

n_nodes_h1 = 3
n_nodes_output = 1

# ... (r,  c)
# xtr (??, 3)
# w_h (3,**)
# res (??,**)

# M.shape(n_nodes_In_me, n_vals_Into_me)
w_hidden = np.random.rand(X_train.shape[1], n_nodes_h1)  # R1 -> w1, w2, w3 -> next Node
w_output = np.random.rand(n_nodes_output, w_hidden.shape[0])  # (1,3) # node is row in M

# M.shape(n_nodes_I_work, n_val_to_give)
b_hidden = np.random.rand(w_hidden.shape[0], w_output.shape[0])  # (3,1) # 1 bias per N
b_output = np.random.rand(n_nodes_output, 1)  # (1,1)

# Activation
relu = lambda x: np.maximum(0, x)
logistic = lambda x: 1 / (1 + np.exp(-x))


# Forward Prop
def forward_prop(X):  # 1 in, 1 hid, 1 out
    # layers become next X_train (kinda)
    Z1 = w_hidden @ X + b_hidden  # w@x so we end up with (3,3) not (800,3)
    # np sum same row or col -> (3,3) + (3,1) = (3,1)
    A1 = relu(Z1)  # (3,1)
    Z2 = w_output @ A1 + b_output  # (unactive output - L2)
    # (1,3) @ (3,1) = (1,1)
    A2 = logistic(Z2)  # active output - L2 (1,1)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


# calculate accuracy
test_predictions = forward_prop(X_test.transpose())["A2"]  # (1, 449)
test_comparisons = np.equal((test_predictions >= 0.5).flatten().astype(int), Y_test)
accuracy = sum(test_comparisons.astype(int)) / Y_test.shape[0]
print(f"Accuracy: {accuracy}")

from sklearn.metrics import confusion_matrix

Y_pred = (np.array(test_predictions) >= 0.5).flatten()
cf = confusion_matrix(Y_test, Y_pred)
print(cf)

cf = list(cf)
cf = [int(x[i].reshape(1, 1)) for x in cf for i in range(len(cf))]
tp, fp, fn, tn = cf
sensitivity = precision = accuracy = specificity = 0
try:
    sensitivity = tp / (tp + fn)
except:
    pass

try:
    precision = tp / (tp + fp)
except:
    pass

try:
    accuracy = (tp + fn) / (tp + fp + fn + tn)
except:
    pass

try:
    specificity = tn / (tn + fp)
except:
    pass
print(
    f"Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, Accuracy: {accuracy}"
)
