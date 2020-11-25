import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVC

#X_train = np.genfromtxt("X_train.csv")
#y_train = np.genfromtxt("y_train.csv")

n = 300

X_train = np.asarray(pd.read_csv("X_train.csv"))[:n]
y_train = np.asarray(pd.read_csv("y_train.csv"), dtype = np.int32)[:n].reshape([X_train.shape[0],])
X_test = np.asarray(pd.read_csv("X_test.csv"))
y_test = np.asarray(pd.read_csv("y_test.csv"), dtype = np.int32).reshape([X_test.shape[0],])

print("Unique train labels: " + str(np.unique(y_train)))

print("x shape: " + str(X_train.shape))
print("y shape: " + str(y_train.shape))
#print("X: " + str(X_train))
#print("y: " + str(y_train))

C = [.01, .1, 1., 10, 100]
for c in C:
#    model = SVC()
    model = SVC(C=c, kernel = 'rbf')
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    confusion_matrix_train = np.array([[sum((y_train == 0) & (pred_train == 0)), sum((y_train == 1) & (pred_train == 0))],
                                       [sum((y_train == 0) & (pred_train == 1)), sum((y_train == 1) & (pred_train == 1))]])
    confusion_matrix_test = np.array([[sum((y_test == 0) & (pred_test == 0)), sum((y_test == 1) & (pred_test == 0))],
                                      [sum((y_test == 0) & (pred_test == 1)), sum((y_test == 1) & (pred_test == 1))]])
    score_test = float(sum((y_test == 0) & (pred_test == 0)) + sum((y_test == 1) & (pred_test == 1))) / len(y_test)

    print()
    print("---------- c = " + str(c) + " ----------")
    print("Training confusion matrix:")
    print(confusion_matrix_train)
    print("Testing confusion matrix:")
    print(confusion_matrix_test)
    print("Testing score: " + str(score_test))
