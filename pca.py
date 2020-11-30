import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pca():
    n = 600

    X_train = np.asarray(pd.read_csv("X_train.csv"))[:n]
    Y_train = np.asarray(pd.read_csv("y_train.csv"))[:n].reshape([X_train.shape[0],])
    X_test = np.asarray(pd.read_csv("X_test.csv"))[:n]
    Y_test = np.asarray(pd.read_csv("y_test.csv"))[:n].reshape([X_test.shape[0],])

    # Center data
    mean_train = np.mean(X_train,0)
    mean_test = np.mean(X_test,0)

    for i in range(len(X_train)):
        X_train[i] -= mean_train
    for i in range(len(X_test)):
        X_test[i] -= mean_test

    # find VT to rotate X
    [U,s,VT] = np.linalg.svd(X_train, full_matrices=False)

    # transform data with rotation
    X_train = X_train.dot(VT.T)

    ## do same with x_test
    [U,s,VT] = np.linalg.svd(X_test, full_matrices=False)
    X_test = X_test.dot(VT.T)

    ## return top two principal components
    return [X_train[:,:2], Y_train, X_test[:,:2], Y_test]

if __name__ == '__main__':
    pca()
