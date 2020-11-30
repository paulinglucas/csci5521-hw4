import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pca():
    n = 600

    X = np.asarray(pd.read_csv("X_train.csv"))[:n]
    Y = np.asarray(pd.read_csv("y_train.csv"))[:n].reshape([X.shape[0],])

    # Center data
    mean = np.mean(X,0)
    for i in range(len(X)):
        X[i] -= mean

    # find VT to rotate X
    [U,s,VT] = np.linalg.svd(X, full_matrices=False)

    # transform data with rotation
    X = X.dot(VT.T)

    print(X[:,:2].shape)

if __name__ == '__main__':
    pca()
