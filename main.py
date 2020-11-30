import os.path
import numpy as np
import pandas as pd
from preprocess import preprocess
from myLinearSVM import linearSVM
from myKernelSVM import kernelSVM
from pca import pca

def main():
    ## assume if y_test exists, all csv files exist
    if not os.path.exists('y_test.csv'):
        preprocess()

    n = 600
    X_train = np.asarray(pd.read_csv("X_train.csv"))[:n]
    Y_train = np.asarray(pd.read_csv("y_train.csv"))[:n].reshape([X_train.shape[0],])
    X_test = np.asarray(pd.read_csv("X_test.csv"))
    Y_test = np.asarray(pd.read_csv("y_test.csv")).reshape([X_test.shape[0],])

    ## #2
    data = [X_train, Y_train, X_test, Y_test]
    final_C_lin, error_lin = linearSVM(data)

    ## #3
    final_C_kern, error_kern = kernelSVM(data)

    ## now compare the two for problem #4
    data = pca()
    final_C_lin, error_lin = linearSVM(data, 0.1)
    final_C_kern, error_kern = kernelSVM(data, 100)

    if error_lin < error_kern:
        print("LINEAR SVM PERFORMS BETTER WITH AN ERROR RATE OF {}".format(error_lin))
    elif error_kern < error_lin:
        print("KERNEL SVM PERFORMS BETTER WITH AN ERROR RATE OF {}".format(error_kern))
    else:
        print("BOTH PERFORM EQUALLY WITH AN ERROR RATE OF {}".format(error_lin))

if __name__ == '__main__':
    main()
