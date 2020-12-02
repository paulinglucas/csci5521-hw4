import os.path
import numpy as np
import pandas as pd
from preprocess import preprocess
from myLinearSVM import linearSVM
from myKernelSVM import kernelSVM
from myMLP import MLP
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

    ## setup for fashion dataset
    ftrain = np.genfromtxt("fashion-mnist_train.csv", delimiter=",")
    ftrain0 = ftrain[ftrain[:, 0] == 0]
    ftrain1 = ftrain[ftrain[:, 0] == 1]
    X_ftrain = np.concatenate((ftrain0[:, 1:], ftrain1[:, 1:]))
    Y_ftrain = np.concatenate((ftrain0[:, 0], ftrain1[:, 0]))
    ftest = np.genfromtxt("fashion-mnist_test.csv", delimiter=",")
    ftest0 = ftest[ftest[:, 0] == 0]
    ftest1 = ftest[ftest[:, 0] == 1]
    X_ftest = np.concatenate((ftest0[:, 1:], ftest1[:, 1:]))
    Y_ftest = np.concatenate((ftest0[:, 0], ftest1[:, 0]))

    X_ftrain /= 255
    X_ftest /= 255

    ## #2
    data = [X_train, Y_train, X_test, Y_test]
    fdata = [X_ftrain, Y_ftrain, X_ftest, Y_ftest]
    final_C_lin, error_lin = linearSVM(data)
    ffinal_C_lin, ferror_lin = linearSVM(fdata)

    ## #3
    final_C_kern, error_kern = kernelSVM(data)
    ffinal_C_kern, ferror_kern = kernelSVM(fdata)

    ## now compare the two for problem #4
    data_pca = pca()
    final_C_lin, error_lin = linearSVM(data_pca, final_C_lin)
    final_C_kern, error_kern = kernelSVM(data_pca, final_C_kern)

    ## #5 multi-layer perceptron for extra credit
    final_C_mlp, error_mlp = MLP(data)

    if error_lin < error_kern:
        print("LINEAR SVM PERFORMS BETTER WITH AN ERROR RATE OF {}".format(error_lin))
    elif error_kern < error_lin:
        print("KERNEL SVM PERFORMS BETTER WITH AN ERROR RATE OF {}".format(error_kern))
    else:
        print("BOTH PERFORM EQUALLY WITH AN ERROR RATE OF {}".format(error_lin))

if __name__ == '__main__':
    main()
