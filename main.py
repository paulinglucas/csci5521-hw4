import os.path
from preprocess import preprocess
from myLinearSVM import linearSVM
from myKernelSVM import kernelSVM
from pca import pca

def main():
    ## assume if y_test exists, all csv files exist
    if not path.exists('y_test.csv'):
        preprocess()

    linearSVM()

    kernelSVM()

    pca()


if __name__ == '__main__':
    main()
