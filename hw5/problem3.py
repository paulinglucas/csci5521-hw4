import numpy as np
from sklearn.svm import SVC

def main():

    X = np.array([[0.2,0.8,1],[2,-1,1],[0.8,2,1],[2,1,1],[1.5,1,1],[-0.5,-0.5,1],[-0.5,-2,1],[-2,-1,1],[-1,-1.5,1]])
    y = np.array([1,1,1,1,1,-1,-1,-1,-1])

    clf = SVC(kernel='linear')

    clf.fit(X, y)

    ## part a
    print("a. Support vectors: \n{}\n".format(clf.support_vectors_))

    ## part b
    print("b. Weight Vector: [{} {}]".format(round(clf.coef_[0][0],6),round(clf.coef_[0][1],6)))
    print("Bias: {}".format(round(clf.intercept_[0],6)))

    ## part c
    print("c. w = c - d = ")
    alphas = abs(clf.dual_coef_)
    n = clf.n_support_[0]
    print(alphas)
    c = np.sum((clf.support_vectors_[n:].T*alphas[0][n:]).T,0)
    d = np.sum((clf.support_vectors_[:n].T*alphas[0][:n]).T,0)

    print("w = c - d = ")
    print(c, "-", d)

if __name__ == "__main__":
    main()
