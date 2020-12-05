import numpy as np
from sklearn.svm import SVC

def main():

    X = np.array([[0.2,0.8],[2,-1],[0.8,2],[2,1],[1.5,1],[-0.5,-0.5],[-0.5,-2],[-2,-1],[-1,-1.5]])
    y = np.array([1,1,1,1,1,-1,-1,-1,-1])

    clf = SVC(kernel='linear')

    clf.fit(X, y)

    ## part a
    print("a. Support vectors: \n{}\n".format(clf.support_vectors_))

    ## part b
    w = clf.coef_[0]
    b = round(clf.intercept_[0],6)
    print("b. Weight Vector: [{} {}]".format(round(w[0],6),round(w[1],6)))
    print("Bias: {}".format(b))
    print()

    ## part b
    print("c. w = c - d = ")
    alphas = abs(clf.dual_coef_)
    n = clf.n_support_[0]
    c = np.sum((clf.support_vectors_[n:].T*alphas[0][n:]).T,0)
    d = np.sum((clf.support_vectors_[:n].T*alphas[0][:n]).T,0)

    print("w = c - d = ", end="")
    print(c, "-", d)
    print()

    ## part b
    print("Decision Function: y = np.sign(w.T * X + b)")
    print("= np.sign([{} {}].T * X + {})".format(round(w[0],6),round(w[1],6),round(b,6)))
    x1 = X[0]
    x2 = X[4]
    x3 = X[7]


    ## part c
    print("c. Distance = (w.T*X + b) / norm(w)")
    print("Distance from {}: {}".format(x1, abs(clf.decision_function([x1]) / np.linalg.norm(w))))
    print("Distance from {}: {}".format(x2, abs(clf.decision_function([x2]) / np.linalg.norm(w))))
    print("Distance from {}: {}".format(x3, abs(clf.decision_function([x3]) / np.linalg.norm(w))))
    print()

    ## part d
    print("d. Removing (-0.5, -0.5) will change the decision boundary because", end="")
    print("the point is a support vector. (0.8, 0.2) is not, so removing that point will", end="")
    print("not affect the decision boundary")
    print()

    ## part e
    print("e. This would lie on the positive decision boundary, and would certainly affect it.", end="")
    print("We would proceed to then use RBF to fit the test data without error.")
    print()

    ## part f
    print("f. Any value other than 1 will change the hard margin SVM to a soft margin SVM.", end="")
    print("The bigger C is, the 'tighter' the decision boundary would become in order to fit", end="")
    print("all of the training samples. The smaller C is, the 'looser' it would get, putting", end="")
    print("less emphasis on a perfect fit, allowing 'stragglers' to be misclassified. Only support vectors", end="")
    print("are affected.")
    print()

if __name__ == "__main__":
    main()
