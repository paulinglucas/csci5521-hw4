import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

## Construct a class to handle the simple perceptron used in problem b
class MLP:
    def __init__(self, weights = (np.random.random([2,3]) - .5), lr=.000005, max_epoch = 150, verbose = False):
#        self.w = (np.random.random([2,3]) - .5)
        self.w = weights
        self.v = (np.random.random(3) - .5)
        self.lr = lr
        self.max_epoch = max_epoch
        self.training_loss = []
        self.verbose = verbose
        
    def fit(self, X, y):
        for epoch in range(self.max_epoch):
            X, y = self.shuffle(X,y)
            training_loss = 0
            for i in range(len(X)):
                w1 = self.w[0]
                w2 = self.w[1]
                v = self.v
                z = np.array([1., self.step(X[i].dot(w1)), self.step(X[i].dot(w2))])
                y_hat = self.step(z.dot(v))
                self.v += self.lr * (y[i] - y_hat) * z
                self.w = np.array([w1+self.lr*(y[i]-y_hat)*v[1]*z[1]*(1-z[1])*X[i], w2+self.lr*(y[i]-y_hat)*v[2]*z[2]*(1-z[2])*X[i]])
                training_loss += abs(y[i] - y_hat)
            self.training_loss.append(np.mean(training_loss))
            if self.verbose: print("Epoch " + str(epoch+1) + ": " + str(self.training_loss[epoch]))
            if self.training_loss[epoch] == 0: break

    def predict(self, X):
        predictions = np.empty(len(X))
        for i, x in enumerate(X):
            z = np.array([1., self.step(x.dot(self.w[0])), self.step(x.dot(self.w[1]))])
            predictions[i] = self.step(z.dot(self.v))
        return predictions

    def relu(self, x):
        if x > 0: return x
        else: return 0

    def sigmoid(self, s):
        try:
            return float(1./(1+np.e**(-1*s)))
        except OverflowError:
            return 0.0
    
    def step(self, x):
        if x > 0: return 1
        else: return 0

    def shuffle(self, Xtrain, ytrain):
        shuffled_index = np.arange(0, len(Xtrain))
        np.random.shuffle(shuffled_index)

        ret_Xtrain = np.empty(Xtrain.shape)
        ret_ytrain = np.empty(ytrain.shape)
        for i in np.arange(0,len(Xtrain)):
            ret_Xtrain[i] = Xtrain[shuffled_index[i]]
            ret_ytrain[i] = ytrain[shuffled_index[i]]
        return ret_Xtrain, ret_ytrain
        
## Construct a class to handle the simple perceptron used in problem a
class Perceptron:
    def __init__(self, d=2, lr=.0001, max_epoch = 200, verbose = False):
        self.w = (np.random.random(d+1) - .5) * .1
        self.lr = lr
        self.max_epoch = max_epoch
        self.training_loss = []
        self.verbose = verbose
        
    def fit(self, X, y):
        for epoch in range(self.max_epoch):
            training_loss = 0
            for i in range(len(X)):
                z = self.step(X[i].dot(self.w))
                self.w += self.lr * (y[i] - z) * X[i]
                training_loss += abs(y[i] - z)
            self.training_loss.append(np.mean(training_loss))
            if self.verbose: print("Epoch " + str(epoch+1) + ": " + str(self.training_loss[epoch]))
            if self.training_loss[epoch] == 0: break

    def predict(self, X):
        predictions = np.empty(len(X))
        for i, x in enumerate(X):
            predictions[i] = self.step(x.dot(self.w))
        return predictions
                
    def step(self, x):
        if x > 0: return 1
        else: return 0

## Construct the data for problem a, figure 2
n = 10000
X = np.empty([n,2])
y = np.zeros(n)
for i in range(n):
    X[i] = np.random.random(size=2) * 20 - 10
    if X[i][0] < 1.: y[i] = 1
X = np.c_[np.ones(n), X]

## Do the model fitting and predicting for problem a, figure 2
model = Perceptron()
model.fit(X,y)
pred = model.predict(X)
print("Final normalized weights vector: " + str(model.w / np.linalg.norm(model.w)))
w1 = model.w / np.linalg.norm(model.w)
print(confusion_matrix(y, model.predict(X)))

## save a plot of the results to visually verify our model's success (problem a, figure 2)
plt.figure()
plt.plot(X[pred == 0][:,1], X[pred == 0][:,2], 'o', markersize = 5, color = "blue", alpha = .3, label = "Class 1")
plt.plot(X[pred == 1][:,1], X[pred == 1][:,2], 'o', markersize = 5, color = "red", alpha = .3, label = "Class 2")
plt.plot(np.ones(100), np.arange(-10,10,.2), color = "black", label = "True Label Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Problem #2, part a, figure 2")
plt.legend()
#plt.show()
plt.savefig("fig_a_2")

##################################################

## Construct the data for problem a, figure 3
n = 10000
X = np.empty([n,2])
y = np.zeros(n)
for i in range(n):
    X[i] = np.random.random(size=2) * 20 - 10
    if X[i][1] > X[i][0] - 1.: y[i] = 1
X = np.c_[np.ones(n), X]

## Do the model fitting and predicting for problem a, figure 3
model = Perceptron()
model.fit(X,y)
pred = model.predict(X)
print("Final normalized weights vector: " + str(model.w / np.linalg.norm(model.w)))
w2 = model.w / np.linalg.norm(model.w)
print(confusion_matrix(y, model.predict(X)))

## save a plot of the results to visually verify our model's success (problem a, figure 3)
plt.figure()
plt.plot(X[pred == 0][:,1], X[pred == 0][:,2], 'o', markersize = 5, color = "blue", alpha = .3, label = "Class 1")
plt.plot(X[pred == 1][:,1], X[pred == 1][:,2], 'o', markersize = 5, color = "red", alpha = .3, label = "Class 2")
plt.plot(np.arange(-10,10,.2), np.arange(-10,10,.2)-1, color = "black", label = "True Label Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Problem #2, part a, figure 3")
plt.legend()
#plt.show()
plt.savefig("fig_a_3")

##################################################

## Construct the data for problem b
n = 1000
X = np.empty([n,2])
y = np.zeros(n)
for i in range(n):
    X[i] = np.random.random(size=2) * 20 - 10
    if X[i][1] > X[i][0] - 1. and X[i][0] < 1.: y[i] = 1
X = np.c_[np.ones(n), X]

## Do the model fitting and predicting for problem a, figure 3
model = MLP(weights = np.array([w1,w2]), verbose=True)
model.fit(X,y)
pred = model.predict(X)
print("Final normalized w1 vector: " + str(model.w[0] / np.linalg.norm(model.w[0])))
print("Final normalized w2 vector: " + str(model.w[1] / np.linalg.norm(model.w[1])))
print("Final normalized v vector " + str(model.v / np.linalg.norm(model.v)))
print(confusion_matrix(y, model.predict(X)))

## save a plot of the results to visually verify our model's success (problem a, figure 3)
plt.figure()
plt.plot(X[pred == 0][:,1], X[pred == 0][:,2], 'o', markersize = 5, color = "blue", alpha = .3, label = "Class 1")
plt.plot(X[pred == 1][:,1], X[pred == 1][:,2], 'o', markersize = 5, color = "red", alpha = .3, label = "Class 2")
plt.plot(np.arange(-10,1,.05), np.arange(-10,1,.05)-1, color = "black", label = "True Label Boundary")
plt.plot(np.ones(100), np.arange(0, 10, .1), color = "black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Problem #2, part b")
plt.legend()
#plt.show()
plt.savefig("fig_b")

plt.plot(model.training_loss)
plt.show()
