# stripped down 3 layer NN.
# adapted from https://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): return x*(1-x)

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1], [0.01,0.01, 1] , [0.99,0.01,1] ])    # INPUT DATA ATTRIBUTES
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1], [0.02,0.02, 1] , [0.98,0.02,1] ])    # INPUT DATA ATTRIBUTES
y = np.array([[0,1,1,0,1,0]]).T                          # INPUT DATA LABELS

syn0 = 2*np.random.random((3,4)) - 1                 # RANDOM INITIALIZATION
syn1 = 2*np.random.random((4,1)) - 1
diary=[]
for j in range(60000):
   l1 = sigmoid(X.dot(syn0))                         # FEED FORWARD
   l2 = sigmoid(l1.dot(syn1))
   l2_error = y - l2                                 # RECORD ERROR FOR TRACKING
   diary.append(l2_error.T.dot(l2_error)[0][0])

   l2_delta = l2_error * dsigmoid(l2)                # BACK PROPAGATION
   l1_delta = l2_delta.dot(syn1.T) * dsigmoid(l1)
   syn1 += l1.T.dot(l2_delta)
   syn0 += X.T.dot(l1_delta)

print("final coefficients")                          # DISPLAY RESULTS
print(str(syn0))
print(str(syn1))
print(sigmoid(sigmoid(X.dot(syn0)).dot(syn1)))

def plot_error():
    plt.semilogy(diary)
    plt.show(block=False)

print(" to see convergence, type: plot_error()")

