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
syn1 = 2*np.random.random((4,3)) - 1                 # NEW HIDDEN LAYER
syn2 = 2*np.random.random((3,1)) - 1
diary=[]
for j in range(60000):
   l1 = sigmoid(X.dot(syn0))                         # FEED FORWARD
   l2 = sigmoid(l1.dot(syn1))
   l3 = sigmoid(l2.dot(syn2))
   l3_error = y - l3                                 # RECORD ERROR FOR TRACKING
   diary.append(l3_error.T.dot(l3_error)[0][0])

   l3_delta = l3_error * dsigmoid(l3)                # BACK PROPAGATION
   l2_delta = l3_delta.dot(syn2.T) * dsigmoid(l2)
   l1_delta = l2_delta.dot(syn1.T) * dsigmoid(l1)
   syn2 += l2.T.dot(l3_delta)
   syn1 += l1.T.dot(l2_delta)
   syn0 += X.T.dot(l1_delta)

print("final coefficients")                          # DISPLAY RESULTS
print(str(syn0))
print(str(syn1))
print(str(syn2))

print("")
print("OUTPUTS")
print(sigmoid(sigmoid(sigmoid(X.dot(syn0)).dot(syn1)).dot(syn2)))

def plot_error():
    plt.semilogy(diary)
    plt.show(block=False)

print(" to see convergence, type: plot_error()")
