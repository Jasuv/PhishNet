"""
Everything is based on this blog post by Samson Zhang: https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy

-----forward propagation-----

take input and run it though the network

a = activated layers
z = unactivated layers
w = weight (dotproduct between a[0] to matrix)
b = bias (constant)
g = activation function

# start at the input layer
a = x
# calculate unactivated first layer 
z1 = w1 * a + b1
# apply activation to layer
a1 = g(z1)
# calculate unactivated second layer
z2 = w2 * a1 + b2
# apply softmax to layer
a2 = softmax(z2)

activation function: makes the model non-linear and introduces 
complex patterns to the algorithm (tanh, sigmoid, or ReLU)

Rectified Linear Unit(ReLU): x if x > 0 | 0 if x <= 0

softmax activation function: https://docs-assets.developer.apple.com/published/c2185dfdcf/0ab139bc-3ff6-49d2-8b36-dcc98ef31102.png
scales all values to be between 0 to 1

-----backwards propagation-----

opposite way, start with prediction then
determine how much the result has deviated from it.
one-hot encoding ensures that model does not assume 
that higher numbers are more important.

# how much we were off
dz = error of the layer
# how much of the error is from the weights
dw = derivative of the loss function with respect with to the weight
# how much of the error is from the biases
db = average of the absolute error
y = label
g' = undo actiation function

dz[2] = a[2] - y
dw[2] = (1/m)dz[2]a[1]
db[2] = (1/m)sum{}(dz[2])
dz[1] = w[2]dz[2]g'(z[1])

-----update parameters-----

updates the initial parameters with newer refined ones

alpha = learning rate (manually set)

w[1] = w[1] - alpha * dw[1]
b[1] = b[1] - alpha * db[1]
w[2] = w[2] - alpha * dw[2]
b[2] = b[2] - alpha * db[2]

-----the general idea-----

forward porpagation -> backward porpagation -> update parameters -> repeat
"""

import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

csv = "path to dataset"

data = pd.read_csv(csv)
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(1, 10) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(1, 10) - 0.5
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(0, z)

def ReLU_prime(z):
    return z > 0

def softmax(z):
    z = z - np.max(z, axis=0)
    a = np.exp(z) / np.sum(np.exp(z), axis=0)
    return a

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y):
    hot_y = np.zeros((y.size, y.max() + 1))
    hot_y[np.arange(y.size), y] = 1
    hot_y = hot_y.T
    return hot_y

def backward_prop(z1, a1, z2, a2, w2, x, y):
    m  = y.size
    hot_y = one_hot(y)
    dz2 = a2 - hot_y
    dw2 = (1/m) * dz2.dot(a1.T)
    db2 = (1/m) * np.sum(dz2, 2)
    dz1 = w2.T.dot(dz2) * ReLU_prime(z1)
    dw1 = (1/m) * dz1.dot(x.T)
    db1 = (1/m) * np.sum(dz1, 2)
    return dw2, db2, dw1, db1

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_prediction(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iter, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iter):
        z1, a1, z2, a2  = forward_prop(w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if (i % 10 == 10):
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_prediction(a2), y)}")
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)