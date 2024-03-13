'''
Based heavily off of SAMSON ZHANG's simple neural network
https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras
'''

import numpy as np
import pandas as pd

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# undo activation function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# softmax activation function
def softmax(z):
    z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return z_exp / np.sum(z_exp, axis=1, keepdims=True) 

# one-hot encoding
def one_hot(y):
    return np.eye(2)[y]

# forward propagation
def forward_prop(w1, b1, w2, b2, x):
    # input to hidden
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    # hidden to output
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# backward propagation
def backward_prop(z1, a1, z2, a2, w2, x, y):
    m = y.shape[0]
    # output to hidden
    dz2 = a2 - one_hot(y)
    dw2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0)
    # hidden to input
    dz1 = np.dot(dz2, w2.T) * sigmoid_prime(z1)
    dw1 = (1 / m) * np.dot(x.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0)
    return dw1, db1, dw2, db2

# update the parameters
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 1)

def get_accuracy(predictions, y):
    return np.mean(predictions == y)

# train model using gradient descent
def gradient_descent(x, y, epochs, alpha):
    # initialize weights and biases with random numbers
    input_size = x.shape[1]
    hidden_size = 128 
    w1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros(hidden_size)
    w2 = np.random.randn(hidden_size, 2)
    b2 = np.zeros(2)
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        # log every 10 results
        if epoch % 10 == 0:
            pre = get_predictions(a2)
            acc = get_accuracy(pre, y)
            print(f"Epoch: {epoch}\nPrediction: {pre} Y: {y}\nAccuracy: {acc}\n")
    return w1, b1, w2, b2

dataset = pd.read_csv("urls.csv")
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Create dataset (features, binary label)
features = ['confirm', 'account', 'signin', 'update', 'login', 'cmd', 'cmnd', 'admin', 'post', 'upload', 
            'exe', 'ph', 'js', 'css', 'my', '/', '_', '-', '.', '?', '!', '&', ',', '#', '$', '%', '@']
def extract_features(url):
    return np.array([url.count(feature) for feature in features])
def encode_labels(y):
    return np.array([1 if label == 'good' else 0 for label in y])
x = np.array([extract_features(url) for url in dataset['URL']])
y = encode_labels(dataset['Label'])

# split dataset into train and test sets
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)
x_url = dataset['URL']
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

'''
for i in range(len(dataset)):
    print(f"URL: {x_url[i]}\nx: {x_train[i]}\ny: {y_train[i]}\n\n")
'''
    
# begin training
w1, b1, w2, b2 = gradient_descent(x_train, y_train, epochs=1000, alpha=0.01)