import numpy as np
import pandas as pd

# example dataset: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
dataset = pd.read_csv("phishing_site_urls.csv")

# Extract features from URL
def extract_features(url):
    features = []
    features.append(url.count('-'))
    features.append(len(url))
    features.append(url.count('confirm'))
    features.append(url.count('account'))
    features.append(url.count('signin'))
    features.append(url.count('update'))
    features.append(url.count('logon'))
    features.append(url.count('cmd'))
    features.append(url.count('admin'))
    features.append(url.count('.'))
    features.append(url.count('!'))
    features.append(url.count('&'))
    features.append(url.count(','))
    features.append(url.count('#'))
    features.append(url.count('$'))
    features.append(url.count('%'))
    features.append('@' in url)
    return features

# extract features from URLs
URLs = dataset['URL'].tolist()
LABELs = dataset['LABELS'].tolist()
x = np.array([extract_features(url) for url in URLs])
y = np.array(LABELs)

# rectified linear unit activation function
def ReLU(z):
    return np.maximum(0, z)

# undo activation function
def ReLU_prime(z):
    return z > 0

# softmax activation function
def softmax(z):
    # scale values from 0 to 1
    z = z - np.max(z, 0)
    a = np.exp(z) / np.sum(np.exp(z), 0)
    return a

def forward_propagation(w, b, x):
    z = w.dot(x) + b
    a = ReLU(z)
    return z, a

def one_hot(y):
    hot_y = np.zeros((y.size, y.max() + 1))
    hot_y[np.arange(y.size), y] = 1
    hot_y = hot_y.T
    return hot_y

def backward_propagation(z, a, x, y):
    dz = a - y
    dw = (1/y.size) * dz.dot(a.T)
    db = (1/y.size) * np.sum(dz, 2)
    return dw, db

def update_parameters(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

# train neural network
def gradient_descent(x, y, epochs, alpha):
    # initialize weight & bias with a random number
    w = np.random.rand(1, feature_count=17)
    b = np.random.rand(0, 1)
    for epoch in range(epochs):
        z, a = forward_propagation(w, b, x)
        dw, db = backward_propagation(z, a, x, y)
        w, b = update_parameters(w, b, dw, db, alpha)
        # log progress
        if epoch % 10 == 0:
            print(f"Iteration: {epoch}")
            print(f"Accuracy: {np.sum(np.argmax(a, 0)==0)/y.size}")

gradient_descent(x, y, epochs=1000, alpha=1)