import numpy as np
import pandas as pd

epochs = 1000
feature_count = 17

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

def forward_propagation(w, x):
    return ReLU(np.dot(x, w))

def backward_propagation(x, y, output):
    error = y - output
    delta = error * ReLU_prime(output)
    adjustment = np.dot(x.T, delta)
    return adjustment

def update_parameters(w, adjustment):
    w += adjustment

# initialize a random weight
w = np.random.random(1, feature_count) 

# train neural network
for epoch in range(epochs):
    output = forward_propagation(w, x)
    adjustment = backward_propagation(x, y, output)
    update_parameters(w, adjustment)
    # log progress
    if epoch % 10 == 0:
        accuracy = np.mean((output > 0.5) == y)
        print(f"Iteration: {epoch}")
        print(output.flatten())
        print(y.flatten())
        print(accuracy)