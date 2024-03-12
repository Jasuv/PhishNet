import numpy as np
import pandas as pd

# rectified linear unit activation function
def ReLU(z):
    return np.maximum(0, z)

# undo ReLU function
def ReLU_prime(z):
    return z > 0

# softmax activation function
def softmax(z):
    z = z - np.max(z, 0)
    a = np.exp(z) / np.sum(np.exp(z), 0)
    return a

# forward propagation
def forward_prop(w1, b1, w2, b2, x):
    z1 = np.dot(x, w1) + b1
    a1 = ReLU(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# one-hot encoding
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

# backward propagation
def backward_prop(z1, a1, z2, a2, w2, x, y):
    m = y.shape[0]
    dz2 = a2 - one_hot(y)
    dw2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2, axis=0)
    dz1 = np.dot(dz2, w2.T) * ReLU_prime(z1)
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
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

# train model using gradient descent
def gradient_descent(x, y, epochs, alpha):
    np.random.seed(42)
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
            print(f"Iteration: {epoch}")
            print(f"Accuracy: {get_accuracy(get_predictions(a2), y)}")
    return w1, b1, w2, b2

dataset = pd.read_csv("phishing_site_urls.csv")
np.random.shuffle(dataset)

features = ['-', 'confirm', 'account', 'signin', 'update', 'logon', 
            'cmd', 'admin', '.', '!', '&', ',', '#', '$', '%', '@']

# extract features from URL
def extract_features(url):
    feature_count = [url.count(feature) for feature in features]
    return np.array(feature_count)
x = np.array(extract_features(url) for url in dataset['URL'])
y = np.array(dataset['Label'])

# split dataset into train and test sets
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# begin training
w1, b1, w2, b2 = gradient_descent(x_train, y_train, epochs=1000, alpha=0.01)

def make_predictions(x, w1, b1, w2, b2):
    a2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predictions(a2)
    return predictions

def test_prediction(index, w1, b1, w2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], w1, b1, w2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

test_prediction(0, w1, b1, w2, b2)