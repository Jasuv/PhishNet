Everything is based on this blog post by Samson Zhang: https://www.samsonzhang.com/2020/11/24/understanding-the-math-behind-neural-networks-by-building-one-from-scratch-no-tf-keras-just-numpy

-----forward propagation-----

take input and run it though the network

a = activated layers
z = unactivated layers
w = weight (dotproduct between a[0] to matrix)
b = bias (constant)
g = activation function

start at the input layer
a = x
calculate unactivated first layer 
z1 = w1 * a + b1
apply activation to layer
a1 = g(z1)
calculate unactivated second layer
z2 = w2 * a1 + b2
apply softmax to layer
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

how much we were off
dz = error of the layer
how much of the error is from the weights
dw = derivative of the loss function with respect with to the weight
how much of the error is from the biases
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

-----testing-----

Epoch: 0
Prediction: [0 0 0 ... 0 0 0] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.28478460712302833

Epoch: 10
Prediction: [1 1 1 ... 0 0 0] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.36878691896713356

Epoch: 20
Prediction: [1 1 1 ... 0 1 0] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.5538300157460249

Epoch: 30
Prediction: [1 1 1 ... 0 1 0] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.5914884999408386

Epoch: 40
Prediction: [1 1 1 ... 0 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.6069478196761597

Epoch: 50
Prediction: [1 1 1 ... 0 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.6083927222419426

...

Epoch: 840
Prediction: [1 1 0 ... 1 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.7574930144080678

Epoch: 850
Prediction: [1 1 0 ... 1 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.7575590020843004

Epoch: 860
Prediction: [1 1 0 ... 1 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.7576704985027624

Epoch: 870
Prediction: [1 1 0 ... 1 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.7577637914243326

Epoch: 880
Prediction: [1 1 0 ... 1 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.7578616352201258

Epoch: 890
Prediction: [1 1 0 ... 1 1 1] Y: [1 1 0 ... 1 1 0]
Accuracy: 0.758109657865276

---plans---

- add make_prediction
- optimize for faster learning & better acc
- save model to file like .pt