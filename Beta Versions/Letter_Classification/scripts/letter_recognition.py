import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numba as nb

# Load and preprocess the dataset
data = pd.read_csv('../dataset/emnist-letters-train.csv').to_numpy()
np.random.shuffle(data)
m, n = data.shape
data_dev = data[:1000].T
Y_dev, X_dev = data_dev[0], data_dev[1:] / 255.

data_train = data[1000:m].T
Y_train, X_train = data_train[0], data_train[1:] / 255.

# Initialize parameters
def init_params():
    return (
        np.random.randn(10, 784) * 0.01,
        np.zeros((10, 1)),
        np.random.randn(27, 10) * 0.01,
        np.zeros((27, 1)),
    )

@nb.jit(nopython=True)
def ReLU(z):
    return np.maximum(z, 0)

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_prop(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(y, num_classes):
    one_hot_y = np.eye(num_classes)[y].T
    return one_hot_y

@nb.jit(nopython=True)
def deriv_ReLU(z):
    return (z > 0).astype(z.dtype)

def back_prop(z1, a1, z2, a2, w2, x, y):
    one_hot_y = one_hot(y, a2.shape[0])
    dz2 = a2 - one_hot_y
    m = x.shape[1]
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = np.dot(w2.T, dz2) * deriv_ReLU(z1)
    dw1 = np.dot(dz1, x.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    return dw1, db1, dw2, db2

@nb.jit(nopython=True)
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, axis=0)

def get_accuracy(predictions, y):
    return np.mean(predictions == y)

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 15 == 0:
            print(f"Iteration {i}: Accuracy = {get_accuracy(get_predictions(a2), y):.4f}")
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 1000, 0.45)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print(f"Prediction: {chr(int(prediction[0]) + 64)}")
    print(f"Label: {chr(int(label) + 64)}")
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, w1, b1, w2, b2)
test_prediction(1, w1, b1, w2, b2)
test_prediction(2, w1, b1, w2, b2)
test_prediction(3, w1, b1, w2, b2)

dev_predictions = make_predictions(X_dev, w1, b1, w2, b2)
print(f"Development Set Accuracy: {get_accuracy(dev_predictions, Y_dev):.4f}")
