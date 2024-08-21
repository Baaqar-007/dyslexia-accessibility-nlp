import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt


data = pd.read_csv('../dataset/emnist-letters-train.csv')

data = np.array(data) 
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T 
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev/255.

data_train = data[1000:m].T 
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255.

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(27, 10) - 0.5
    b2 = np.random.rand(27, 1) - 0.5
    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(z, 0)

def softmax(z):
    a = np.exp(z - np.max(z)) / np.sum(np.exp(z - np.max(z)), axis=0)
    return a

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1 
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2 

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max()+1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T 
    return one_hot_y

def deriv_ReLU(z):
    return z > 0

def back_prop(z1, a1, z2, a2, w2, x, y):
    one_hot_y = one_hot(y) 
    dz2 = a2 - one_hot_y
    dw2 = dz2.dot(a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)

    dw1 = dz1.dot(x.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2 

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 15 == 0:
            print("iteration:", i)
            print("Accuracy:", get_accuracy(get_predictions(a2), y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 1000, 0.55)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", chr(int(prediction[0])+64))
    print("Label: ", chr(int(label)+64))
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, w1, b1, w2, b2)
test_prediction(1, w1, b1, w2, b2)
test_prediction(2, w1, b1, w2, b2)
test_prediction(3, w1, b1, w2, b2)

dev_predictions = make_predictions(X_dev, w1, b1, w2, b2)
print(get_accuracy(dev_predictions, Y_dev))