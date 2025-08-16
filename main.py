import numpy as np


def sigmoid(x):
    '''Returns the sigmoid of given input array'''
    sig = 1 / (1 + np.exp(-x))
    return sig


def sigmoid_der(sig):
    '''Returns the derivative input(which is the output of the sigmoid function)'''
    der = sig * (1 - sig)
    return der


def wt_init(input_size, hidden_layer_size, output_size):
    '''Initializes the weights and biases of the network.'''
    w1 = np.random.randn(input_size, hidden_layer_size)
    b1 = np.zeros((1 , hidden_layer_size))
    w2 = np.random.randn(hidden_layer_size, output_size)
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2


def loss(y, y_pred):
    '''Calculates the loss between actual value and predicted probability.'''
    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps , 1.0-eps)
    cross_entropy_loss = - y * np.log(y_pred_clipped) - (1 - y) * np.log(1 - y_pred_clipped)
    return cross_entropy_loss


def forward_propagation(w1, b1, w2, b2, X):
    '''Performs forward propagation on input X and returns the predicted probability'''
    layer1 = w1 * X + b1
    A1 = sigmoid(layer1)
    layer2 = w2 * A1 + b2
    A2 = sigmoid(layer2)

    return A1, A2


def back_prop(w1, b1, w2, b2, X, y, y_pred, lr):
    m = y.shape[0]

    a1, a2 = forward_propagation(w1, b1, w2, b2, X)
    
    # output layer gradients
    dz2 = y_pred - y
    dw2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # hidden layer gradients
    da1 = dz2 @ w2.T
    dz1 = da1 @ sigmoid_der(w1 @ X + b1)
    dw1 = X.T @ dz1 /m
    db1 = np.sum(dz1, axis=0, keep_dims=True) / m

    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

        