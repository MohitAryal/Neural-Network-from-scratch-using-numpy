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
    w1 = np.random.randn(input_size, hidden_layer_size)
    b1 = np.zeros(hidden_layer_size)
    w2 = np.random.randn(hidden_layer_size, output_size)
    b2 = np.zeros(output_size)
    return w1, b1, w2, b2

