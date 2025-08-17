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
    w1 = np.random.randn(input_size, hidden_layer_size) * np.sqrt(2 / hidden_layer_size)
    b1 = np.zeros((1 , hidden_layer_size))
    w2 = np.random.randn(hidden_layer_size, output_size) * np.sqrt(2 / output_size)
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
    layer1 = X @ w1 + b1
    A1 = sigmoid(layer1)
    layer2 = A1 @ w2 + b2
    A2 = sigmoid(layer2)

    return A1, A2


def back_prop(w1, b1, w2, b2, lr, X, y, a1, a2):
    m = y.shape[0]
    
    # output layer gradients
    dz2 = a2 - y
    dw2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # hidden layer gradients
    da1 = w2 @ dz2.T
    dz1 = da1 @ sigmoid_der(X @ w1 + b1)
    dw1 = X @ dz1.T /m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2-= lr * db2
    return w1, b1, w2, b2


def train(X, y, lr, epoch, input_size, hidden_layer_size, output_size):
    w1, b1, w2, b2 = wt_init(input_size, hidden_layer_size, output_size)
    losses = []
    for i in range(epoch):
        a1, a2 = forward_propagation(w1, b1, w2, b2, X)
        error = loss(y, a2)
        losses.append(error)
        w1, b1, w2, b2 = back_prop(w1, b1, w2, b2, lr, X, y, a1, a2)
        if i % 20 == 0:
            print(f'At iteration {i}, loss = {loss}')
    return w1, b1, w2, b2

def generate_xor_data(n_samples=1000, noise=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.randint(0, 2, (n_samples, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int).reshape(-1, 1)

    # Add small noise to input
    X = X + noise * np.random.randn(n_samples, 2)
    return X, y

# Generate dataset
X, y = generate_xor_data(n_samples=1000, noise=0.1)

# Shuffle and split into train/test
split_idx = int(0.8 * len(X))  # 80% train, 20% test

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

learned_w1, learned_b1, learned_w2, learned_b2 = train(X_train, y_train, lr=0.01, epoch=200, input_size = 2, hidden_layer_size = 3, output_size=1)
_, predicted_output = forward_propagation(learned_w1, learned_b1, learned_w2, learned_b2, X_test)
predicted_classes = (predicted_output > 0.5).astype(int)
loss = np.mean(predicted_classes == y_test)
print(loss)