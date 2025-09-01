import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    '''Returns the sigmoid of given input array'''
    x = np.clip(x, -500, 500)
    sig = 1 / (1 + np.exp(-x))
    return sig


def sigmoid_der(x):
    '''Returns the derivative input(which is the output of the sigmoid function)'''
    y = sigmoid(x)
    der = y * (1 - y)
    return der


def wt_init(input_size, hidden_layer_size, output_size):
    '''Initializes the weights and biases of the network.'''
    w1 = np.random.randn(input_size, hidden_layer_size) * (2 / (input_size + hidden_layer_size))
    b1 = np.zeros((1 , hidden_layer_size))
    w2 = np.random.randn(hidden_layer_size, output_size) * (2 / (hidden_layer_size + output_size))
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2


def loss(y, y_pred):
    '''Calculates the loss between actual value and predicted probability.'''
    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps , 1.0-eps)
    cross_entropy_loss = - y * np.log(y_pred_clipped) - (1 - y) * np.log(1 - y_pred_clipped)
    return np.mean(cross_entropy_loss)


def forward_propagation(w1, b1, w2, b2, X):
    '''Performs forward propagation on input X and returns the predicted probability'''
    layer1 = X @ w1 + b1
    A1 = sigmoid(layer1)
    layer2 = A1 @ w2 + b2
    A2 = sigmoid(layer2)

    return A1, A2


def back_prop(w1, b1, w2, b2, lr, X, y, a1, a2):
    '''Performs back propagation and updates the weights and biases accordingly'''
    m = y.shape[0]
    
    # output layer gradients
    dz2 = a2 - y
    dw2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # hidden layer gradients
    da1 = dz2 @ w2.T
    dz1 = da1 * sigmoid_der(X @ w1 + b1)
    dw1 = X.T @ dz1 /m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # update weights
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2-= lr * db2
    return w1, b1, w2, b2


def train(X, y, lr, epoch, input_size, hidden_layer_size, output_size):
    '''Trains the model to learn weights and biases by minimizing loss on training data'''
    w1, b1, w2, b2 = wt_init(input_size, hidden_layer_size, output_size)
    losses = []
    for i in range(epoch + 1):
        a1, a2 = forward_propagation(w1, b1, w2, b2, X)
        error = loss(y, a2)
        losses.append(error)
        w1, b1, w2, b2 = back_prop(w1, b1, w2, b2, lr, X, y, a1, a2)
        if i % 500 == 0:
            print(f'At iteration {i}, loss = {error}')
    return w1, b1, w2, b2, losses

def generate_xor_data(n_samples=1000, noise=0.1, seed=42):
    '''Generates noise added xor dataset'''
    np.random.seed(seed)
    X = np.random.randint(0, 2, (n_samples, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int).reshape(-1, 1)

    # Add small noise to input
    X = X + noise * np.random.randn(n_samples, 2)
    return X, y

# Generate training dataset
X_train, y_train = generate_xor_data(n_samples=1000, noise=0.01)


def generate_clean_xor_data(n_samples=500, seed=45):
    ''' Generates xor dataset without noise'''
    np.random.seed(seed)
    # Base XOR inputs and labels
    base_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]], dtype=np.float32)
    base_labels = np.array([0, 1, 1, 0], dtype=np.int64)  # XOR: 0^0=0, 0^1=1, etc.

    # Repeat the 4 XOR points to get n_samples total
    reps = n_samples // 4
    remainder = n_samples % 4

    X = np.tile(base_inputs, (reps, 1))
    y = np.tile(base_labels, reps)

    # Add remaining samples (to reach exact n_samples)
    if remainder > 0:
        X = np.vstack([X, base_inputs[:remainder]])
        y = np.concatenate([y, base_labels[:remainder]])

    return X, y

# Generate testing dataset
X_test, y_test = generate_clean_xor_data(seed=45)

# Train the model
learned_w1, learned_b1, learned_w2, learned_b2, losses = train(X_train, y_train, lr=0.5, epoch=4000, input_size = 2, hidden_layer_size = 14, output_size=1)

# Evaluate on the test set
_a, predicted_output = forward_propagation(learned_w1, learned_b1, learned_w2, learned_b2, X_test)
predicted_classes = (predicted_output > 0.5).astype(int)
accuracy = np.mean((predicted_classes.flatten() == y_test))
print(np.mean(accuracy))