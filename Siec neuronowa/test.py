from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()

pixels = digits.data
pixels = np.array([matrix / 16.0 for matrix in pixels])
numbers = digits.target

pixels_train, pixels_test, numbers_train, numbers_test = train_test_split(pixels, numbers, test_size=0.2)
# wektor obrazu to teraz kolumna
pixels_train = pixels_train.T
pixels_test = pixels_test.T
m = numbers_train.size


def init_params():
    W1 = np.random.rand(10, 64) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def tanh(Z):
    return np.tanh(Z)


def tanh_deriv(Z):
    return 1 - np.tanh(Z) ** 2


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = tanh(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = tanh(Z2)
    return Z1, A1, Z2, A2


def get_perfect_clasification(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def mse_cost(A, Y):
    return (1 / (2 * m)) * np.sum((A - get_perfect_clasification(Y))**2)


def mse_cost_derivative_bias(dZ):
    return 1 / m * np.sum(dZ)


def mse_cost_derivative_weights(dZ, A):
    return 1 / m * dZ.dot(A.T)


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    dZ2 = A2 - get_perfect_clasification(Y)
    dW2 = mse_cost_derivative_weights(dZ2, A1)
    db2 = mse_cost_derivative_bias(dZ2)

    dZ1 = W2.T.dot(dZ2) * tanh_deriv(Z1)
    dW1 = mse_cost_derivative_weights(dZ1, X)
    db1 = mse_cost_derivative_bias(dZ1)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(pixels_train, numbers_train, 0.1, 500)
