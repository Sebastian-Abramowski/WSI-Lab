from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from abc import abstractmethod, ABC
from typing import List
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


class Layer(ABC):
    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative) -> np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.TanhLayer = Tanh()
        self.Loss = Loss()

        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.random.rand(output_size, 1) - 0.5

    def forward(self, X1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.weights.dot(X1) + self.bias
        a = self.TanhLayer.forward(z)
        return a, z

    def backward(self, X1: np.ndarray, Z1: np.ndarray, dZ2: np.ndarray, W2: np.ndarray
                 ) -> tuple[np.ndarray, float, np.ndarray]:
        dZ = self.TanhLayer.backward(Z1, dZ2, W2)
        dW = self.Loss.mse_cost_derivative_weights(dZ, X1)
        db = self.Loss.mse_cost_derivative_bias(dZ)

        return dW, db, dZ

    def backward_last_layer(self, A1: np.ndarray, A2: np.ndarray, Y) -> tuple[np.ndarray, float, np.ndarray]:
        dZ = self.TanhLayer.backward_last_layer(A2, Y)
        dW = self.Loss.mse_cost_derivative_weights(dZ, A1)
        db = self.Loss.mse_cost_derivative_bias(dZ)

        return dW, db, dZ

    def update_with_gradient_descent_step(self, dW: np.ndarray, db: float) -> tuple[np.ndarray, float]:
        self.weights = self.weights - self.learning_rate * dW
        self.bias = self.bias - self.learning_rate * db

        return self.weights, self.bias


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()

    def _tanh_deriv(self, z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(z) ** 2

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def backward(self, Z1: np.ndarray, dZ2: np.ndarray, W2: np.ndarray) -> np.ndarray:
        return W2.T.dot(dZ2) * self._tanh_deriv(Z1)

    def backward_last_layer(self, A2: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return A2 - Loss.get_perfect_clasification(Y)


class Loss:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_perfect_clasification(Y: np.ndarray) -> np.ndarray:
        perfect_Y = np.zeros((Y.size, Y.max() + 1))
        perfect_Y[np.arange(Y.size), Y] = 1
        perfect_Y = perfect_Y.T
        return perfect_Y

    def loss(self, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (1 / (2 * m)) * np.sum((A - self.get_perfect_clasification(Y))**2)

    def mse_cost_derivative_bias(self, dZ: np.ndarray) -> float:
        return 1 / m * np.sum(dZ)

    def mse_cost_derivative_weights(self, dZ: np.ndarray, A: np.ndarray):
        return 1 / m * dZ.dot(A.T)


class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        pass

    def get_predictions(self, A2: np.ndarray):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions: np.ndarray, Y: np.ndarray) -> float:
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              epochs: int,
              *, verbose: bool = True) -> tuple[np.ndarray, float, np.ndarray, float]:
        for layer in layers:
            layer.learning_rate = self.learning_rate

        layer1: FullyConnected = self.layers[0]
        layer2: FullyConnected = self.layers[1]
        weights1, b1, weights2, b2 = [None] * 4
        for i in range(epochs):
            A1, Z1 = layer1.forward(x_train)
            A2, Z2 = layer2.forward(A1)
            dW2, db2, dZ2 = layer2.backward_last_layer(A1, A2, y_train)
            dW1, db1, _ = layer1.backward(x_train, Z1, dZ2, layer2.weights)

            weights1, b1 = layer1.update_with_gradient_descent_step(dW1, db1)
            weights2, b2 = layer2.update_with_gradient_descent_step(dW2, db2)

            if verbose and i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(A2)
                print(self.get_accuracy(predictions, y_train))

        return weights1, b1, weights2, b2


if __name__ == "__main__":
    layer1 = FullyConnected(64, 10)
    layer2 = FullyConnected(10, 10)
    layers = [layer1, layer2]

    network = Network(layers, learning_rate=0.1)
    network.train(pixels_train, numbers_train, 1000, verbose=True)
