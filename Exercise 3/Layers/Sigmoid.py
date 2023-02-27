import numpy as np
class Sigmoid:
    def __init__(self):
        return

    def forward(self, input_tensor):
        neg_input = np.negative(input_tensor)
        self.sigmoid_activation = 1 / (1 + np.exp(neg_input))
        copy = np.copy(self.sigmoid_activation)
        return copy

    def backward(self, error_tensor):
        sigmoid_derivative = self.sigmoid_activation * (1 - self.sigmoid_activation)
        copy = np.copy(error_tensor * sigmoid_derivative)
        return copy