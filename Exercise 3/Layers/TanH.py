import numpy as np
class TanH:
    def __init__(self):
        return

    def forward(self, input_tensor):
        self.tanh_activation = np.tanh(input_tensor)
        copy = np.copy(self.tanh_activation)
        return copy

    def backward(self, error_tensor):
        tanh_derivative = 1 - (self.tanh_activation * self.tanh_activation)
        copy = np.copy(error_tensor * tanh_derivative)
        return copy