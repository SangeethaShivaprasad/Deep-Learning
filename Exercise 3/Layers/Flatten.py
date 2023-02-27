import numpy as np

class Flatten():
    def __init__ (self):
        self.shape = None
        self.phase = None
        return

    def forward(self, input_tensor):
        self.shape = input_tensor.shape[0]
        self.input_shape = input_tensor[0].shape
        output_tensor = input_tensor.reshape(self.shape, np.prod(self.input_shape))
        return  (output_tensor)



    def backward(self,error_tensor):
        return (np.reshape(error_tensor, (self.shape, *self.input_shape )))
