import numpy as np

class Flatten():
    def __init__ (self):
        self.shape = None
        return

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return  (np.reshape(input_tensor,(self.shape[0],self.shape[1]*self.shape[2]*self.shape[3])))



    def backward(self,error_tensor):
        return (np.reshape(error_tensor, (self.shape[0],self.shape[1],self.shape[2],self.shape[3])))
