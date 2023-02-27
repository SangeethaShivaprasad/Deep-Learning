import numpy as np
class FullyConnected :
    def __init__(self,input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.delta = 1
        self.weights = np.random.rand( self.input_size+1 , self.output_size )

    def  forward(self,input_tensor):
        bias = np.ones((input_tensor.shape[0],1), dtype=int)
        self.input_tensor_bias = np.concatenate((input_tensor,bias), axis=1)
      #  self.input_tensor_bias = np.transpose(self.input_tensor_bias)
      #  local_weights = np.transpose( self.weights)
        output_tensor = np.dot(self.input_tensor_bias,  self.weights)
        copy = np.copy(output_tensor)
        return (copy)

    def backward(self, error_tensor):
        local_weights = np.delete( self.weights,self.input_size , 0)
        error_tensor_return = np.dot(error_tensor, np.transpose(local_weights))
        self.gradient_weights = np.dot(np.transpose( self.input_tensor_bias),error_tensor)
        self.weights = self.weights  - (self.delta*self.gradient_weights)
        copy = np.copy(error_tensor_return)
        return (copy)

    def get_gradient_weights(self):
        return self.gradient_weights

