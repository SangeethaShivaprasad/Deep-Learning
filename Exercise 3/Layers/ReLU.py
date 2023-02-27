import numpy as np

class ReLU :

    def __init__(self ):
        self.savepositive = None
        self.phase = None


    def  forward(self,input_tensor):
        self.savepositive = input_tensor
        return_tensor   = np.where(input_tensor > 0, input_tensor,0)
        copy = np.copy(return_tensor)
        return (copy)

    def backward(self, error_tensor):
        temp_zero = np.where( self.savepositive > 0, self.savepositive, 0)
        temp_one =  np.where( temp_zero <= 0, temp_zero, 1)
        return_tensor = np.multiply(error_tensor,temp_one)
        copy = np.copy(return_tensor)
        return (copy)

# input = np.matrix([[1.0, 0.02], [-1.0, 0.7837747]])
# output = np.matrix([[-10.38293, 20.36273], [30.3232, 40.23283]])
# relu = ReLU()
# relu.forward(input)
# relu.backward(output)