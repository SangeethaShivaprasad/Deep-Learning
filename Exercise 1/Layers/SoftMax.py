import numpy as np
class SoftMax :
    def __init__(self):
        self.savepositive = None

    def forward(self,input_tensor, label_tensor):
        self.savepositive = self.predict(input_tensor)
        #mult_label_tensor = self.savepositive *label_tensor
        mult_label_tensor = -1 * np.log(self.savepositive)
        mult_label_tensor = mult_label_tensor * label_tensor
        mult_label_tensor = np.sum(mult_label_tensor)
        copy = np.copy(mult_label_tensor)
        return (copy)

    def backward(self, label_tensor):
        return_vect =  ( self.savepositive - label_tensor)
        copy = np.copy(return_vect)
        return (copy)


    def predict(self,input_tensor):
        input_tensor_shift = input_tensor - np.amax(input_tensor)
        exp_input = np.exp(input_tensor_shift)
        row_input = np.sum(exp_input, axis=1)
        predict_output = exp_input/row_input[:,None]
        copy = np.copy(predict_output)
        return(copy)


# input_tensor = np.zeros([3,2]) + 1
# label_tensor = np.array([[0,1], [1, 0], [0, 1]])
#
# c = SoftMax()
# c.forward(input_tensor,label_tensor)