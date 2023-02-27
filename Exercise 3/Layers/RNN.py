import numpy as np
from  Layers import TanH
from Layers import FullyConnected
import copy

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_state = np.zeros(self.hidden_size)
        self.output_size = output_size
        self.state = False
        self.bias = None
        self.h = []
        self.tanh_layer_fw = []
        self.input_term_fw = []
        self.h.append(np.zeros(self.hidden_size))
        self.fcl_input = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fcl_output = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.tanh = TanH.TanH()
        self.optimizer = None
        return

    def set_memorize(self, state):
        self.state = state
        return

    def forward(self, input_tensor):
        output_temp = []
        self.input_val = input_tensor
        if (self.state is False):
            self.h = []
            self.input_term_fw = []
            self.tanh_layer_fw = []
            h = np.zeros(self.hidden_size)
            self.h.append(h)
        else:
            tmp = self.h[-1]
            self.h = []
            self.h.append(tmp)
            self.input_term_fw = np.zeros(self.input_size)
            tmp = self.input_term_fw[-1]
            self.input_term_fw = []
            self.input_term_fw.append(tmp)

        for time_ref in range(input_tensor.shape[0]):
            input_term = np.expand_dims(np.append(input_tensor[time_ref], self.h[-1]), axis=0)
            # temp = np.copy(input_term)


            first_fcl = self.fcl_input.forward(input_term)
            self.input_term_fw.append(self.fcl_input.input_tensor_bias)

            tanh_layer = self.tanh.forward(first_fcl)
            self.tanh_layer_fw.append(tanh_layer)
            self.h.append(np.squeeze(tanh_layer))

            second_fcl = self.fcl_output.forward(tanh_layer)
            output_temp.append(np.squeeze(second_fcl))

        output = np.copy(output_temp)
        return output

    def backward(self, error_tensor):
        self.grad_out = np.zeros(self.fcl_output.weights.shape)
        self.grad_in = np.zeros(self.fcl_input.weights.shape)
        bw_h = [np.zeros(self.hidden_size)]
        del_x_total = np.zeros((error_tensor.shape[0], self.input_val.shape[1]))
        for time_ref in range(error_tensor.shape[0]):
            time_ref_bw = ((error_tensor.shape[0] - 1) - time_ref)
            self.fcl_output.input_tensor_bias = np.concatenate((self.tanh_layer_fw[time_ref_bw], np.ones((1, 1))),
                                                               axis=1)
            sec_fcl_bw = self.fcl_output.backward(np.expand_dims(error_tensor[time_ref_bw], axis=0))
            hidden_sum = sec_fcl_bw + np.expand_dims(bw_h[time_ref], axis=0)

            self.tanh.tanh_activation = self.tanh_layer_fw[time_ref_bw]
            tanh_layer_bw = self.tanh.backward(hidden_sum)

            self.fcl_input.input_tensor_bias = self.input_term_fw[time_ref_bw]
            first_fcl_bw = self.fcl_input.backward(tanh_layer_bw)

            del_x = first_fcl_bw[:, 0:self.input_size]
            del_h = first_fcl_bw[:, self.input_size:]
            bw_h.append(np.squeeze(del_h))
            del_x_total[time_ref_bw] = np.squeeze(del_x)
            self.grad_in += self.fcl_input.get_gradient_weights()
            self.grad_out += self.fcl_output.get_gradient_weights()

        if (self.optimizer is not None):
            self.fcl_input.weights = self.optimizer.calculate_update(self.fcl_input.weights, self.grad_in)
            self.fcl_output.weights = self.optimizer.calculate_update(self.fcl_output.weights, self.grad_out)
        output = np.copy(del_x_total)
        return output

    def set_optimizer(self, optimizer):
        self.optimizer = copy.deepcopy(optimizer)

    def calculate_regularization_loss(self):
        return

    def initialize(self, weights_initializer, bias_initializer):
        self.fcl_input.initialize(weights_initializer, bias_initializer)
        self.fcl_output.initialize(weights_initializer, bias_initializer)
        return

    def get_gradient_weights(self):
        return self.grad_in

    # def get_weights(self):
    #     return self.fcl_input.weights
    #
    # def set_weights(self, weight):
    #     self.fcl_input.weights = weight

    @property
    def weights(self):
        return self.fcl_input.weights

    @weights.setter
    def weights(self, weights):
        self.fcl_input.weights = weights