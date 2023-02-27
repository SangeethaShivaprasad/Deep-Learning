import numpy as np
import copy
from  Layers.Base  import *
class NeuralNetwork():

    def __init__(self,optimizer,weights_initializer,bias_initializer):
        self.loss=[]
        self.layers=[]
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer

    def append_trainable_layer(self,layer):
        layer.initialize(self.weights_initializer,self.bias_initializer)
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        self.layers.append(layer)


    def forward(self):
        input_tensor,self.label_tensor = self.data_layer.forward()
        iter_input_tensor = input_tensor
        regularization_loss = 0.0
        for x in self.layers:
            iter_input_tensor = x.forward(iter_input_tensor)

            if hasattr(x, 'optimizer'):
                if (x.optimizer != None):
                    regularization_loss += x.optimizer.regularization_loss

        net_loss = self.loss_layer.forward(iter_input_tensor, self.label_tensor) + regularization_loss

        self.loss.append(net_loss)
        copy = np.copy(net_loss)
        return (copy)

    def backward(self):
        pass_back_tensor = self.loss_layer.backward(self.label_tensor)
        for j in reversed(self.layers):
            pass_back_tensor = j.backward(pass_back_tensor)


    def train(self,iterations):
        for i in range(iterations):
            self.forward()
            self.backward()
        self.set_phase(Phase.train)

		
    def test(self,input_tensor):
        self.set_phase(Phase.test)
        for x in self.layers:
            input_tensor = x.forward(input_tensor)
        probabilities = self.loss_layer.predict(input_tensor)
        copy = np.copy(probabilities)
        return (copy)

    def set_phase(self, phase):
        for layer in self.layers:
            layer.phase = phase
        self.loss_layer.phase = phase
        return

