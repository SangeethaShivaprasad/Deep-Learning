import numpy as np
class NeuralNetwork():

    def __init__(self,weights_initializer,bias_initializer):
        self.loss=[]
        self.layers=[]
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def append_trainable_layer(self,layer):
        layer.initialize(self.weights_initializer,self.bias_initializer)


    def forward(self):
        input_tensor,self.label_tensor = self.data_layer.forward()

        for x in self.layers:

            input_tensor = x.forward(input_tensor)
        net_loss = self.loss_layer.forward(input_tensor, self.label_tensor)
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

		
    def test(self,input_tensor):
        for x in self.layers:
            input_tensor = x.forward(input_tensor)
        probabilities = self.loss_layer.predict(input_tensor)
        copy = np.copy(probabilities)
        return (copy)

