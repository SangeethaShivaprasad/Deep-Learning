import numpy as np

class L2_Regularizer():
    def __init__(self, alpha):
        self.alpha = alpha
        self.regularization_loss = None
        return

    def calculate(self, weights):
        weight_shrink = (self.alpha)*weights
        return  weight_shrink

    def norm(self, weights):
        self.norm_weights = weights
        self.norm_weights =  self.norm_weights.reshape((np.prod(self.norm_weights.shape), 1))
        self.norm_weights = np.square(np.linalg.norm((self.norm_weights),2))
        copy = np.copy(self.alpha * self.norm_weights)
        #copy = np.copy(self.norm_weights)
        return copy


class L1_Regularizer():
    def __init__(self, alpha):
        self.alpha = alpha
        self.regularization_loss = None
        return

    def calculate(self, weights):
        weight_shrink =  (self.alpha * np.sign(weights))
        return weight_shrink

    def norm(self, weights):
        self.norm_weights = (self.alpha * weights)
        self.norm_weights =  self.norm_weights.reshape(((np.prod(self.norm_weights.shape)), 1))
       # self.regularization_loss = self.alpha * self.norm_weights
        copy = np.copy(np.linalg.norm((self.norm_weights), ord=1))
        return copy