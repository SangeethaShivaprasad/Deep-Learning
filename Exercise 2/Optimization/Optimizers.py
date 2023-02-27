import numpy as np
class Sgd():

	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
		return

	def calculate_update(self, weight_tensor, gradient_tensor):
		update_weights = weight_tensor - (self.learning_rate  * gradient_tensor)
		return update_weights


class SgdWithMomentum():

	def __init__(self,learning_rate,momentum_rate):
		self.learning_rate = learning_rate
		self.momentum_rate = momentum_rate
		self.previous_momentum = 0.0
		return

	def calculate_update(self, weight_tensor, gradient_tensor):
		intermediate_term = (self.momentum_rate * self.previous_momentum) - (self.learning_rate  * gradient_tensor)
		self.previous_momentum = intermediate_term
		update_weights = weight_tensor + intermediate_term
		return update_weights


class Adam():

	def __init__(self,learning_rate,beta1,beta2):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.prev_single_gradient = 0.0
		self.prev_double_gradient = 0.0
		self.k = 0.0
		self.eps = 1e-6
		return

	def calculate_update(self, weight_tensor, gradient_tensor):
		self.k += 1.0
		single_gradient = (self.beta1 * self.prev_single_gradient) + ((1.0 - self.beta1) * gradient_tensor)
		self.prev_single_gradient = single_gradient
		double_gradient = (self.beta2 * self.prev_double_gradient) + ((1.0 - self.beta2) * gradient_tensor * gradient_tensor)
		self.prev_double_gradient = double_gradient
		single_gradient = single_gradient / (1.0 - (self.beta1 ** self.k))
		double_gradient = double_gradient / (1.0 - (self.beta2 ** self.k))
		update_weights = weight_tensor - (self.learning_rate * ((single_gradient + self.eps) / ((np.sqrt(double_gradient)) + self.eps)))
		return update_weights