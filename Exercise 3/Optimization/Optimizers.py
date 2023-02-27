import numpy as np
from  Optimization.Constraints  import *

class OptimizerBase():

	def __init__(self):
		self.regularizer = None
		self.regularization_loss = 0.0

	def add_regularizer(self, regularizer):
		self.regularizer = regularizer

class Sgd(OptimizerBase):

	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
		self.phase = None
		super().__init__()
		return

	def calculate_update(self, weight_tensor, gradient_tensor):

		if(self.regularizer != None):
			self.regularization_loss = self.regularizer.alpha * self.regularizer.norm(weight_tensor)
			update_weights = weight_tensor - (self.learning_rate * self.regularizer.calculate(weight_tensor)) - (self.learning_rate * gradient_tensor)
		else:
			update_weights = weight_tensor - (self.learning_rate * gradient_tensor)
		return update_weights


class SgdWithMomentum(OptimizerBase):

	def __init__(self,learning_rate,momentum = 0.9):
		self.momentum_rate = momentum
		self.learning_rate = learning_rate
		#self.momentum_rate = momentum_rate
		self.intermediate_term = 0.0
		self.phase = None
		super().__init__()
		return

	def calculate_update(self, weight_tensor, gradient_tensor):
		if (self.regularizer != None):
			self.regularization_loss = self.regularizer.alpha * self.regularizer.norm(weight_tensor)
			self.intermediate_term = (self.momentum_rate * 	self.intermediate_term)  - ((self.learning_rate*(gradient_tensor)))
			#self.previous_momentum = intermediate_term
			update_weights =  weight_tensor + self.intermediate_term - (self.learning_rate*(self.regularizer.calculate(weight_tensor)))
		else:
			self.intermediate_term = (self.momentum_rate * self.intermediate_term ) - (self.learning_rate  * gradient_tensor)
			#self.previous_momentum = intermediate_term
			update_weights = weight_tensor + self.intermediate_term

		return update_weights


class Adam(OptimizerBase):

	def __init__(self,learning_rate,beta1 = 0.9,beta2 = 0.999):
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		self.prev_single_gradient = 0.0
		self.prev_double_gradient = 0.0
		self.k = 0.0
		self.eps = 1e-6
		self.phase = None
		super().__init__()
		return

	def calculate_update(self, weight_tensor, gradient_tensor):
		self.k += 1.0
		if (self.regularizer != None):
			self.regularization_loss = self.regularizer.alpha * self.regularizer.norm(weight_tensor)

			single_gradient = (self.beta1 * self.prev_single_gradient) + ((1.0 - self.beta1) * gradient_tensor)
			self.prev_single_gradient = single_gradient
			double_gradient = (self.beta2 * self.prev_double_gradient) + ((1.0 - self.beta2) * gradient_tensor * gradient_tensor)
			self.prev_double_gradient = double_gradient
			single_gradient = single_gradient / (1.0 - (self.beta1 ** self.k))
			double_gradient = double_gradient / (1.0 - (self.beta2 ** self.k))

			update_weights = weight_tensor - (self.learning_rate * self.regularizer.calculate(weight_tensor)) - (self.learning_rate * ((single_gradient + self.eps) / ((np.sqrt(double_gradient)) + self.eps)))
		else:
			single_gradient = (self.beta1 * self.prev_single_gradient) + ((1.0 - self.beta1) * gradient_tensor)
			self.prev_single_gradient = single_gradient
			double_gradient = (self.beta2 * self.prev_double_gradient) + ((1.0 - self.beta2) * gradient_tensor * gradient_tensor)
			self.prev_double_gradient = double_gradient
			single_gradient = single_gradient / (1.0 - (self.beta1 ** self.k))
			double_gradient = double_gradient / (1.0 - (self.beta2 ** self.k))
			update_weights = weight_tensor - (self.learning_rate * ((single_gradient + self.eps) / ((np.sqrt(double_gradient)) + self.eps)))
		return update_weights

