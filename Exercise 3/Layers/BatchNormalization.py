
import numpy as np
from Layers.Base import *
from Layers.Helpers import *
from copy import deepcopy
class BatchNormalization():

	def __init__(self,channels=0):
		self.bias = 0.0
		self.weights = 1.0
		self.channels = channels
		self.input_shape = None
		self.alpha = 0.8
		self.mean = 0
		self.variance = 0
		self.batch_mean_prev = 0
		self.batch_std_prev = 0
		self.phase = None
		self.delta = 1.0
		return

	def forward(self,input_tensor):

		if(self.phase==None or self.phase==Phase.train):

			if(self.channels > 0):
				self.input_shape = input_tensor.shape
				input_tensor = self.reformat(input_tensor)
			self.input_tensor = input_tensor
			self.batch_mean = np.mean(input_tensor,axis = 0)
			self.batch_std  = np.std(input_tensor,axis = 0)
			if isinstance(self.weights, float):
				self.weights = np.ones(self.batch_mean.shape)
				self.bias = np.zeros(self.batch_mean.shape)
			self.normalized_input = (input_tensor - self.batch_mean)/(np.sqrt(self.batch_std * self.batch_std))
			normalized_input_bias_weight = (self.weights*self.normalized_input) + self.bias
			if (self.channels > 0):
				normalized_input_bias_weight = self.reformat(normalized_input_bias_weight)

			self.mean = (1 - self.alpha)*self.batch_mean + self.alpha*self.batch_mean_prev
			self.variance = (1 - self.alpha)*self.batch_std+ self.alpha*self.batch_std_prev

			self.batch_mean_prev = self.batch_mean
			self.batch_std_prev = self.batch_std
			return normalized_input_bias_weight
		else:
			if (self.channels > 0):
				self.input_shape = input_tensor.shape
				input_tensor = self.reformat(input_tensor)
			self.input_tensor = input_tensor
			self.normalized_input = (input_tensor - self.mean) / (np.sqrt(self.variance  * self.variance ))
			normalized_input_bias_weight = (self.weights * self.normalized_input) + self.bias
			if (self.channels > 0):
				normalized_input_bias_weight = self.reformat(normalized_input_bias_weight)
			return normalized_input_bias_weight


	def reformat(self,input_tensor):
		shape = input_tensor.shape
		if(len(shape) == 4):
			output_tensor = np.reshape(input_tensor,(input_tensor.shape[0],input_tensor.shape[1],(input_tensor.shape[2]*input_tensor.shape[3])))
			output_tensor = np.transpose(output_tensor,(0,2,1))
			output_tensor = np.reshape(output_tensor,((output_tensor.shape[0]*output_tensor.shape[1]),output_tensor.shape[2]))
		else:
			output_tensor = np.reshape(input_tensor,(self.input_shape[0],(self.input_shape[2]*self.input_shape[3]),self.input_shape[1]))
			output_tensor = np.transpose(output_tensor, (0, 2, 1))
			output_tensor = np.reshape(output_tensor,(self.input_shape[0],self.input_shape[1],self.input_shape[2],self.input_shape[3]))
		return output_tensor

	def backward(self, error_tensor):
		if (self.channels > 0):
			#self.input_shape = error_tensor.shape
			error_tensor = self.reformat(error_tensor)
		self.grad_weights = error_tensor * self.normalized_input
		self.grad_weights =  np.sum(self.grad_weights , axis=0)
		self.grad_bias =  np.sum(error_tensor, axis=0)
		grad_input =  compute_bn_gradients(error_tensor,self.input_tensor,self.weights,self.batch_mean,np.square(self.batch_std))
		if hasattr(self, 'optimizer'):
			self.weights = self.optimizer.calculate_update(self.weights,self.grad_weights)
		if hasattr(self, 'bias_optimizer'):
			self.bias = self.bias_optimizer.calculate_update(self.bias,self.grad_bias)
		if (self.channels > 0):
			grad_input = self.reformat(grad_input)
		return grad_input

	def set_optimizer(self, optimizer):

		self.optimizer = deepcopy(optimizer)
		self.bias_optimizer = deepcopy(optimizer)

	def get_gradient_weights(self):
		return self.grad_weights

	def get_gradient_bias(self):
		return self.grad_bias

	def get_weights(self):
		return self.weights

	def set_weights(self, weights):
		self.weights = weights

	def initialize(self,weights_initializer, bias_initializer):
		return