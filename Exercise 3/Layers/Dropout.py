import numpy as np
from  Layers.Base  import *
import random
class Dropout():

	def __init__(self,probability):
		self.probability = probability
		self.phase = None
		return


	def forward(self, input_tensor):
		if (self.phase == None or self.phase == Phase.train):
			self.dropout_mask = (np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.probability)
			return input_tensor * self.dropout_mask / self.probability

		elif (self.phase == Phase.test):
			return input_tensor

	def backward(self, error_tensor):
		return error_tensor * self.dropout_mask