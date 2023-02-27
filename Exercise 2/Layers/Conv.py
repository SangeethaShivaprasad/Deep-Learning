import numpy as np
from scipy import signal
from copy import deepcopy
#import pdb
class Conv():

	def __init__(self,stride_shape, convolution_shape, num_kernels):
		self.stride_shape = stride_shape
		self.convolution_shape = convolution_shape
		self.num_kernels = num_kernels
		self.stride_length = len(self.stride_shape)

		if (self.stride_length == 1):
			self.stride_shape = (1, self.stride_shape[0])
			self.convolution_shape = (self.convolution_shape[0],1,self.convolution_shape[1])

		#self.padding_dimension = int(convolution_shape[0]/2)
		self.input_kernel = np.zeros((self.num_kernels,self.convolution_shape[0],self.convolution_shape[1],self.convolution_shape[2]))
		for i in range(self.num_kernels):
			self.input_kernel[i,:] =  np.random.rand(self.convolution_shape[0],self.convolution_shape[1],self.convolution_shape[2])

		self.weights = np.array(self.input_kernel)

		self.bias = np.random.rand(self.num_kernels, 1)
		self.bias_gradient = np.zeros((self.num_kernels, 1))
		self.input_tensor_sizex = 0
		self.input_tensor_sizey = 0
		self.gradient = None
		self.input_tensor = None
		self.optimizer = None
		self.bias_optimizer = None
		return

	def forward(self, input_tensor):
		counteri = 0
		counterj = 0

		if (self.stride_length == 1):
			input_tensor = np.expand_dims(input_tensor, 2)
		self.input_tensor = input_tensor
		self.input_tensor_sizex = input_tensor.shape[2]
		self.input_tensor_sizey = input_tensor.shape[3]
		output_stride_x = int((input_tensor.shape[2] - 1)/self.stride_shape[0]+ 1)
		output_stride_y = int((input_tensor.shape[3] - 1)/(self.stride_shape[1])+ 1)
		output_batch_tensor = np.zeros((input_tensor.shape[0],self.num_kernels,output_stride_x,output_stride_y))
		output_image_tensor = np.zeros((self.num_kernels,output_stride_x,output_stride_y))
		channel_correl_tensor_strided = np.zeros((output_stride_x,output_stride_y))
		channel_correl_tensor_strided_sum = np.zeros((output_stride_x, output_stride_y))
		for inti in range(self.num_kernels):
			self.input_kernel[inti, :]  = self.weights[inti,:]

		#input_tensor = np.pad(input_tensor,(self.padding_dimension,self.padding_dimension), 'constant', constant_values=(0))
		for image_idx in range(input_tensor.shape[0]):
			for kernel_idx in range(self.input_kernel.shape[0]):
				for channel_idx in range(self.input_kernel.shape[1]):
					channel_correl_tensor = signal.correlate2d(input_tensor[image_idx,channel_idx,:,:], self.input_kernel[kernel_idx,channel_idx,:,:],mode='same')

					for i in range(0,channel_correl_tensor.shape[0],self.stride_shape[0]):
						for j in range(0,channel_correl_tensor.shape[1],self.stride_shape[1]):
							channel_correl_tensor_strided[counteri][counterj] = channel_correl_tensor[i][j]
							counterj = counterj + 1
						counterj = 0
						counteri = counteri + 1
					counteri = 0
					channel_correl_tensor_strided_sum = channel_correl_tensor_strided_sum + channel_correl_tensor_strided
				channel_correl_tensor_strided_sum =  channel_correl_tensor_strided_sum + self.bias[kernel_idx]
				output_image_tensor[kernel_idx,:,:] = channel_correl_tensor_strided_sum
				channel_correl_tensor_strided_sum = np.zeros((output_stride_x, output_stride_y))
			output_batch_tensor[image_idx,:,:,:] = output_image_tensor
		if (self.stride_length == 1):
			output_batch_tensor = np.squeeze(output_batch_tensor, axis=2)
		copy = np.copy(output_batch_tensor)
		return (copy)

	def backward(self, error_tensor):
		if (self.stride_length == 1):
			error_tensor = np.expand_dims(error_tensor, 2)
		output_stride_x = ((np.shape(error_tensor)[2]*(self.stride_shape[0])))
		output_stride_y = ((np.shape(error_tensor)[3]*(self.stride_shape[1])))
		error_tensor_with_stride =  np.zeros((np.shape(error_tensor)[0],np.shape(error_tensor)[1],self.input_tensor_sizex ,self.input_tensor_sizey))
		stride_index_x = 0
		stride_index_y = 0
		for channel_index in range(np.shape(error_tensor)[0]):
			for kernel_index in range(np.shape(error_tensor)[1]):
				for i in range(np.shape(error_tensor)[2]):
					stride_index_x = i * self.stride_shape[0]
					for j in range(np.shape(error_tensor)[3]):
						stride_index_y = j * self.stride_shape[1]
						error_tensor_with_stride[channel_index,kernel_index,stride_index_x,stride_index_y] = error_tensor[channel_index,kernel_index,i,j]
					stride_index_y = 0
				stride_index_x = 0

		backward_copy_bias_gradient = np.zeros((self.num_kernels, 1))
		channel_kernel = np.zeros((self.convolution_shape[0],self.num_kernels,  self.convolution_shape[1], self.convolution_shape[2]))
		error_tensor_prev_layer = np.zeros((np.shape(error_tensor_with_stride)[0],self.convolution_shape[0],np.shape(error_tensor_with_stride)[2],np.shape(error_tensor_with_stride)[3]))
		kernel_correl_tensor = np.zeros((self.convolution_shape[1],self.convolution_shape[2]))
		kernel_correl_tensor_sum = np.zeros((np.shape(error_tensor_with_stride)[2],np.shape(error_tensor_with_stride)[3]))
		bias_gradient_image = np.zeros((np.shape(error_tensor_with_stride)[0],self.num_kernels,self.num_kernels))

		for channel_idx in range(self.convolution_shape[0]):
			for kernel_idx in range(self.num_kernels):
				channel_kernel[channel_idx,kernel_idx,:] = self.input_kernel[kernel_idx,channel_idx,:]

		for image_idx in range(np.shape(error_tensor_with_stride)[0]):
			for channel_idx in range(self.convolution_shape[0]):
				for kernel_idx in range(self.num_kernels):
					kernel_correl_tensor = signal.correlate2d(error_tensor_with_stride[image_idx,kernel_idx, :, :],np.rot90(channel_kernel[channel_idx,kernel_idx, :, :],2),mode='same')
					kernel_correl_tensor_sum = kernel_correl_tensor_sum + kernel_correl_tensor

				error_tensor_prev_layer[image_idx,channel_idx,:] = kernel_correl_tensor_sum
				kernel_correl_tensor_sum = np.zeros((np.shape(error_tensor_with_stride)[2],np.shape(error_tensor_with_stride)[3]))




		padx1 = int(np.floor(self.convolution_shape[1]/2))
		pady1 = int(np.floor(self.convolution_shape[2]/2.0 ))
		padx2 = int(np.floor(self.convolution_shape[1] / 2.0 - 0.5))
		pady2 = int(np.floor(self.convolution_shape[2] / 2.0 - 0.5))
		gradient_tensor =  np.zeros((np.shape(error_tensor_with_stride)[1],self.convolution_shape[0],self.convolution_shape[1],self.convolution_shape[2]))
		gradient_tensor_batch = np.zeros((np.shape(error_tensor_with_stride)[1], self.convolution_shape[0], self.convolution_shape[1],self.convolution_shape[2]))
		input_tensor_padded = np.zeros((np.shape(self.input_tensor)[0],np.shape(self.input_tensor)[1],np.shape(self.input_tensor)[2] + padx1 + padx2 ,np.shape(self.input_tensor)[3] + pady1 + pady2))
		for image_idx in range(np.shape(error_tensor_prev_layer)[0]):

			for channel_idx in range(np.shape(error_tensor_prev_layer)[1]):
				input_tensor_padded[image_idx,channel_idx,:] = np.array(np.pad(self.input_tensor[image_idx,channel_idx,:,:],((padx1, padx2), (pady1, pady2)), 'constant'))




		for image_idx in range(np.shape(input_tensor_padded)[0]):

			for kernel_idx in range(np.shape(error_tensor_with_stride)[1]):

				backward_copy_bias_gradient[kernel_idx] += np.sum(error_tensor_with_stride[image_idx, kernel_idx, :])


				for channel_idx in range(np.shape(input_tensor_padded)[1]):
					channel_mult_tensor = signal.correlate2d(input_tensor_padded[image_idx, channel_idx, :, :],error_tensor_with_stride[image_idx, kernel_idx, :, :],mode='valid')
					gradient_tensor[kernel_idx,channel_idx,:] = channel_mult_tensor
			gradient_tensor_batch += gradient_tensor
		self.gradient = np.copy(gradient_tensor_batch)


		if (self.stride_length == 1):
			error_tensor_prev_layer = np.squeeze(error_tensor_prev_layer, axis=2)
		self.bias_gradient = np.copy(backward_copy_bias_gradient)

		if self.optimizer != None:
			updated_weights = self.optimizer.calculate_update(self.input_kernel,self.gradient)

			self.weights = updated_weights
			self.input_kernel = self.weights

		if self.bias_optimizer != None:
			self.bias = self.bias_optimizer.calculate_update(self.bias, self.bias_gradient)




		copy = np.copy(error_tensor_prev_layer)
		return (copy)


	def get_gradient_weights(self):
		return self.gradient

	def get_gradient_bias(self):
		return self.bias_gradient


	def set_optimizer(self, optimizer):
		self.optimizer = deepcopy(optimizer)
		self.bias_optimizer = deepcopy(optimizer)

	def initialize(self, weights_initializer, bias_initializer):
		fan_in = np.prod(self.convolution_shape)
		fan_out = self.num_kernels * self.convolution_shape[-2] * self.convolution_shape[-1]
		self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)

		bias_initialized = bias_initializer.initialize(self.bias.shape, self.bias.shape[0], self.bias.shape[1])
		self.bias = bias_initialized.reshape(len(bias_initialized), 1)

		return