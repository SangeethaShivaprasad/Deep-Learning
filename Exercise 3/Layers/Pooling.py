import numpy as np
import pdb
class Pooling():

	def __init__(self,stride_shape,pooling_shape):
		self.pooling_shape = pooling_shape
		self.stride_shape = stride_shape
		self.max_positx = None
		self.max_posity = None
		self.input_tensor = None
		self.phase = None
		return

	def forward(self,input_tensor):
		self.input_tensor = input_tensor
		pool_tensor = np.zeros((self.pooling_shape[0]*self.pooling_shape[1] , 1))
		onedimage =	np.zeros((input_tensor.shape[0],input_tensor.shape[1]))
		pool_mov_x = (input_tensor.shape[2] - self.pooling_shape[0])+ 1
		pool_mov_y = (input_tensor.shape[3] - self.pooling_shape[1])+ 1
		pool_mov_x_stride = int(np.ceil(pool_mov_x / self.stride_shape[0]))
		pool_mov_y_stride = int(np.ceil(pool_mov_y / self.stride_shape[1]))
		output_tensor = np.zeros((input_tensor.shape[0],input_tensor.shape[1],pool_mov_x_stride,pool_mov_y_stride))
		self.max_positx = np.zeros((input_tensor.shape[0],input_tensor.shape[1],(pool_mov_x_stride * pool_mov_y_stride)))
		self.max_posity = np.zeros((input_tensor.shape[0],input_tensor.shape[1],(pool_mov_x_stride * pool_mov_y_stride)))
		iter = 0
		outer_iter = 0
		save_loc_x = 0
		save_loc_y = 0
		temp_mov_x = 0
		temp_mov_y = 0
		temp_result = []
		for image_idx in range(input_tensor.shape[0]):
			for channel_idx in range(input_tensor.shape[1]):
				onedimage = input_tensor[image_idx,channel_idx,:]
				for mov_y in range(0,(pool_mov_y),(self.stride_shape[1])):
					for mov_x in range(0,(pool_mov_x),(self.stride_shape[0])):
						for pool_id_y in range(self.pooling_shape[1]):
							for pool_id_x in range(self.pooling_shape[0]):
								pool_tensor[iter] = onedimage[mov_x + pool_id_x,mov_y + pool_id_y]
								iter = iter + 1

						maxElement = np.max(pool_tensor)
						element_argmax = np.argmax(pool_tensor)
						result = np.where(maxElement == np.max(pool_tensor,axis= 1))
						save_loc_x = int(((element_argmax)%int(self.pooling_shape[1])))
						save_loc_y = int(((element_argmax)//int(self.pooling_shape[0])))
                      #  if(result.size() == 3)

						# if(element_argmax == 0):
						# 	save_loc_x = 0
						# 	save_loc_y = 0
						# if (element_argmax == 1):
						# 	save_loc_x = 1
						# 	save_loc_y = 0
						# if (element_argmax == 2):
						# 	save_loc_x = 0
						# 	save_loc_y = 1
						# if (element_argmax == 3):
						# 	save_loc_x = 1
						# 	save_loc_y = 1
					#	save_loc_x = save_loc_x +  mov_x
					#	save_loc_y = save_loc_y +  mov_y

						if(mov_x != 0):
							mov_x_out = int(np.ceil(mov_x / self.stride_shape[0]))
						else:
							mov_x_out = 0
						if(mov_y != 0):
							mov_y_out = int(np.ceil(mov_y/ self.stride_shape[1]))
						else:
							mov_y_out = 0

						self.max_positx[image_idx,channel_idx,outer_iter] = save_loc_x
						self.max_posity[image_idx,channel_idx,outer_iter] = save_loc_y
						outer_iter = outer_iter + 1
						output_tensor[image_idx,channel_idx,mov_x_out,mov_y_out] = maxElement
						iter = 0
				outer_iter = 0
		return	output_tensor

	def backward(self, error_tensor):
		error_tensor_prev_layer = np.zeros((self.input_tensor.shape[0],self.input_tensor.shape[1],self.input_tensor.shape[2],self.input_tensor.shape[3] ))
		extra_tensor = []
		x_stack = []
		prev_layer_1d_image_stack = []
		prev_layer_1d_image = []
		onedimage = None
		stride_x_cond = self.stride_shape[0] - 1
		stride_y_cond = self.stride_shape[1] - 1
		if(stride_x_cond > 1):
			stride_x_tensor = np.zeros((self.stride_shape[0] - 2,self.pooling_shape[1]))
		if (stride_y_cond > 1):
			stride_y_tensor = np.zeros((self.stride_shape[1] - 2, self.pooling_shape[0]))
		pool_tensor = []
		iter = 0

		for image_idx in range(error_tensor.shape[0]):
			for channel_idx in range(error_tensor.shape[1]):
				onedimage = error_tensor[image_idx,channel_idx,:]
				for iter_y in range(error_tensor.shape[3]):
					for iter_x in range(error_tensor.shape[2]):
						pool_tensor = np.zeros((self.pooling_shape[0], self.pooling_shape[1]))
						pool_tensor[int(self.max_positx[image_idx,channel_idx,iter]),int(self.max_posity[image_idx,channel_idx,iter])] = onedimage[iter_x,iter_y]

						iter = iter + 1
						if (iter_x == 0) :
							x_stack = pool_tensor
							if (stride_x_cond > 1):
								x_stack = np.concatenate((x_stack, stride_x_tensor), axis=0)
						else:
							x_stack = np.concatenate((x_stack, pool_tensor), axis=0)
							if((iter_x < (error_tensor.shape[2] - 1) and (stride_x_cond > 1))):
								x_stack = np.concatenate((x_stack, stride_x_tensor), axis=0)

					prev_layer_1d_image_stack.append(x_stack)
				iter = 0
				for iter_list in range(len(prev_layer_1d_image_stack)):
					if (iter_list == 0):
						prev_layer_1d_image = prev_layer_1d_image_stack[iter_list]
						if (stride_y_cond > 1):
							prev_layer_1d_image = np.concatenate((prev_layer_1d_image_stack[iter_list], stride_y_tensor), axis=1)
					else:
						prev_layer_1d_image = np.concatenate((prev_layer_1d_image,prev_layer_1d_image_stack[iter_list]),axis=1 )
						if (stride_y_cond > 1):
							prev_layer_1d_image = np.concatenate((prev_layer_1d_image_stack[iter_list], stride_y_tensor), axis=1)

				prev_layer_1d_image_new_x = np.zeros((self.input_tensor.shape[2],self.input_tensor.shape[3]))
				iter = 0
				if(stride_y_cond == 0):
					for iter_y in range(self.input_tensor.shape[3]):
						if(iter_y == 0):
							prev_layer_1d_image_new_x[:,iter_y] = prev_layer_1d_image[:,iter]
							iter = iter + 2
						else :
							if (iter_y == (self.input_tensor.shape[3] - 1)):
								prev_layer_1d_image_new_x[:, iter_y] = prev_layer_1d_image[:, (iter - 1)]
							else :
								prev_layer_1d_image_new_x[:,iter_y] = prev_layer_1d_image[:,iter]+prev_layer_1d_image[:,(iter - 1)]
								iter = iter + 2
					prev_layer_1d_image =  prev_layer_1d_image_new_x

				iter = 0
				if (stride_x_cond == 0):
					for iter_x in range(self.input_tensor.shape[2]):
						if (iter_x == 0):
							prev_layer_1d_image_new_x[iter_x,:] = prev_layer_1d_image[iter,:]
							iter = iter + 2
						else:
							if (iter_x == (self.input_tensor.shape[2] - 1)):
								prev_layer_1d_image_new_x[iter_x,:] = prev_layer_1d_image[(iter - 1),:]
							else:
								prev_layer_1d_image_new_x[iter_x,:] = prev_layer_1d_image[iter,:] + prev_layer_1d_image[(iter - 1),:]
								iter = iter + 2
					prev_layer_1d_image = prev_layer_1d_image_new_x




				if(prev_layer_1d_image.shape[1] != 	self.input_tensor.shape[3] ):
					extra_tensor = np.zeros((prev_layer_1d_image.shape[0],(self.input_tensor.shape[3] - prev_layer_1d_image.shape[1])) )
					prev_layer_1d_image = np.concatenate((prev_layer_1d_image,extra_tensor),axis=1 )
				if (prev_layer_1d_image.shape[0] != self.input_tensor.shape[2]):
					extra_tensor = np.zeros(((self.input_tensor.shape[2] - prev_layer_1d_image.shape[0]),prev_layer_1d_image.shape[1]))
					prev_layer_1d_image = np.concatenate((prev_layer_1d_image, extra_tensor), axis=0)
				prev_layer_1d_image_stack = []
				error_tensor_prev_layer[image_idx,channel_idx,:] = prev_layer_1d_image

		return error_tensor_prev_layer