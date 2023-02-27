import numpy as np
class Constant:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        return  np.ones(weights_shape[1],weights_shape[0])*0.1

class UniformRandom:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.rand(weights_shape[1],weights_shape[0])
        return weights


class Xavier:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2/(fan_in + fan_out))
        weights = np.random.normal(loc=0.0, scale=variance, size=weights_shape)
        return weights

class He:
    def __init__(self):
        return

    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2/(fan_in))
        weights = np.random.normal(loc=0.0, scale=variance, size=weights_shape)
        return weights