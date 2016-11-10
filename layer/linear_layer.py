from layer import Layer
import numpy as np

class LinearLayer(Layer):
    """
    Effectively no activation
    """
    def activation(self, z):
        return z

    def activation_deriv(self):
        return np.ones(self.output.shape)