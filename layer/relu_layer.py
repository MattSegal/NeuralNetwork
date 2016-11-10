from layer import Layer
import numpy as np

class ReluLayer(Layer):
    """
    Rectified linear units.
    Draw lines through Thingspace
    """
    def activation(self, z):
        """
        x if x > 0 otherwise 0
        """
        z[z < 0] = 0
        return z

    def activation_deriv(self):
        """
        Relu derivative is Heaviside function.
        """
        assert self.output.size > 0
        shape = self.linear_output.shape
        mask = self.linear_output > 0
        deriv = np.zeros(shape)
        deriv[mask] = self.linear_output[mask]
        return deriv