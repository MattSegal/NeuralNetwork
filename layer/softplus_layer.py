from layer import Layer
import numpy as np

class SoftplusLayer(Layer):
    def activation(self, z):
        """
        Softplus function is a smooth approximation of a
        rectified linear unit
        """
        return np.log(1 + np.exp(z))

    def activation_deriv(self):
        """
        Softplus derivative is the logistic function.
        That's kind of cool.
        """
        assert self.output.size > 0
        return 1.0 / (1.0 + np.exp(-self.linear_output))
