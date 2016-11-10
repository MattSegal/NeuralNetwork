from layer import Layer
import numpy as np

class SoftmaxLayer(Layer):
    """
    Outputs a set of probabilities which sum to 1.
    Softmax uses cross entropy for its cost function, rather than mean squared error. 
    """

    def activation(self, z):
        """
        Softmax is exp(z_i) / sum of exp(z)
        """
        # Subtract max from each example 
        # to prevent overflow when using exp function
        z = (z.T - z.max(1)).T
        z = np.exp(z)
        # Normalize by sum for each example
        z = (z.T / (z.sum(axis=1)*1.0)).T
        return z

    def activation_deriv(self):
        """
        This derivative assumes that number of inputs == number outputs
        """
        return self.output * (1 - self.output)

    def get_error(self, target):
        """
        Calculates cross entropy cost of output state given a target
        """
        assert self.output.shape == target.shape
        return np.sum( -1 * target * np.log(self.output) ) / float(self.output.shape[0])

    def get_error_deriv(self, target):
        """
        Calculates cross entropy cost derivative of output layer given a target
        This assumes that this layer is the final layer in the network.
        """
        assert self.output.shape == target.shape
        return self.output - target
