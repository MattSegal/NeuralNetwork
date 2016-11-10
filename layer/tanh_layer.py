from layer import Layer
import numpy as np

class TanhLayer(Layer):
    """
    If you use this guy as your final layer for classification,
    then your values should take the range of [-1,1]
    """
    def activation(self, z):
        """
        It's like sigmoid but with better gradients or something
        it's (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        use built-in tanh to prevent overflow
        """
        return np.tanh(z)

    def activation_deriv(self):
        """
        Tanh derivative is the 'secret sauce' in the optimization batter.
        The other secret sauce is bear semen - make gradient strong, like bear.
        """
        assert self.output.size > 0
        return 1 - self.output * self.output

