import numpy as np

class ParallelLayer:
    """
    This class wraps around a Layer object.
    Duplicates a Layer in parallel - used for embedding layer.
    This class allows us to keep using the Layer interface with more complex architectures.
    This layer can currently only be used as the first layer in a network
    """
    def __init__(self, layer, number_in_parallel):
        """
        Takes the layer object that is to be set up in parallel
        """
        self.number_in_parallel = number_in_parallel
        self.layer = layer
        self.num_inputs = layer.num_inputs * number_in_parallel
        self.num_outputs = layer.num_outputs * number_in_parallel

        # Output is shown to the outside
        self.output = np.zeros((1,self.num_outputs))

        # Inputs and outputs are used internally
        self.inputs = [np.zeros((1,layer.num_inputs)) for x in range(number_in_parallel)]
        self.outputs = [np.zeros((1,layer.num_outputs)) for x in range(number_in_parallel)]

    def initialize_weights(self, init_factor):
        self.layer.initialize_weights(init_factor)

    def set_learning_rate(self,rate):
        self.layer.set_learning_rate(rate)

    def set_momentum(self,momentum):
        self.layer.set_momentum(momentum)

    def get_output(self, inputs):
        """
        Inputs comes in the shape batch_size x (num_inputs * number_in_parallel)
        The inner layer accepts an input of size batch_size x num_inputs

        The inner layer returns an output of size batch_size x num_outputs
        Output returns as batch_size x (num_outputs * number_in_parallel)
        """
        self.output = np.zeros((inputs.shape[0],self.num_outputs))

        in_start = 0
        in_end = self.layer.num_inputs

        out_start = 0
        out_end = self.layer.num_outputs

        for i in range(self.number_in_parallel):
            self.inputs[i] = inputs[:,in_start:in_end]
            self.outputs[i] = self.layer.get_output(self.inputs[i])
            self.output[:,out_start:out_end] = self.outputs[i]

            in_start = in_end
            in_end += self.layer.num_inputs

            out_start = out_end
            out_end += self.layer.num_outputs

        return self.output

    def update_weights(self, error_deriv):
        """
        Error deriv comes in as batch_size x (num_outputs * number_in_parallel)
        """
        out_start = 0
        out_end = self.layer.num_outputs

        for i in range(self.number_in_parallel):
            self.layer.input = self.inputs[i]
            self.layer.output = self.outputs[i]

            self.layer.update_weights(error_deriv[:,out_start:out_end])

            out_start = out_end
            out_end += self.layer.num_outputs

    def next_error_deriv(self, error_deriv):
        """
        This method not implemented in a way that works with backprop.
        This is a selfish class that will not share error information.
        This method is just here to satisfy the Network class.
        """
        return None

    def get_error(self, target):
        raise NotImplementedError()

    def get_error_deriv(self, target):
        raise NotImplementedError()

    def activation(self, z):
        raise NotImplementedError()

    def activation_deriv(self):
        raise NotImplementedError()
