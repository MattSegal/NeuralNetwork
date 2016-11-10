import numpy as np

class Network(object):
    def __init__(self,layers):
        """
        Layers is a list of objects that implement Layer
        """
        self.layers = layers
        self.num_layers = len(self.layers)
        self.verify_layers()
        self.final_layer = self.layers[-1]

    def verify_layers(self):
        """
        Checks that the number of inputs to each layer
        is the same as the number of outputs of the previous layer
        """
        prev_layer_outputs = self.layers[0].weights.shape[0]
        for layer in self.layers:
            this_layer_inputs = layer.weights.shape[0]
            this_layer_outputs = layer.weights.shape[1]
            assert prev_layer_outputs == this_layer_inputs
            prev_layer_outputs = this_layer_outputs

    def set_learning_rate(self,rate):
        for layer in self.layers:
            layer.learning_rate = rate

    def initialize_weights(self,init_factor):
        for layer in self.layers:
            layer.initialize_weights(init_factor)

    def set_momentum(self,momentum):
        for layer in self.layers:
            layer.momentum = momentum

    def forward_prop(self,inputs):
        """
        Calculate network output given a set of inputs
        """
        self.layers[0].get_output(inputs)
        for i in range(1,self.num_layers):
            inputs = self.layers[i-1].output
            self.layers[i].get_output(inputs)
        return self.final_layer.output

    def back_prop(self,target):
        """
        Update network weights based on the
        networks output and the training set

        Start at the output layer and propagate
        error information back to earlier layers 
        """
        error_deriv = self.get_error_deriv(target)
        next_error_deriv = self.final_layer.next_error_deriv(error_deriv)
        self.final_layer.update_weights(error_deriv)
        error_deriv = next_error_deriv

        for layer in reversed(self.layers[:-1]):
            next_error_deriv = layer.next_error_deriv(error_deriv)
            layer.update_weights(error_deriv)
            error_deriv = next_error_deriv

    def get_error(self,target):
        """
        Get the error / cost / cross-entropy
        of the network over the current batch of training examples
        """
        return  self.final_layer.get_error(target)

    def get_error_deriv(self,target):
        """
        Get derivative of error function at the
        for output layer
        """
        return  self.final_layer.get_error_deriv(target)