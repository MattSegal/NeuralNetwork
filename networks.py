import numpy as np
from network import Network

class NetworkArray(object):
    def __init__(self,layers,num_nets,batch_size):
        """
        Layers is a list of Layer objects
        """
        self.networks = []
        for i in range (num_nets):
            net = Network(layers)
            self.networks.append(net)

        self.output = np.zeros((batch_size,num_nets))

    def set_learning_rate(self,rate):
        for net in self.networks:
            net.set_learning_rate(rate)

    def initialize_weights(self,init_factor):
        for net in self.networks:
            net.initialize_weights(init_factor)

    def set_momentum(self,momentum):
        for net in self.networks:
            net.set_momentum(momentum)

    def forward_prop(self,inputs):
        """
        Calculate network output given a set of inputs
        """
        
        for i in range(len(self.networks)):
            self.output[:,i] = self.networks[i].forward_prop(inputs)[:,0]
        return self.output

    def back_prop(self,target):
        """
        Update network weights based on the
        networks output and the training set

        The target is a batch x output matrix
        Each network accepts a batch x 1 matrix
        We assume each outcome is mutually exclusive

        Each net gets its own target

        target is a number in [0,num_nets)
        """
        for i in range(len(self.networks)):
            self.networks[i].back_prop(self.get_net_target(target,i))

    def get_error(self,target):
        """
        Get the error / cost / cross-entropy
        of the network over the current batch of training examples
        """
        error = np.zeros(len(self.networks))
        for i in range(len(self.networks)):
            error[i] = self.networks[i].get_error(self.get_net_target(target,i))
        return np.mean(error)

    def get_net_target(self,target,desired_value):
        self.net_target = np.zeros((target.shape[0],1))
        self.net_target[target == desired_value] = 1
        return self.net_target