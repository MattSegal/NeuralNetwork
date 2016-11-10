"""
This example is here to give you an ideal of how
this module could be used
"""
import math
import numpy as np

# import from within module
from layer import TanhLayer
from network import Network

def training_example():
    # Setup hyperparameters
    num_training_ex = 500
    batch_size      = 20
    num_iterations  = 50
    plot_interval   = 10
    noise           = 0

    initial_weights = 1
    learning_rate   = 0.1
    momentum        = 0.5

    # Setup some dummy training data
    # Use a circle of radius 2 with positive examples inside
    # and negative examples outside
    data = TrainingDataCircle()
    data.lower_value = -1 # because of Tanh final layer
    data.gen_input_data(num_training_ex,span=4)
    data.add_noise(mean=0,scale=noise)
    data.gen_batches(batch_size)

    # Setup network
    layers = [
        TanhLayer(2,3), # 2 inputs feed into 3 hidden neurons
        TanhLayer(3,1)  # 3 hidden neurons feed into 1 output neuron
    ]

    net = Network(layers)
    net.initialize_weights(initial_weights)
    net.set_learning_rate(learning_rate)
    net.set_momentum(momentum)

    # Train network
    error_history = []
    for itr in range(num_iterations):
        for batch in range(data.num_batches):

            # Get the next batch
            input_data  = data.get_input_batch(batch)
            target_data = data.get_output_batch(batch)

            # Calculate the predicted outputs
            net.forward_prop(input_data)
            
            # Update the weights by comparing prediction with result
            net.back_prop(target_data)

            # Record progress - this could be plotted after training
            if itr % plot_interval == 0 and batch == 0:
                error = net.get_error(target_data)
                error_history.append(error)

    print "Finished training - results:"

    # Print layer data
    count = 0
    for layer in net.layers:
        print "\nParameters for layer {0}:".format(count)
        print "Weights"
        print layer.weights
        print "Bias"
        print layer.bias
        count += 1

    # Print network accuracy
    net_output = net.forward_prop(data.input)
    accuracy_arr = (net_output.flatten() >= 0.5) == (data.output >= 0.5)
    accuaracy = 100 * np.sum(accuracy_arr) / float(accuracy_arr.size)
    print "\nNetwork classified %s percent of training examples correctly." % int(accuaracy)


class TrainingDataCircle:
    """
    This creates a mock data-set that is a circle of radius 2 at (0,0)
    """
    lower_value = 0 # logistic default
    upper_value = 1 # logistic default
    def circleize(self,arr):
        """
        Circleize (verb): to make a circle.
        function for a cirlce of radius 2
        takes a n x 2 array
        returns a n x 1 array
        the column is 1 if in circle, 0 if outside
        """
        z = arr[:,0]*arr[:,0] + arr[:,1]*arr[:,1]
        z[z<=4] = self.upper_value
        z[z>4] = self.lower_value
        return z

    def gen_input_data(self,num_training_ex,span):
        """
        Use 1:1 ratio of pos:neg examples, could make this an imput param later
        We are hard coding the circle of radius 2 into this function
        """
        circ_radius = 2
        self.num_training_ex = num_training_ex
        num_pos_ex =  num_training_ex / 2
        num_neg_ex = num_training_ex - num_pos_ex

        neg_radius = (span - circ_radius -1) * np.random.rand(num_neg_ex) + circ_radius +1
        neg_theta = 2 * math.pi * np.random.rand(num_neg_ex)
        neg_input = np.zeros((num_neg_ex,2))
        neg_input[:,0] = neg_radius * np.cos(neg_theta)
        neg_input[:,1] = neg_radius * np.sin(neg_theta)

        pos_radius = circ_radius * np.random.rand(num_pos_ex) 
        pos_theta = 2 * math.pi * np.random.rand(num_pos_ex)
        pos_input = np.zeros((num_pos_ex,2))
        pos_input[:,0] = pos_radius * np.cos(pos_theta)
        pos_input[:,1] = pos_radius * np.sin(pos_theta)

        # scramble input data so that batching isn't biased
        self.input = np.append(neg_input,pos_input,axis=0)
        np.random.shuffle(self.input)
        self.output = self.circleize(self.input)

    def add_noise(self,mean,scale=1):
        if scale == 0:
            return
        noise = np.random.normal(mean,scale,size=self.num_training_ex)
        self.output = self.output + noise

    def gen_batches(self,batch_size):
        self.batch_size = batch_size
        self.num_batches = self.num_training_ex / batch_size
        self.input_batch = np.reshape(self.input,(-1,batch_size,2))
        self.output_batch = np.reshape(self.output,(-1,batch_size,1))

    def get_input_batch(self,batch_num):
        return self.input_batch[batch_num,:]

    def get_output_batch(self,batch_num):
        return self.output_batch[batch_num,:]


if __name__ == "__main__":
    training_example()