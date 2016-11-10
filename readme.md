# Network

### Overview
This package provides a simple fully connected neural network implementation. The module API allows you to stack layers of neurons to create a neural network which can perform classification or regression tasks.

This module was originally written to assit in attempting the first assignment of Coursera's Neural Networks course.

### Layers

The network may be composed of several types of layer: 

* Linear
** fits linear regression to input
** cannot fit non-linear models
* Softplus
** non-linear regression unit
* Rectified Linear (ReLu)
** discontinuous approximation of softplus
** supposed to be computationally cheaper
* Logistic
** fits data to range between 0 and 1
** useful for estimating probabilities
* Tanh
** logistic unit scaled between -1 and 1
** supposedly trains faster than the logistic unit
* Softmax
** generalised case of logistic units
** useful for guessing probabilities across multiple classes

The parallel layer was created in an attempt to support alternate network architectures that were not fully connected. I think this kind of layer leads to leakiness in the module's abstractions. For this kind of different architecture a node based model may make more sense.

## Notes

* See layer\test_layer.py for some unit tests for each layer (WIP)
* See example.py for simple example.
* See requirements.txt for module dependencies.

## Ideas

Further work could involve adding support for
* Normalisation (massage input data)
* Regularlisation (force a preference for simpler models)
* Different network architectures (eg. convolutional)
* Using Theano to speed up computation