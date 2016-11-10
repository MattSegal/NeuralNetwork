import py.test
import numpy as np
from layer          import Layer
from linear_layer   import LinearLayer
from logistic_layer import LogisticLayer
from relu_layer     import ReluLayer
from softmax_layer  import SoftmaxLayer
from softplus_layer import SoftplusLayer
from tanh_layer     import TanhLayer

def check_weight_range(layer, n):
    layer.initialize_weights(n)
    assert np.max(layer.weights) <= n
    assert np.min(layer.weights) >= -1*n

def test_initialize_weights():
    lay = Layer(10,10)
    check_weight_range(lay,1)
    check_weight_range(lay,10)
    check_weight_range(lay,100)

    lay = Layer(20,20)
    check_weight_range(lay,1)
    check_weight_range(lay,10)
    check_weight_range(lay,100)

def test_get_output_linear():
    lay = LinearLayer(1,1)
    inputs = np.array([[2]])
    lay.weights = np.ones(lay.weights.shape)
    lay.bias = 5* np.ones(lay.bias.shape)
    output = lay.get_output(inputs)
    assert output.shape == (1L,1L)
    expected_output = np.array([[7]])
    np.testing.assert_array_equal(output,expected_output)

    lay = LinearLayer(2,3)
    inputs = np.array([[2,1]])
    lay.weights = np.ones(lay.weights.shape)
    lay.bias = 5* np.ones(lay.bias.shape)
    output = lay.get_output(inputs)
    assert output.shape == (1L,3L)
    expected_output = np.array([
        [8,8,8]
    ])
    np.testing.assert_array_equal(output,expected_output)

def test_get_output_logistic():
    lay = LogisticLayer(1,1)
    inputs = np.array([[2]])
    lay.weights = np.ones(lay.weights.shape)
    lay.bias = 5* np.ones(lay.bias.shape)
    output = lay.get_output(inputs)
    assert output.shape == (1L,1L)
    lin_out = np.array([[7]])
    expected_output = 1/(1+np.exp(-lin_out))
    np.testing.assert_array_equal(output,expected_output)

    lay = LogisticLayer(2,3)
    inputs = np.array([[2,1]])
    lay.weights = np.ones(lay.weights.shape)
    lay.bias = 5* np.ones(lay.bias.shape)
    output = lay.get_output(inputs)
    assert output.shape == (1L,3L)
    lin_out = np.array([
        [8,8,8]
    ])
    expected_output = 1/(1+np.exp(-lin_out))
    np.testing.assert_array_equal(output,expected_output)

def test_get_error():
    lay = Layer(1,1)
    
    lay.output = np.zeros((1,1))
    target = np.ones((1,1))
    expected_error = 0.5
    actual_error = lay.get_error(target)
    assert expected_error == actual_error

    target = target * 0
    expected_error = 0
    actual_error = lay.get_error(target)
    assert expected_error == actual_error

    lay = Layer(2,2)
    lay.output = np.zeros((2,2))
    target = np.array([
        [1,2], # batch 1
        [3,4]  # batch 2
    ])
    expected_error = 0.5*(1*1 + 2*2 + 3*3 + 4*4)
    actual_error = lay.get_error(target)
    assert expected_error == actual_error

def test_get_error_deriv():
    lay = Layer(1,1)
    lay.output = np.zeros((1,1))
    target = np.ones((1,1))
    expected_deriv = -target
    actual_deriv = lay.get_error_deriv(target)
    np.testing.assert_array_equal(expected_deriv,actual_deriv)

    target = target * 0
    expected_deriv = target
    actual_deriv = lay.get_error_deriv(target)
    np.testing.assert_array_equal(expected_deriv,actual_deriv)

    lay = Layer(2,2)
    lay.output = np.zeros((2,2))
    target = np.array([
        [1,2],
        [3,4]
    ])
    expected_deriv = np.array([
        [-1,-2],
        [-3,-4]
    ])
    actual_deriv = lay.get_error_deriv(target)
    np.testing.assert_array_equal(expected_deriv,actual_deriv)

def test_next_error_deriv_linear():
    """
    we have
         
     zi  i  yi  zj  j   yj
    ---> O -------> O ------ > output (yj), target (t), error (E)
            wij

    we know dE/dyj (error_deriv) and we want dE/dyi

    first get dE/dzj

    dE/dzj = dyj/dzj * dE/dyj

           =  activation_deriv * error_deriv

    dE/dyi = dzj/dyi * dE/dzj 

            =  wij   * activation_deriv * error_deriv
    """
    # Simplest case
    lay                 = LinearLayer(1,1)
    lay.input           = np.array([[1]]) # for shape
    lay.output          = np.array([[1]])
    lay.weights         = np.array([[2]])
    error_deriv         = np.array([[2]])
    expected_next_deriv = np.array([[4]])

    actual_next_deriv = lay.next_error_deriv(error_deriv)
    np.testing.assert_array_equal(expected_next_deriv,actual_next_deriv)

    # Multiple batches
    lay                 = LinearLayer(1,1)
    lay.input           = np.array([[1],[1]]) # for shape
    lay.output          = np.array([[2],[1]])
    lay.weights         = np.array([[2]])
    error_deriv         = np.array([[3],[2]])
    expected_next_deriv = np.array([[6],[4]])

    actual_next_deriv = lay.next_error_deriv(error_deriv)
    np.testing.assert_array_equal(expected_next_deriv,actual_next_deriv)


    # Multiple dimensions
    lay                 = LinearLayer(3,2)
    lay.input           = np.array([[1,2,3]]) # for shape
    lay.output          = np.array([[1,2]])
    lay.weights         = np.array([
                            [1,1],
                            [2,2],
                            [3,3]
                        ])
    error_deriv         = np.array([[1,2]])
    expected_next_deriv = np.array([[3,6,9]])

    actual_next_deriv = lay.next_error_deriv(error_deriv)
    np.testing.assert_array_equal(expected_next_deriv,actual_next_deriv)

def test_next_error_deriv_logistic():
    # Simplest case
    lay                 = LogisticLayer(1,1)
    lay.input           = np.array([[1]]) # for shape
    lay.output          = np.array([[2]])
    logistic_deriv      = lay.output * (1-lay.output)
    lay.weights         = np.array([[2]])
    error_deriv         = np.array([[2]])
    expected_next_deriv = np.array([[4]]) * logistic_deriv

    actual_next_deriv = lay.next_error_deriv(error_deriv)
    np.testing.assert_array_equal(expected_next_deriv,actual_next_deriv)

    # Multiple batches
    lay                 = LogisticLayer(1,1)
    lay.input           = np.array([[1],[1]]) # for shape
    lay.output          = np.array([[2],[1]])
    logistic_deriv      = lay.output * (1-lay.output)
    lay.weights         = np.array([[2]])
    error_deriv         = np.array([[3],[2]])
    expected_next_deriv = np.array([[6],[4]]) * logistic_deriv

    actual_next_deriv = lay.next_error_deriv(error_deriv)
    np.testing.assert_array_equal(expected_next_deriv,actual_next_deriv)

    # Multiple dimensions
    lay                 = LogisticLayer(3,2)
    lay.input           = np.array([[1,2,3]]) # for shape
    lay.output          = np.array([[1,2]])
    logistic_deriv      = np.array([[0,-2]])
    lay.weights         = np.array([
                            [1,1],
                            [2,2],
                            [3,3]
                        ])
    error_deriv         = np.array([[1,2]])
    expected_next_deriv = np.array([[-4,-8,-12]])

    actual_next_deriv = lay.next_error_deriv(error_deriv)
    np.testing.assert_array_equal(expected_next_deriv,actual_next_deriv)

def test_update_weights():
    # cbf
    pass

def test_softmax_activation():
    # Simple case
    lay = SoftmaxLayer(1,1)
    input_arr = np.array([[7]])
    expected_arr = np.array([[1]])
    actual_arr = lay.activation(input_arr)
    np.testing.assert_array_equal(expected_arr,actual_arr)

    # Simple case with batches
    lay = SoftmaxLayer(1,1)
    input_arr = np.array([[2],[7]])
    expected_arr = np.array([[1],[1]])
    actual_arr = lay.activation(input_arr)
    np.testing.assert_array_equal(expected_arr,actual_arr)

    # Multiple inputs
    lay = SoftmaxLayer(4,4)
    input_arr = np.array([[1,2,3,4]])
    exp_sum = np.exp(1)+np.exp(2)+np.exp(3)+np.exp(4)
    expected_arr = np.exp(np.array([[1,2,3,4]])) / exp_sum
    actual_arr = lay.activation(input_arr)

    # round down to nearest 10 decimal places 
    # because floating point sucks
    expected_arr = np.round(expected_arr,10)
    actual_arr = np.round(actual_arr,10)
    np.testing.assert_array_equal(expected_arr,actual_arr)

    # Multiple inputs
    lay = SoftmaxLayer(4,4)
    input_arr = np.array([[1,2,3,4],[5,6,7,8]])

    exp_sum = np.exp(1)+np.exp(2)+np.exp(3)+np.exp(4)
    expected_arr1 = np.exp(np.array([1,2,3,4])) / exp_sum
    
    exp_sum = np.exp(5)+np.exp(6)+np.exp(7)+np.exp(8)
    expected_arr2 = np.exp(np.array([5,6,7,8])) / exp_sum

    expected_arr = np.array([expected_arr1,expected_arr2])

    actual_arr = lay.activation(input_arr)

    # round down to nearest 10 decimal places 
    # because floating point sucks
    expected_arr = np.round(expected_arr,10)
    actual_arr = np.round(actual_arr,10)
    np.testing.assert_array_equal(expected_arr,actual_arr)    