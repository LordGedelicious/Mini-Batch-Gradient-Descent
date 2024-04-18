import numpy as np

def error (output, target):
    return 0.5 * (target - output) ** 2

def dE_dOut(output, target):
    return - (target - output)

def dNet_dW(inputs, index):
    return inputs[index]

def back_propagation(model_src_name):
    pass

# Activation functions
def linear(arr):
    return arr

def relu(val):
    if val < 0:
        return 0
    else:
        return val
    
def sigmoid(arr):
    numerator = np.exp(arr)
    denominator = numerator + 1
    return numerator / denominator

def softmax(arr):
    arr = np.array(arr)
    arr = np.exp(arr)
    sum_arr = np.sum(arr)
    return arr/sum_arr

# Derivative versions for activation functions
# Derivative of linear activation function is simply 1
def linear_derivative(arr):
    return np.ones_like(arr)

# Derivative of ReLU activation function
def relu_derivative(arr):
    return np.where(arr > 0, 1, 0)

# Derivative of sigmoid activation function
def sigmoid_derivative(arr):
    sigmoid_output = sigmoid(arr)
    return sigmoid_output * (1 - sigmoid_output)

# Derivative of softmax activation function
def softmax_derivative(arr):
    softmax_output = softmax(arr)
    jacobian_matrix = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)
    return jacobian_matrix