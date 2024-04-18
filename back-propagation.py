import numpy as np
import math

# Error
def error (output, target):
    return 0.5 * (target - output) ** 2

def error_softmax(prob):
    return -math.log(prob)

# Partial Derivative
def dE_dOut(output, target):
    return - (target - output)

def dE_dOut_softmax(output):
    pass

def dNet_dW(inputs, index):
    return inputs[index]

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

# activation function derivative
def linear_derivative(arr):
    return np.ones_like(arr)

def relu_derivative(arr):
    return np.where(arr > 0, 1, 0)

def sigmoid_derivative(arr):
    sigmoid_output = sigmoid(arr)
    return sigmoid_output * (1 - sigmoid_output)

def softmax_derivative(arr):
    softmax_output = softmax(arr)
    jacobian_matrix = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)
    return jacobian_matrix

def dOut_dNet(arr, activation):
    if activation == 'linear':
        return linear_derivative(arr)
    elif activation == 'relu':
        return relu_derivative(arr)
    elif activation == 'sigmoid':
        return sigmoid_derivative(arr)
    elif activation == 'softmax':
        return softmax_derivative(arr)

# Calculate gradient
def calculate_gradient_output(output, target, activation):
    if activation == 'softmax':
        return dE_dOut_softmax(output)
    else:
        return dE_dOut(output, target) * dOut_dNet(output, activation)

# pseudocode
def back_propagation(model):
    # Asumsi input model yang sudah ffnn

    # Calculate error
    # output_error
    output_error = []
    for output_layer in model.output_layer:
        output_error.append(calculate_gradient_output(output_layer.output, output_layer.target, output_layer.activation))

    # hidden_error
    hidden_error = []
    for hidden_layer in model.hidden_layer:
        weight_error = 0
        for i, output_layer in enumerate(model.output_layer):
            weight_error += output_error[i] * output_layer.weight[i]
        hidden_error.append(weight_error * dOut_dNet(hidden_layer.output, hidden_layer.activation))

    # Update weight
    # output weight
    for i, output_layer in enumerate(model.output_layer):
        for j, weight in enumerate(output_layer.weight):
            output_layer.weight[j] = weight - model.learning_rate * output_error[i] * dNet_dW(output_layer.input, j)

    # hidden weight
    for i, hidden_layer in enumerate(model.hidden_layer):
        for j, weight in enumerate(hidden_layer.weight):
            hidden_layer.weight[j] = weight - model.learning_rate * hidden_error[i] * dNet_dW(hidden_layer.input, j)



