from matplotlib import pyplot as plt
import numpy as np
import os

import json
import os

def parse_json_file(file_path: str):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    
    case = data['case']
    expect = data['expect']
    
    return case, expect

def parse_case(data: dict):
    model = data['model']
    input_data = data['input']
    initial_weights = data['initial_weights']
    target = data['target']
    learning_parameters = data['learning_parameters']

    return model, input_data, initial_weights, target, learning_parameters

def parse_model(data: dict):
    input_size = data['input_size']
    layers = data['layers']

    return input_size, layers

def parse_layers(data: dict):
    number_of_neurons = data['number_of_neurons']
    activation_function = data['activation_function']

    return number_of_neurons, activation_function

def parse_learning_parameters(data: dict):
    learning_rate = data['learning_rate']
    batch_size = data['batch_size']
    max_iteration = data['max_iteration']
    error_threshold = data['error_threshold']

    return learning_rate, batch_size, max_iteration, error_threshold

def parse_expect(data: dict):
    stopped_by = data['stopped_by']
    final_weights = data['final_weights']

    return stopped_by, final_weights

def save_json_file(file_path: str, data: dict):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

class ActivationFunctions:
    @staticmethod
    def linear(x, derivative=False):
        if derivative:
            return 1
        return x
    
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x <= 0, 0, 1)
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x, derivative=False):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class NN:
    def __init__(self, model_config, initial_weights):
        self.input_size, self.layers = parse_model(model_config)
        self.weights = np.array(initial_weights)    
        self.bias = 1
        self.learning_rate = None
        self.batch_size = None
        self.max_iteration = None
        self.error_threshold = None
        
    def forward_propagation(self, x):
        input_data = np.array(x)

        for layer, weight in zip(self.layers, self.weights):
            input_data = np.append(self.bias, input_data)
            activation_function = getattr(ActivationFunctions, layer['activation_function'])
            output = activation_function(np.dot(input_data, weight))
            
            input_data = output

        return output

    def train(self, input_data, target, learning_rate, batch_size, max_iteration, error_threshold):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.error_threshold = error_threshold

        for epoch in range(max_iteration):
            total_error = 0.0

            for i in range(0, len(input_data), batch_size):
                batch_input = input_data[i:i+batch_size]
                batch_target = target[i:i+batch_size]

                batch_error = 0.0

                deltas = [None] * len(self.layers)

                for x, t in zip(batch_input, batch_target):
                    output = self.forward_propagation(x)
                    
                    # Compute error
                    error = t - output
                    batch_error += np.sum(error ** 2 / 2)

                    # Backpropagation
                    # delta = - error * f'(output)
                    delta = - error 

                    # Output layer
                    for idx, layer in enumerate(self.layers[::-1]):
                        activation = getattr(ActivationFunctions, layer['activation_function'])
                        delta_output = delta * activation(output, derivative=True)
                        dw = np.append(self.bias, x)
                        delta_output = learning_rate * np.outer(delta_output, dw)
                        if deltas[-idx-1] is None:
                            deltas[-idx-1] = delta_output
                        else:
                            deltas[-idx-1] += delta_output

                        delta = np.dot(self.weights[-idx-1][:, 1:].T, delta_output.T)
                        
                total_error += batch_error

                # Update weights
                for idx, delta in enumerate(deltas):
                    self.weights[idx] -= delta.T

            avg_error = total_error / len(input_data)


            print(f"Epoch {epoch + 1}/{max_iteration}, Average Error: {avg_error}")

            if avg_error < error_threshold:
                print("Training stopped: Error threshold reached.")
                break

        print("Training complete.")


if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), 'testcase', 'linear.json')

    try:
        case, expect = parse_json_file(file_path)
        model, input_data, initial_weights, target, learning_parameters = parse_case(case)
        stopped_by, final_weights = parse_expect(expect)
        
    except Exception as e:
        raise e

    try:
        nn = NN(model, initial_weights)
        learning_rate, batch_size, max_iteration, error_threshold = parse_learning_parameters(learning_parameters)
        nn.train(input_data, target, learning_rate, batch_size, max_iteration, error_threshold)

        print(nn.weights)

    except Exception as e:
        raise e
