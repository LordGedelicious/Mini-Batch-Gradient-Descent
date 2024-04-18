
def error (output, target):
    return 0.5 * (target - output) ** 2

def dE_dOut(output, target):
    return - (target - output)

def dNet_dW(inputs, index):
    return inputs[index]

def back_propagation(model_src_name):
    pass