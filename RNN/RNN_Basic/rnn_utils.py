import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis = 0)

def initialize_adam(parameters):
    #Function return keys and values:
            #Keys : dW1, db1, ..., dWL, dbL
            #Values : zeros matrix corresponding to the dimension of parameters

    #Parameters : Dictionary contains Parameters
            #parameters["W" + str(1)] = W1
            #parameters["b" + str(1)] = b1

    #Return:
        # V : python Dictionary that Contains exponentially weighted average of the gradient
        # S : python Dictionary that contains exponentially weighted average of the squared gradient

    L = len(parameters)
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

    return v,s

def update_parameters_with_adam(parameters, gradients, v, s, t, learning_rate = 0.1,
                                    beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
    L = len(parameters)
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l+1)] = beta1 * V["dW" + str(l+1)] + (1 - beta1) * gradients["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * V["db" + str(l+1)] + (1 - beta1) * gradients["db" + str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)

        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * gradients["dW" + str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * gradients["db" + str(l+1)]**2

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / np.sqrt(s_corrected["dW" + str(l+1)] + eps)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / np.sqrt(s_corrected["db" + str(l+1)] + eps)

    return parameters, v, s
