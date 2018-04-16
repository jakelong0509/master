import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle as pck
from utils import read_csv, sigmoid, sigmoid_backward, softmax, softmax_backward, convert_to_oh, convert_dictionary_to_vector, convert_vector_to_dictionary, initialize_parameters

def cell_forward(W, b, A_prev, type="sigmoid"):
    Z = np.dot(W, A_prev) + b
    if type == "softmax":
        A = softmax(Z)
    elif type == "sigmoid":
        A = sigmoid(Z)

    cache = (W,b,A_prev,Z)
    return cache, A

def forward_probagation(X_train, layers_dim, parameters):
    A_prev = X_train
    caches = []
    L = len(layers_dim)
    for n in range(L-2):
        cache, A = cell_forward(parameters["W" + str(n+1)], parameters["b" + str(n+1)], A_prev)
        A_prev = A
        caches.append(cache)

    cacheL, AL = cell_forward(parameters["W" + str(L-1)], parameters["b" + str(L-1)], A_prev, type="softmax")
    caches.append(cacheL)
    caches = (caches, AL)

    return caches, AL

def cost_func(Y_train, AL):
    cost = -np.sum(np.multiply(Y_train, np.log(AL)), axis=0)
    return cost

def cell_backward(Y_train ,cache,type="sigmoid", AL = None, dA = None):
    W,b,A_prev,Z = cache
    dZ = []
    if type == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
    elif type == "softmax":
        dZ = softmax_backward(AL, Y_train)
    dW = np.dot(dZ, A_prev.T)
    db = dZ
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_probagation(Y_train ,caches, layers_dim):
    caches, AL = caches
    L = len(layers_dim)
    current_cache = caches[-1]
    dA_prev, dW, db = cell_backward(Y_train ,current_cache, type="softmax", AL = AL)
    first = True
    grads = {"dW3": dW, "db3": db }
    count = L - 2
    for cache in reversed(caches):
        if first:
            first = False
        else:
            dA_prev, dW, db = cell_backward(Y_train, cache, type="sigmoid", AL = None, dA = dA_prev)
            grad = {"dW" + str(count): dW, "db" + str(count): db}
            grads.update(grad)
            count -= 1


    return grads

def update_parameters(parameters, grads, layers_dim, alpha = 0.01):
    L = len(layers_dim)
    for l in range(L-1):
        parameters["W" + str(l+1)] -= alpha * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= alpha * grads["db" + str(l+1)]

    return parameters

def model(X_train, Y_train, user, layers_dim, C = 4, lr = 0.1):
    L = len(layers_dim)
    parameters = []
    flag = False
    with open("all_users.csv", "r+") as f:
        for l in f:
            l = np.array(l, dtype=np.int)
            print(l)
            if user in l:
                parameters = pck.load(open("users/user" + str(user) + "_profile.p", "rb"))
                flag = True
                break
        if not flag:
            parameters = initialize_parameters(L, layers_dim)
            f.write(str(user) + "\n")
    # global_cost = {}
    # for id in all_user:
    #     global_cost[str(id)] = []
    # Stochastic gradient descent
    X = X_train.reshape((2,1))
    Y_oh = convert_to_oh(Y_train).reshape((C+1, 1))
    caches, AL = forward_probagation(X, layers_dim, parameters)
    cost = cost_func(Y_oh, AL)
    grads = backward_probagation(Y_oh, caches, layers_dim)
    parameters = update_parameters(parameters, grads, layers_dim, lr)
    pck.dump(parameters, open("users/user" + str(user) + "_profile.p", "wb"))

    # global_cost[str(userID)].append(cost)

    # for id in all_user:
    #     plt.plot(global_cost[str(id)])
    #     plt.show()

    return parameters

def predict(X, user, parameters, layers_dim):
    A_prev = X
    L = len(layers_dim)
    for l in range(L-2):
        Z = np.dot(parameters["W" + str(l+1)], A_prev) + parameters["b" + str(l+1)]
        A = sigmoid(Z)
        A_prev = A

    ZL = np.dot(parameters["W" + str(L-1)], A_prev) + parameters["b" + str(L-1)]
    AL = softmax(ZL)
    print (AL)
    Y = np.argmax(AL)
    return Y

if __name__ == "__main__":
    training_input = input("please enter input (user, prev1, prev2, Y): ")
    training_input = training_input.split(",")
    user = int(training_input[0])
    X_train = np.array(training_input[1:3], dtype=np.int).reshape((2,1))
    Y_train = np.array(training_input[-1], dtype=np.int)
    layers_dim = [2,6,10,5]
    parameters = model(X_train, Y_train, user, layers_dim)


    test = input("Do you want to test?(y/n)")
    if test == "y":
        test_input = input("Please enter test input (userID, prev1, prev2)")
        test_input = test_input.split(",")
        user = int(test_input[0])
        X = np.array(test_input[1:3], dtype=np.int).reshape((2,1))
        Y = predict(X, user, parameters, layers_dim)
        print (Y)
