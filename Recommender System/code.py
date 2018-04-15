import numpy as np
import csv
import matplotlib.pyplot as plt



def read_csv(file):
    with open(file) as f:
        userIDs = []
        X_train = []
        Y_train = []
        first = True
        for line in f:
            if first:
                first = False
            else:
                line = line.split(",")
                userIDs.append(line[0])
                X_train.append(line[1:3])
                Y_train.append(line[3])

        return np.array(X_train, dtype=np.int), np.array(Y_train, dtype=np.int), np.unique(np.array(userIDs, dtype=np.int)), np.array(userIDs, dtype=np.int)

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def sigmoid_backward(dA, Z):
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def softmax(Z):
    exps = np.exp(Z - np.max(Z))
    return exps / np.sum(exps)

def softmax_backward(AL, Y):
    dZ = AL - Y
    return dZ

def convert_to_oh(Y_train, C=4):
    Y = np.eye(C+1)[Y_train.reshape(-1)]
    return Y

def convert_dictionary_to_vector(dictionary):
    row = dictionary.shape[0] # 6
    column = dictionary.shape[1] # 2
    vector = np.zeros(row * column) # 12
    start = 0
    for c in range(column):
        temp = dictionary[:,c] # (6,1)
        end = row * (c + 1)
        vector[start : end] = temp
        start = end


    return np.array(vector, dtype = np.float)

def convert_vector_to_dictionary(vector, layers_dim, l):
    row = layers_dim[l+1]
    column = layers_dim[l]
    dictionary = []
    start = 0
    for c in range(column):
        end = row * (c + 1)
        temp = vector[start:end]
        print (temp)
        start = end
        dictionary.append(temp)

    return np.array(dictionary, dtype=np.float).T


def initialize_parameters(n_a, layers_dim):
    parameters = {}
    for n in range(n_a-1):
        parameters["W" + str(n+1)] = np.random.randn(layers_dim[n+1], layers_dim[n]) * np.sqrt(2/(layers_dim[n+1] + layers_dim[n]))
        parameters["b" + str(n+1)] = np.ones((layers_dim[n+1],1))

    return parameters

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
    with open("all_users.csv") as f:
        for l in f:
            l = l.split(",")
            l = np.array(l, dtype=np.int)
            if user in l:
                None
                # open users/user##_profile.csv
            else:
                parameters = initialize_parameters(L, layers_dim)
    print (parameters)
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
    with open("users/user1_profile.csv", "w") as csvfile:
        fieldnames = ["W1", "b1", "W2", "b2", "W3", "b3"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(parameters)
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
    Y = np.argmax(AL)
    print (AL)
    return Y



if __name__ == "__main__":
    # data = input("Input data: ")
    # print(data.split(" "))
    # X_train, Y_train, all_user, user_train = read_csv("data.csv")
    # Y_oh = convert_to_oh(Y_train[0])
    training_input = input("please enter input (user, prev1, prev2, Y): ")
    training_input = training_input.split(",")
    user = int(training_input[0])
    X_train = np.array(training_input[1:3], dtype=np.int)
    Y_train = np.array(training_input[-1], dtype=np.int)

    # softmax = softmax(np.array(data.split(" "), dtype=np.int).reshape((4,1)))
    # print (softmax)
    # print(np.sum(softmax))
    layers_dim = [2,6,10,5]
    # parameters = initialize_parameters(userIDs, len(layers_dim), layers_dim)
    # caches, AL = forward_probagation(X_train[1].reshape((2,1)), layers_dim, parameters, userIDs[0])
    # # caches, userID, AL = caches
    # cost = cost_func(Y_oh.reshape((5,1)), AL)
    # grads, userID = backward_probagation(Y_oh.reshape((5,1)) ,caches, layers_dim)
    # updated_parameters = update_parameters(parameters, grads, userID, layers_dim, alpha = 0.01)
    # print(AL)
    # print(np.sum(AL))
    # print(Y_oh.shape)
    # print(cost)
    # X = [[1],[1]]
    # user = 3
    parameters = model(X_train, Y_train, user, layers_dim)

    # vector = convert_dictionary_to_vector(parameters[str(user)]["W2"])
    # dictionary = convert_vector_to_dictionary(vector, layers_dim, l = 1)
    # Y = predict(X, user, parameters, layers_dim)
    # print (Y)
    # print (vector)
    # print (dictionary)
    # print(parameters[str(user)]["W2"])