import numpy as np
import random
import matplotlib.pyplot as plt
import nltk
import csv
from tempfile import TemporaryFile

# preprocessing data
# read data from text file
data = open("shakespeare.txt", "r").read()

# lower case all character in data
data = data.lower()

# get all unique letters and symbols
token = nltk.word_tokenize(data)

# # split line into array
# data = data.split("\n")

# get alphabet array
# alphabet = 'abcdefghijklmnopqrstuvwxyz'
# alphabet = list(set(alphabet))

# Initialize
specialSymbol = []
dictionary = []
# dictionaryTemp = []
index = 0

#
for line in data:
    if (line == ''):
        data[index] = '\n'
    index += 1

# assign all symbols
specialSymbol = [',', '.', '<', '>', '/', '?', ':', ';', "'", '"', '{', '}', '[', ']', '(', ')', '\n', '-', '_']

# assign single word to dictionary
# for i in data:
#     words = i.split(" ")
#     for word in words:
#         if not word in dictionaryTemp:
#             dictionaryTemp.append(word)
for word in token:
    if not word in dictionary:
        dictionary.append(word)
dictionary.append("\n")


# handle special symbol
# for word in dictionaryTemp:
#     count = 1
#     if word != "" and word not in specialSymbol:
#         if word[-1] in specialSymbol or word[0] in specialSymbol:
#             realWord = word[:len(word)-1]
#             if realWord[-1] in specialSymbol:
#                 count += 1
#                 realWord = word[:len(word)-count]
#             if word[0] in specialSymbol:
#                 realWord = realWord[1:]
#             if not word[-1] in dictionary:
#                 dictionary.append(word[-1])
#             if not realWord in dictionary:
#                 dictionary.append(realWord)
#         else:
#             dictionary.append(word)
# dictionary.append("\n")



word_to_idx = {w:i for i, w in enumerate(dictionary)}
idx_to_word = {i:w for i, w in enumerate(dictionary)}

data_size, vocab_size = len(data), len(dictionary)

def initialize_parameters(n_a, n_x, n_y):
    Wc = np.random.randn(n_a, n_a + n_x) * 0.01
    bc = np.zeros((n_a, 1))
    Wu = np.random.randn(n_a, n_a + n_x) * 0.01
    bu = np.zeros((n_a, 1))
    Wf = np.random.randn(n_a, n_a + n_x) * 0.01
    bf = np.zeros((n_a, 1))
    Wo = np.random.randn(n_a, n_a + n_x) * 0.01
    bo = np.zeros((n_a, 1))
    Wy = np.random.randn(n_y, n_a) * 0.01
    by = np.zeros((n_y, 1))

    parameters = {"Wc": Wc, "Wu": Wu, "Wf": Wf, "Wo": Wo, "Wy": Wy, "bc": bc, "bu": bu, "bf": bf, "bo": bo, "by": by}
    return parameters

def smooth(cur_loss, loss):
    return loss * 0.999 + cur_loss * 0.001

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size) * seq_length

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def softmax(v):
    e_v = np.exp(v - np.max(v))
    return e_v / np.sum(e_v, axis = 0)

def clip(gradients, maxValue):
    dWc, dWu, dWf, dWo, dWy, dbc, dbu, dbf, dbo, dby = gradients["dWc"], gradients["dWu"], gradients["dWf"], gradients["dWo"], gradients["dWy"], gradients["dbc"], gradients["dbu"], gradients["dbf"], gradients["dbo"], gradients["dby"]
    da_prev, dc_prev, dx = gradients["da_prev"], gradients["dc_prev"], gradients["dx"]

    for gradient in [dWc, dWu, dWf, dWo, dWy, dbc, dbu, dbf, dbo, dby, da_prev, dc_prev, dx]:
        gradient = np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWc": dWc, "dWu": dWu, "dWf": dWf, "dWo": dWo, "dWy": dWy, "dbc": dbc, "dbu": dbu, "dbf": dbf, "dbo": dbo, "dby": dby,
                    "da_prev": da_prev, "dc_prev": dc_prev, "dx": dx}
    return gradients

def rnn_lstm_cell_forward(parameters, a_prevt, c_prevt, xt):
    Wc, Wu, Wf, Wo, Wy = parameters["Wc"], parameters["Wu"], parameters["Wf"], parameters["Wo"], parameters["Wy"]
    bc, bu, bf, bo, by = parameters["bc"], parameters["bu"], parameters["bf"], parameters["bo"], parameters["by"]

    concat = np.vstack((a_prevt, xt))

    ut = sigmoid(np.dot(Wu, concat) + bu)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    cct = np.tanh(np.dot(Wc, concat) + bc)

    c_next = np.multiply(ut, cct) + np.multiply(ft, c_prevt)
    a_next = np.multiply(ot, np.tanh(c_next))

    yt_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prevt, c_prevt, ft, ut, ot, cct, xt, parameters)
    return cache, yt_pred, a_next, c_next

def rnn_lstm_full_forward(parameters, a_prev, X, Y, vocab_size):
    x, y, a, c, y_hat = {}, {}, {}, {}, {}
    caches = []
    a[-1] = np.copy(a_prev)
    c[-1] = np.copy(a_prev)

    loss = 0

    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        y[t] = np.zeros((vocab_size, 1))
        y[t][Y[t]] = 1
        if X[t] != None:
            x[t][X[t]] = 1

        cache, y_hat[t], a[t], c[t] = rnn_lstm_cell_forward(parameters, a[t-1], c[t-1], x[t])
        loss -= np.log(y_hat[t][Y[t]])


        caches.append(cache)
    loss = np.sum(loss, axis = 0)
    caches = (caches, y_hat, a, c, x)
    return caches, loss, a

def rnn_lstm_cell_backward(cache, da_next, dc_next, dy):
    a_next, c_next, a_prevt, c_prevt, ft, ut, ot, cct, xt, parameters = cache
    n_x, _ = xt.shape
    n_a, _ = a_next.shape

    dot = np.multiply(np.multiply(da_next, np.tanh(c_next)), np.multiply(ot, 1 - ot))
    dft = np.multiply(np.multiply(dc_next, c_prevt) + np.multiply(np.multiply(ot, 1 - np.tanh(c_next)**2), np.multiply(c_prevt, da_next)), np.multiply(ft, 1- ft))
    dut = np.multiply(np.multiply(dc_next, cct) + np.multiply(np.multiply(ot, 1 - np.tanh(c_next)**2), np.multiply(cct, da_next)), np.multiply(ut, 1 - ut))
    dcct = np.multiply(np.multiply(dc_next, ut) + np.multiply(np.multiply(ot, 1 - np.tanh(c_next)**2), np.multiply(ut, da_next)), 1 - np.tanh(cct)**2)

    dWot = np.dot(dot, np.vstack((a_prevt, xt)).T)
    dWft = np.dot(dft, np.vstack((a_prevt, xt)).T)
    dWut = np.dot(dut, np.vstack((a_prevt, xt)).T)
    dWct = np.dot(dcct, np.vstack((a_prevt, xt)).T)
    dWy = np.dot(dy, a_next.T)
    dbo = np.sum(dot, axis = 1, keepdims = True)
    dbf = np.sum(dft, axis = 1, keepdims = True)
    dbu = np.sum(dut, axis = 1, keepdims = True)
    dbc = np.sum(dcct, axis = 1, keepdims = True)
    dby = dy

    da_prevt = np.dot(parameters["Wo"].T[: n_a, :], dot) + np.dot(parameters["Wf"].T[: n_a, :], dft) + np.dot(parameters["Wu"].T[: n_a, :], dut) + np.dot(parameters["Wc"].T[: n_a, :], dcct)
    dc_prevt = np.multiply(dc_next, ft) + np.multiply(np.multiply(ot, 1 - np.tanh(c_next)**2), np.multiply(ft, da_next))
    dxt = np.dot(parameters["Wo"].T[n_a :, :], dot) + np.dot(parameters["Wf"].T[n_a :, :], dft) + np.dot(parameters["Wu"].T[n_a :, :], dut) + np.dot(parameters["Wc"].T[n_a :, :], dcct)

    gradients = {"da_prevt": da_prevt, "dc_prevt": dc_prevt, "dxt": dxt, "dWot": dWot, "dWft": dWft, "dWut": dWut,
                    "dWct": dWct, "dWy": dWy, "dbo": dbo, "dbf": dbf, "dbu": dbu, "dbc": dbc, "dby": dby}

    return gradients

def rnn_lstm_full_backward(caches, X, Y):
    (caches, y_hat, a, c, x) = caches
    (a_next, c_next, a_prevt, c_prevt, ft, ut, ot, cct, xt, parameters) = caches[0]

    n_a, _ = a_next.shape
    n_x, _ = xt.shape
    n_y, _ = parameters["by"].shape

    dx = np.zeros((n_x, len(X)))
    da0 = np.zeros((n_a, 1))
    dc0 = np.zeros((n_a, 1))
    da_prev = np.zeros((n_a, 1))
    dc_prev = np.zeros((n_a, 1))
    dWo = np.zeros((n_a, n_a + n_x))
    dWf = np.zeros((n_a, n_a + n_x))
    dWu = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWy = np.zeros((n_y, n_a))
    dbo = np.zeros((n_a, 1))
    dbf = np.zeros((n_a, 1))
    dbu = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dby = np.zeros((n_y, 1))

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        da = np.dot(parameters["Wy"].T, dy)
        gradients = rnn_lstm_cell_backward(caches[t], da_prev + da, dc_prev, dy )
        da_prev, dc_prev, dx = gradients["da_prevt"], gradients["dc_prevt"], gradients["dxt"]
        dWot, dWft, dWut, dWct, dWyt = gradients["dWot"], gradients["dWft"], gradients["dWut"], gradients["dWct"], gradients["dWy"]
        dbot, dbft, dbut, dbct, dbyt = gradients["dbo"], gradients["dbf"], gradients["dbu"], gradients["dbc"], gradients["dby"]
        dWo += dWot
        dWf += dWft
        dWu += dWut
        dWc += dWct
        dWy += dWyt
        dbo += dbot
        dbf += dbft
        dbu += dbut
        dbc += dbct
        dby += dbyt

    da0 = da_prev
    dc0 = dc_prev
    gradients = {"dx": dx, "da_prev": da0, "dc_prev": dc0, "dWo": dWo, "dWf": dWf, "dWu": dWu, "dWc": dWc,
                    "dWy": dWy, "dbo": dbo, "dbf": dbf, "dbu": dbu, "dbc": dbc, "dby": dby}

    return gradients


def update_parameters(gradients, parameters, lr = 0.01):

    parameters["Wo"] -= lr * gradients["dWo"]
    parameters["Wf"] -= lr * gradients["dWf"]
    parameters["Wu"] -= lr * gradients["dWu"]
    parameters["Wc"] -= lr * gradients["dWc"]
    parameters["Wy"] -= lr * gradients["dWy"]
    parameters["bo"] -= lr * gradients["dbo"]
    parameters["bf"] -= lr * gradients["dbf"]
    parameters["bu"] -= lr * gradients["dbu"]
    parameters["bc"] -= lr * gradients["dbc"]
    parameters["by"] -= lr * gradients["dby"]

    return parameters

def optimize(X, Y, a_prev, parameters, vocab_size):
    caches, loss, a = rnn_lstm_full_forward(parameters, a_prev, X, Y, vocab_size)
    gradients = rnn_lstm_full_backward(caches, X, Y)
    gradients = clip(gradients, 5)
    parameters = update_parameters(gradients, parameters)
    return loss, parameters, a[len(X) - 1]

def print_sample(sample_ix, idx_to_word):
    txt = ' '.join(idx_to_word[ix] for ix in sample_ix)
    print ('%s' % (txt, ), end='')

def model (vocab_size, word_to_idx, idx_to_word, specialSymbol, n_a, lr = 0.01, iteration = 1000, seq_length = 100):
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    initial_loss = get_initial_loss(vocab_size, seq_length)

    with open("shakespeare.txt") as f:
        data = f.readlines()
    data = [x.lower().strip() for x in data]
    a_prev = np.zeros((n_a, 1))
    loss_plot = []
    for i in range(iteration):
        index = i % len(data)
        words = nltk.word_tokenize(data[index])

        X = [None] + [word_to_idx[w] for w in words]
        Y = X[1:] + [word_to_idx["\n"]]

        loss, parameters, a_prev = optimize(X, Y, a_prev, parameters, vocab_size)
        if i % 10 == 0:
            for a in range(5):
                sampled_indices = sample(parameters, word_to_idx)
                print_sample(sampled_indices, idx_to_word)
                print("\n")
        print ("Loss value at iteration %d: %f" % (i, loss))

        loss_plot.append(loss)

    plt.plot(loss_plot)
    plt.show()
    return parameters

def sample(parameters, word_to_idx):
    Wo, Wf, Wu, Wc, Wy = parameters["Wo"], parameters["Wf"], parameters["Wu"], parameters["Wc"], parameters["Wy"]
    bo, bf, bu, bc, by = parameters["bo"], parameters["bf"], parameters["bu"], parameters["bc"], parameters["by"]
    vocab_size = by.shape[0]
    n_a = Wo.shape[0]
    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a , 1))
    c_prev = np.zeros((n_a , 1))
    indices = []
    idx = -1
    counter = 0
    newline_character = word_to_idx["\n"]

    while(idx != newline_character and counter != 10):
        cct = np.tanh(np.dot(Wc, np.vstack((a_prev, x))) + bc)
        ut = sigmoid(np.dot(Wu, np.vstack((a_prev, x))) + bu)
        ft = sigmoid(np.dot(Wf, np.vstack((a_prev, x))) + bf)
        ot = sigmoid(np.dot(Wo, np.vstack((a_prev, x))) + bo)
        c = np.multiply(ut, cct) + np.multiply(ft, c_prev)
        a = np.multiply(ot, np.tanh(c))
        y = softmax(np.dot(Wy, a) + by)
        idx = np.random.choice(list(range(vocab_size)), p = y[:,0])
        indices.append(idx)
        counter += 1
        a_prev = a
        c_prev = c
        x = y

    return indices
if __name__ == "__main__":
    # gradients = model(len(dictionary), word_to_idx, idx_to_word, specialSymbol, len(dictionary) + 100, 0.01, 100, 100)
    # with open("shakespeare.txt") as f:
    #     data = f.readlines()
    # data = [x.lower().strip() for x in data]
    # words = nltk.word_tokenize(data[11])
    # print (dictionary)

    # for word in dictionary:
    #     if word == "":
    #         continue
    #     if word[-1] in specialSymbol:


    #         print (dictionary[word_to_idx[word]])

    x = np.arange(10)
    with open("parameters.csv", "w") as out_file:
        for i in range(len(x)):
            out_string = ""
            out_string += str(x[i])
