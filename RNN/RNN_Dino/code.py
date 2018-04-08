import numpy as np
from rnn_utils import *
import random
import matplotlib.pyplot as pyplot

#Dataset and preprocessing
data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

# Map each character to an index from 0-26
char_to_idx = { ch:i for i, ch in enumerate(sorted(chars)) }
idx_to_char = { i:ch for i, ch in enumerate(sorted(chars)) }

def clip(gradients, maxValue):
    dWaa, dWax, dWya, db, dby = gradients["dWaa"], gradients["dWax"], gradients["dWya"], gradients["db"], gradients["dby"]

    for gradient in [dWax, dWaa, dWya, db, dby]:
        gradient = np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients


def sample(parameters, char_to_idx):
    Waa, Wax, Wya, by, b = parameters["Waa"], parameters["Wax"], parameters["Wya"], parameters["by"], parameters["b"]
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size,1))

    a_prev = np.zeros((n_a, 1))

    indices = []

    idx = -1

    counter = 0
    newline_character = char_to_idx["\n"]

    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        idx = np.random.choice(list(range(vocab_size)), p = y[:,0])

        indices.append(idx)

        x = np.eye(vocab_size)[idx,np.newaxis].T

        counter += 1

        a_prev = a

    return indices


# np.random.seed(2)
# _, n_a = 20, 100
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
#
#
# indices = sample(parameters, char_to_idx, 0)
# print("Sampling:")
# print("list of sampled indices:", indices)
# print("list of sampled characters:", [idx_to_char[i] for i in indices])

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_lost * 0.001

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size) * seq_length

def initialize_parameters(n_a, n_x, n_y):
    Wax = np.random.randn(n_a, n_x) * 0.01
    Waa = np.random.randn(n_a, n_a) * 0.01
    Wya = np.random.randn(n_y, n_a) * 0.01
    b = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by }
    return parameters

def rnn_cell_forward(parameters, a_prev, xt):
    Waa, Wax, Wya, b, by = parameters["Waa"], parameters["Wax"], parameters["Wya"], parameters["b"], parameters["by"]
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + b)
    yt = softmax(np.dot(Wya, a_next) + by)
    return a_next, yt

# def rnn_cell_backward(dyt, da_next, parameters, a_prev, xt, at, dby, db):
#     Waa, Wax, Wya, b, by = parameters["Waa"], parameters["Wax"], parameters["Wya"], parameters["b"], parameters["by"]
#
#     da = np.dot(Wya.T, dyt) + da_next
#     # dtanh = np.multiply(da_next, 1-np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + b)**2)
#     dtanh = (1 - at*at) * da
#     dWya = np.dot(dyt, at.T)
#     dWax = np.dot(dtanh, xt.T)
#     dWaa = np.dot(dtanh, a_prev.T)
#     dby += dyt
#     db += dtanh
#     da_next = np.dot(Waa.T, dtanh)
#     gradients = {"dWya": dWya, "dWax": dWax, "dWaa": dWaa, "db": db, "dby": dby, "da_next": da_next}
#
#     return gradients


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters["Wax"] += -lr * gradients["dWax"]
    parameters["Waa"] += -lr * gradients["dWaa"]
    parameters["Wya"] += -lr * gradients["dWya"]
    parameters["b"] += -lr * gradients["db"]
    parameters["by"] += -lr * gradients["dby"]

    return parameters

def rnn_forward(X, Y, a0, parameters, vocab_size = 27):
    x, a, y_hat = {}, {}, {}

    a[-1] = np.copy(a0)

    loss = 0

    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if (X[t] != None):
            x[t][X[t]] = 1

        a[t], y_hat[t] = rnn_cell_forward(parameters, a[t-1], x[t])

        loss -= np.log(y_hat[t][Y[t]])

    cache = (y_hat, a, x)

    return loss, cache

# def rnn_backward(X, Y, parameters, cache):
#     (y_hat, a, x) = cache
#     Waa, Wax, Wya, b, by = parameters["Waa"], parameters["Wax"], parameters["Wya"], parameters["b"], parameters["by"]
#
#     dWax, dWaa, dWya = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
#     db, dby = np.zeros_like(b), np.zeros_like(by)
#     da_next = np.zeros_like(a[0])
#
#     for t in reversed(range(len(X))):
#         dyt = np.copy(y_hat[t])
#         dyt[Y[t]] -= 1
#         gradients = rnn_cell_backward(dyt, da_next, parameters, a[t-1], x[t], a[t], dby, db)
#         dWaxt, dWaat, dWyat, dbt, dby, da_next = gradients["dWax"], gradients["dWaa"], gradients["dWya"], gradients["db"], gradients["dby"], gradients["da_next"]
#         dWax += dWaxt
#         dWaa += dWaat
#         dWya += dWyat
#         db += dbt
#
#     return gradients, a

def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}

    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    ### END CODE HERE ###

    return gradients, a

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print ('%s' % (txt, ), end='')

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

    loss, cache = rnn_forward(X, Y, a_prev, parameters)

    gradients, a = rnn_backward(X, Y, parameters, cache)

    gradients = clip(gradients, 5)

    parameters = update_parameters(parameters, gradients,learning_rate)
    return loss, gradients, a[len(X) - 1]

# vocab_size, n_a = 27, 100
# a_prev = np.random.randn(n_a, 1)
# Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
# b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
# parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
# X = [12,3,5,11,22,3]
# Y = [4,14,11,22,25, 26]
#
# loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
# print("Loss =", loss)
# print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
# print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
# print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
# print("gradients[\"db\"][4] =", gradients["db"][4])
# print("gradients[\"dby\"][1] =", gradients["dby"][1])
# print("a_last[4] =", a_last[4])

def model(data, idx_to_char, char_to_idx, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, dino_names)
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    np.random.shuffle(examples)

    a_prev = np.zeros((n_a, 1))

    for j in range(num_iterations):

        index = j % len(examples)
        X = [None] + [char_to_idx[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_idx["\n"]]



        loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        if j % 2000 == 0:
            print ("Iteration: %d, Loss: %f" % (j, loss) + "\n")

            for name in range(dino_names):
                sampled_indices = sample(parameters, char_to_idx)
                print_sample(sampled_indices, idx_to_char)

            print("\n")


    return parameters


if __name__ == "__main__":
    parameters = model(data, idx_to_char, char_to_idx)
