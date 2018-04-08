import numpy as np
from rnn_utils import *

def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters["Wax"] # (n_a, n_x)
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

def rnn_forward(x, a0, parameters):
    # Tx = Ty
    # a0.shape is (n_a, m)
    caches = []
    # Get the shapes from x
    n_x, m, Tx = x.shape # n_x: length of vector x (one-hot); m: number of training set; Tx: number of words
    n_y, n_a = parameters["Wya"].shape

    # Initialize a for 1 sequence (n_a, m, Tx)
    a = np.zeros((n_a, m, Tx))
    # Initialize y_pred for 1 sequence (n_y, m, Tx)
    y_pred = np.zeros((n_y, m, Tx))

    # a_next.shape is (n_a, m)
    a_next = a0

    #loop through every word in a sequence (time-step)
    for t in range(Tx):
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)

    # caches for backwardpropagation
    caches = (caches, x)

    return a, y_pred, caches

# ------------------------------------------------------------------------------
# Basic RNN test
# np.random.seed(1)
# x = np.random.randn(3,10,4)
# a0 = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#
# a, y_pred, caches = rnn_forward(x, a0, parameters)
# print("a[4][1] = ", a[4][1])
# print("a.shape = ", a.shape)
# print("y_pred[1][3] =", y_pred[1][3])
# print("y_pred.shape = ", y_pred.shape)
# print("caches[1][1][3] =", caches[1][1][3])
# print("len(caches) = ", len(caches))
# ------------------------------------------------------------------------------
#LSTM
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    # a_prev.shape is (n_a, m)
    # c_prev.shape is (n_a, m)
    # xt.shape is (n_x, m)
    # get parameters from parameters
    #-------------------------------
    # Wf.shape is (n_a, n_a + n_x)
    Wf = parameters["Wf"]
    # bf.shape is (n_a, 1)
    bf = parameters["bf"]
    # Wi.shape is (n_a, n_a + n_x)
    Wi = parameters["Wi"]
    # bi.shape is (n_a, 1)
    bi = parameters["bi"]
    # Wc.shape is (n_a, n_a + n_x)
    Wc = parameters["Wc"]
    # bc.shape is (n_a, 1)
    bc = parameters["bc"]
    # Wo.shape is (n_a, n_a + n_x)
    Wo = parameters["Wo"]
    # bo.shape is (n_a, 1)
    bo = parameters["bo"]
    # Wy.shape is (n_y, n_a)
    Wy = parameters["Wy"]
    # bi.shape is (n_y, 1)
    by = parameters["by"]

    # Concat a_prev and xt using np.vstack
    # shape of concat is (n_a + n_x, m)
    concat = np.vstack((a_prev, xt))

    # calculate gates (update, forgot, output)
    ft = sigmoid(np.dot(Wf, concat) + bf) # (n_a, m)
    it = sigmoid(np.dot(Wi, concat) + bi) # (n_a, m)
    cct = np.tanh(np.dot(Wc, concat) + bc) # (n_a, m)
    # c_next.shape is (n_a, m)
    c_next = np.multiply(it, cct) + np.multiply(ft, c_prev)
    ot = sigmoid(np.dot(Wo, concat) + bo) # (n_a ,m)
    a_next = np.multiply(ot, np.tanh(c_next)) # (n_a, m)

    # Predict y
    yt_pred = softmax(np.dot(Wy, a_next) + by) # (n_y, m)

    # Cache values in cache for backwardpropagation
    cache = (a_next, c_next, a_prev, c_prev, ft, it, ot, cct, xt, parameters)

    return a_next, c_next, yt_pred, cache
# ------------------------------------------------------------------------------
#LSTM cell test
# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", c_next.shape)
# print("c_next[2] = ", c_next[2])
# print("c_next.shape = ", c_next.shape)
# print("yt[1] =", yt[1])
# print("yt.shape = ", yt.shape)
# print("cache[1][3] =", cache[1][3])
# print("len(cache) = ", len(cache))
# ------------------------------------------------------------------------------
# Only have a0 because a0 = c0
def lstm_forward(x, a0, parameters):
    # x.shape is (n_x, m, T_x)
    # a0.shape is (n_a, m)
    #Initialize caches
    caches = []
    # Receive shape from x
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape
    # Initialize a, c, y
    # a.shape is (n_a, m, T_x)
    a = np.zeros((n_a, m, T_x))
    # c.shape is (n_c, m, T_x)
    c = np.zeros((n_a, m, T_x)) # n_a = n_c
    # y.shape is (n_y, m, T_x)
    y = np.zeros((n_y, m, T_x)) # T_x = T_y

    #Initialize a_next and c_next
    a_next = a0 # (n_a, m)
    c_next = a0 # (n_a , m)

    # Loop through every word in an sequence
    for t in range(T_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
        a[:,:,t] = a_next
        c[:,:,t] = c_next
        y[:,:,t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, c, t, caches

# ------------------------------------------------------------------------------
#LSTM test
# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a, y, c, caches = lstm_forward(x, a0, parameters)
# print("a[4][3][6] = ", a[4][3][6])
# print("a.shape = ", a.shape)
# print("y[1][4][3] =", y[1][4][3])
# print("y.shape = ", y.shape)
# print("caches[1][1[1]] =", caches[1][1][1])
# print("c[1][2][1]", str(c[1][2][1]))
# print("len(caches) = ", len(caches))
# ------------------------------------------------------------------------------
# Basic RNN backwardpropagation

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache

    Waa = parameters["Waa"] # (n_a, n_a)
    Wax = parameters["Wax"] # (n_x, n_a)
    Wya = parameters["Wya"] # (n_y, n_a)
    ba = parameters["ba"] # (n_a, m)
    by = parameters["by"] # (n_y, m)

    # Compute the gradient of tanh with respect to a_next
    dtanh = np.multiply(da_next, 1-np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)**2)

    # Compute the gradient of the loss with respect to Wax
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # Compute the gradient of the loss with respect to Waa
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # Compute the gradient of the loss with respect to b
    dba = np.sum(dtanh, axis=1)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

def rnn_backward(da, caches):
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dWaa = np.zeros((n_a, n_a))
    dWax = np.zeros((n_a, n_x))
    dx = np.zeros((n_x, m, T_x))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        da_next = da_prevt + da[:,:,t]
        gradients = rnn_cell_backward(da_next, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:,:,t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, ot, cct, xt, parameters) = cache
    n_x, m = xt.shape
    n_a, m = a_next.shape

    dot = np.multiply(np.multiply(da_next, np.tanh(c_next)), np.multiply(ot, 1-ot))
    dit = np.multiply(np.multiply(dc_next, cct) + np.multiply(np.multiply(ot, 1-np.tanh(c_next)**2), np.multiply(cct, da_next)), np.multiply(it, 1-it))
    dft = np.multiply(np.multiply(dc_next, c_prev) + np.multiply(np.multiply(ot, 1-np.tanh(c_next)**2), np.multiply(c_prev, da_next)), np.multiply(ft, 1-ft))
    dcct = np.multiply(np.multiply(dc_next, it) + np.multiply(np.multiply(ot, 1-np.tanh(c_next)**2), np.multiply(it, da_next)), 1-np.tanh(cct)**2)

    dWf = np.dot(dft, np.vstack((a_prev, xt)).T)
    dWi = np.dot(dit, np.vstack((a_prev, xt)).T)
    dWo = np.dot(dot, np.vstack((a_prev, xt)).T)
    dWc = np.dot(dcct, np.vstack((a_prev, xt)).T)
    dbf = np.sum(dft, axis = 1, keepdims = True)
    dbi = np.sum(dit, axis = 1, keepdims = True)
    dbo = np.sum(dot, axis = 1, keepdims = True)
    dbc = np.sum(dcct, axis = 1, keepdims = True)

    da_prev = np.dot(parameters["Wf"].T[: n_a, :], dft) + np.dot(parameters["Wi"].T[: n_a, :], dit) + np.dot(parameters["Wc"].T[: n_a, :], dcct) + np.dot(parameters["Wo"].T[: n_a, :], dot)
    dc_prev = np.multiply(dc_next, ft) + np.multiply(np.multiply(ot, 1-np.tanh(c_next)**2), np.multiply(ft, da_next))
    dxt = np.dot(parameters["Wf"].T[n_a :, :], dft) + np.dot(parameters["Wi"].T[n_a :, :], dit) + np.dot(parameters["Wc"].T[n_a :, :], dcct) + np.dot(parameters["Wo"].T[n_a :, :], dot)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dWi": dWi, "dWo": dWo, "dWc": dWc, "dbf": dbf, "dbi": dbi, "dbo": dbo, "dbc": dbc}

    return gradients


# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# c_prev = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
#
# da_next = np.random.randn(5,10)
# dc_next = np.random.randn(5,10)
# gradients = lstm_cell_backward(da_next, dc_next, cache)
# print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
# print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
# print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
# print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
# print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
# print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
# print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
# print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
# print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
# print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
# print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
# print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
# print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
# print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
# print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
# print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
# print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
# print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
# print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
# print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
# print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
# print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)


def lstm_backward(da, caches):
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da_prevt + da[:,:,t], dc_prevt, caches[t])
        dxt, da_prevt, dc_prevt, dWft, dWit, dWot, dWct, dbft, dbit, dbct, dbot = gradients["dxt"], gradients["da_prev"], gradients["dc_prev"], gradients["dWf"], gradients["dWi"], gradients["dWo"], gradients["dWc"], gradients["dbf"], gradients["dbi"], gradients["dbc"], gradients["dbo"]
        dx[:,:,t] = dxt
        dWf += dWft
        dWi += dWit
        dWc += dWct
        dWo += dWot
        dbf += dbft
        dbi += dbit
        dbc += dbct
        dbo += dbot

    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients



np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)

da = np.random.randn(5, 10, 4)
gradients = lstm_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)
