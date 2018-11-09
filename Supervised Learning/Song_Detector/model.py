import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import random
import sys
import io
import os
import glob
import IPython
from song_preprocessing import *

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import tensorflow as tf

# IPython.display.Audio("./songs/chuyentinhtoi.wav")
# x = graph_spectrogram("./songs/chuyentinhtoi.wav") # (101,156017)
# x_y = graph_spectrogram("./songs/yemsao.wav") # (101, 116171)
# # number of column depends on the duration of the song
# _, data = wavfile.read("./songs/chuyentinhtoi.wav") # (12481536, 2)

def one_step_attention(a, s_prev):
    # shape of a = (None,S,256)
    s_prev = repeator(s_prev) # (None,S,256)
    concat = concatenator([a, s_prev]) # (None,S,512)
    e = densor1(concat) # (None,S,10)
    energies = densor2(e) # (None,S,1)
    alphas = activator(energies) # (None,S,1)
    context = dotor([alphas, a]) # (None,1,1)
    return context


def model(Tx, Ty, n_a, n_s, n_x, n_y):
    X = Input(shape=(Tx, n_x))
    s0 = Input(shape=(n_s,), name = 's0')
    c0 = Input(shape=(n_s,), name = 'c0')
    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences = True))(X) #(None, Tx, n_a*2)

    for t in range(Ty):

        context = one_step_attention(a,s)
        s, _, c = post_activation_LSTM(context, initial_state = [s,c])

        out = output_layer(s)
        outputs.append(out)

    model = Model(inputs=[X,s0,c0], outputs = outputs)
    return model

if __name__ == "__main__":

    S = 1000
    Tx = get_Tx("./songs/")
    Ty = 1

    _, n  = get_songs("./songs/")
    x,y = preprocessing_data("./songs/", Tx, Ty)

    n_a = 128
    n_s = 256
    n_x = 101
    n_y = n

    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    sliding = tf.contrib.data.sliding_window_batch(S)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    activator = Activation(K.softmax, name="attention_weights")
    dotor = Dot(axes= 1)

    post_activation_LSTM = LSTM(n_s, return_state = True)
    output_layer = Dense(n, activation=K.softmax)
    print(x.shape)
    print(y.shape)
    outputs = list(y.swapaxes(0,1))

    model = model(Tx, Ty, n_a, n_s, n_x, n_y)
    model.compile(optimizer = Adam(lr=0.005, beta_1 = 0.9, beta_2 = 0.999, decay =0.1), metrics = ['accuracy'], loss = 'categorical_crossentropy')
    s0 = np.zeros((n, n_s))
    c0 = np.zeros((n, n_s))
    model.fit([x, s0, c0], outputs, epochs = 100, batch_size = 1)
