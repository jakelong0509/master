import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from song_preprocessing import *

def one_step_attention(a, s_prev):
    


if __name__ == "__main__":
    S = 1000
    Tx = get_Tx("./songs/")
    Ty = 101

    _,n = get_songs("./songs/")
    x,y = preprocessing_data("./songs/", Tx, Ty)
    n_a = 128
    n_s = 256
    n_x = 101
    n_y = n
