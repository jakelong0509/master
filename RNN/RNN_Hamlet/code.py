import numpy as np
import matplotlib.pyplot as plt
import wikipedia
from utils import *
from tqdm import tqdm
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

training_set, max_length = load_data()
word_embedding, word_to_idx, idx_to_word = load_embedding_word("glove.6B.50d.txt")

def sentence_to_indices(X, word_to_idx):
    m = len(X)
    sentence_indices = np.zeros((m, max_length))

    for i in tqdm(range(m)):
        words = X[i]
        index = 0

        for word in words:
            if word not in word_to_idx or word == None:
                sentence_indices[i][index] = 0
            else:
                sentence_indices[i][index] = word_to_idx[word]
            index += 1

    return sentence_indices

def pretrained_embedding(word_embedding, word_to_idx):
    vocab_size = len(word_to_idx) + 1
    em_dim = word_embedding["chicken"].shape[0]
    emb_matrix = np.zeros((vocab_size, em_dim))
    for word, index in word_to_idx.items():
        num_of_0 = 0
        if (word_embedding[word].shape[0] < em_dim):
            num_of_0 = em_dim - word_embedding[word].shape[0]
        emb_matrix[index, :] = np.append(word_embedding[word], np.array(["0"] * num_of_0, dtype=np.float64))

    embedding_layer = Embedding(vocab_size, em_dim, input_length = 100, trainable = True)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def main(input_shape, word_embedding, word_to_idx):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)
    embedding_layer = pretrained_embedding(word_embedding, word_to_idx)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation="softmax", input_shape=(input_shape[0],))(X)

    model = Model(sentence_indices, X)

    return model

if __name__ == "__main__":
    model = main((max_length,), word_embedding, word_to_idx)
    model.compile(optimizer="adam", loss="mean_squared_logarithmic_error", metrics=['accuracy'])
    X = training_set
    for i in range(len(X)):
        X[i] = [None] + X[i]
    X = sentence_to_indices(X, word_to_idx)
    Y = training_set
    for i in range(len(Y)):
        Y[i] = Y[i][1:] + ["\n"]
    Y = sentence_to_indices(Y, word_to_idx)
    model.fit(X, Y, batch_size=128, epochs=20, shuffle=True)
    string = ["Barack", "Obama", "", "", "", "", "", "", ""]
    string = sentence_to_indices(string, word_to_idx)
    prediction = model.predict(string)
    sentence = ""
    for i in range(len(prediction)):
        sentence = sentence + " " + idx_to_word[i]

    print (sentence)
