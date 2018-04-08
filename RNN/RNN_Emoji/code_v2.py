import numpy as np
from utils import *
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

maxLen = 10
X_train, Y_train = read_csv("Data/train_emoji.csv")
X_test, Y_test = read_csv("Data/tesss.csv")
idx_to_word, word_to_idx, word_to_vec_map = read_glove_vecs("Data/glove.6B.50d.txt")

def sentence_to_indices(X, word_to_idx, max_len):
    m = X.shape[0]
    sentence_indices = np.zeros((m, max_len))

    for j in range(m):
        words = X[j].lower().split()
        index = 0
        for word in words:
            sentence_indices[j][index] = word_to_idx[word]
            index += 1

    return sentence_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_idx):
    vocab_len = len(word_to_idx) + 1 # adding 1 to fit keras embeddings (Mandatory)
    emb_dim = word_to_vec_map["cucumber"].shape[0] # 50

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_idx.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_idx):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_idx)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(embeddings)

    X = Dropout(0.5)(X)

    X = LSTM(128, return_sequences=False)(X)

    X = Dropout(0.5)(X)

    X = Dense(5)(X)

    X = Activation("softmax")(X)

    model = Model(sentence_indices, X)

    return model

if __name__ == "__main__":
    # X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    # X1_indices = sentence_to_indices(X1,word_to_idx, max_len = 5)
    # print("X1 =", X1)
    # print("X1_indices =", X1_indices)
    # print(len(word_to_idx))

    # embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_idx)
    # print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


    model = Emojify_V2((maxLen,), word_to_vec_map, word_to_idx)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train_indices = sentence_to_indices(X_train, word_to_idx, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=5)
    model.fit(X_train_indices, Y_train_oh, epochs = 100, batch_size=128, shuffle=True)
    X_test_indices = sentence_to_indices(X_test, word_to_idx, maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test Accuracy = ", acc)

    # Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.
    x_test = np.array(["i want some coffee"])
    X_test_indices = sentence_to_indices(x_test, word_to_idx, maxLen)
    print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
