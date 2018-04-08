import emoji
import numpy as np
from utils import *


X_train, Y_train = read_csv("Data/train_emoji.csv")
X_test, Y_test = read_csv("Data/tesss.csv")
idx_to_word, word_to_idx, word_to_vec_map = read_glove_vecs("Data/glove.6B.50d.txt")


def sentence_to_avg(sentence, word_to_vec_map):
    avg = np.zeros((50,))
    words = sentence.lower().split()
    for word in words:
        avg += word_to_vec_map[word]
    avg = avg/len(words)
    return avg

def model(X, Y, word_to_vec_map, lr = 0.01, num_iteration = 400):
    m = X.shape[0]
    n_y = 5
    n_h = 50

    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    Y_oh = convert_to_one_hot(Y, C = n_y)

    for i in range(num_iteration):
        for j in range(m):
            avg = sentence_to_avg(X[j], word_to_vec_map)

            Z = np.dot(W, avg) + b

            A = softmax(Z)


            cost = -np.sum(Y_oh[j] * np.log(A), axis=0)

            dz = A - Y_oh[j]

            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1,n_h))
            db = dz

            W = W - lr*dW
            b = b - lr*db

        if i % 100 == 0:
            print("Epoch: " + str(i) + " --- cost = " + str(cost))
            pred = predict(X, W, b, word_to_vec_map)

    return pred, W, b

if __name__ == "__main__":
    pred, W, b = model(X_train, Y_train, word_to_vec_map)

    X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
     'Lets go party and drinks','Congrats on the new job','Congratulations',
     'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
     'You totally deserve this prize', 'Let us go play football',
     'Are you down for football this afternoon', 'Work hard play harder',
     'It is suprising how people can be dumb sometimes',
     'I am very disappointed','It is the best day in my life',
     'I think I will end up alone','My life is so boring','Good job',
     'Great so awesome'])

    pred = predict(X, W, b, word_to_vec_map)
    print_prediction(X, pred)
