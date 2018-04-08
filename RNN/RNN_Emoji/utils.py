import csv
import numpy as np
import emoji


def read_glove_vecs(file):
    with open(file, 'r', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.lower().strip().split()
            word = line[0]
            words.add(word)
            word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)

        index = 1
        idx_to_word = {}
        word_to_idx = {}

        for word in sorted(words):
            idx_to_word[index] = word
            word_to_idx[word] = index
            index += 1

        return idx_to_word, word_to_idx, word_to_vec_map

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/np.sum(e_x)

def read_csv(file):
    phrase = []
    emoji = []

    with open(file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X,Y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label): # Converts a label (int or string) into corresponding emoji code(string)
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

def print_prediction(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))

def predict(X, W, b, word_to_vec_map):
    m = X.shape[0]
    pred = np.zeros((m , 1))
    for j in range(m):
        words = X[j].lower().split()
        avg = np.zeros((50,))
        for w in words:
            e_w = word_to_vec_map[w]
            avg += e_w
        avg = avg/len(words)
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)

    # print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))

    return pred
