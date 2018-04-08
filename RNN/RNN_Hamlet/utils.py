import numpy as np
import nltk
import wikipedia
from tqdm import tqdm


def load_embedding_word(file):
    with open(file, "r", encoding='utf-8') as f:
        words = set()
        word_embedding = {}
        count = 0
        # for line in f:
        #     if count == 52343:
        #         line = line.strip().split(" ")
        #         word = line[0]
        #         print (word)
        #         print (line[301])
        #     count += 1

        for line in tqdm(f):
            line = line.strip().split(" ")
            word = line[0]
            words.add(word)
            word_embedding[word] = np.array(line[1:], dtype=np.float64)
            count += 1
        word_embedding["\n"] = np.array(np.zeros((50,)), dtype=np.float64)
        word_to_idx = {}
        idx_to_word = {}
        index = 0
        for word in words:
            word_to_idx[word] = index
            idx_to_word[index] = word
            index += 1
        word_to_idx["\n"] = index
        idx_to_word[index] = "\n"
        return word_embedding, word_to_idx, idx_to_word

def load_data():
    training_set = []
    max_length = 0
    news = wikipedia.page("Presidency of Barack Obama")
    news = news.content.strip().lower().split("\n")
    for line in news:
        token = nltk.word_tokenize(line)
        length = len(token)
        if length > max_length:
            max_length = length
        training_set.append(token)
    max_length += 1
    return training_set, max_length

# def convert_to_one_hot(Y, C):
#     return np.eye(C)[Y.reshape(-1)]

# if __name__=='__main__':
#
#     word_embedding, word_to_idx, idx_to_word = load_embedding_word("glove.840B.300d.txt")
#     training_set, max_length = load_data()
#     print (word_embedding["chicken"].shape[0])
