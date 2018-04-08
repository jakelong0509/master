import numpy as np

class utils:
    def __init__(self):
        self.words = set()
        self.word_to_vec_map = {}
        self.softmax = 0
        self.s = 0

    def read_glove_vecs(self, file):
        with open(file, 'r', encoding = 'utf-8') as f:

            for line in f:
                line = line.strip().split()
                word = line[0]
                self.words.add(word)
                self.word_to_vec_map[word] = np.array(line[1:], dtype = np.float64)

    def softmax(self, z):
        e_x = np.exp(z - np.max(z))
        self.softmax = e_x/e_x.sum()

    def relu(self, x):
        self.s = np.maximum(0,x)

def cosine_similarity(u,v):
    dot = np.dot(u,v)
    u_l2 = np.sqrt(np.sum(u**2, axis=0))
    v_l2 = np.sqrt(np.sum(v**2, axis=0))
    cosine = dot/(u_l2 * v_l2)

    return cosine

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):

    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    best_cosine = -100
    best_word = None
    words = word_to_vec_map.keys()

    for w in words:
        if w in [word_a, word_b, word_c]:
            continue

        cosine_value = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)

        if cosine_value > best_cosine:
            best_cosine = cosine_value
            best_word = w

    return best_word, best_cosine

def neutralize(word, g, word_to_vec_map):
    e_word = word_to_vec_map[word]
    e_bias_component = np.multiply(np.dot(e_word, g)/(np.sqrt(np.sum(g**2)))**2, g)
    e_debiased = e_word - e_bias_component

    return e_debiased

def equalize(pair, bias_axis, word_to_vec_map):
    w1,w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    mu = (e_w1 + e_w2) / 2

    mu_B = np.multiply(np.dot(mu, bias_axis) / np.sum(bias_axis**2), bias_axis)
    mu_orth = mu - mu_B

    e_w1B = np.multiply(np.dot(e_w1, bias_axis) / np.sum(bias_axis**2), bias_axis)
    e_w2B = np.multiply(np.dot(e_w2, bias_axis) / np.sum(bias_axis**2), bias_axis)

    e_corrected_w1B = np.multiply(np.sqrt(np.abs(1 - np.sum(mu_orth**2))), (e_w1B - mu_B) / np.sqrt(np.sum(((e_w1 - mu_orth) - mu_B)**2)))
    e_corrected_w2B = np.multiply(np.sqrt(np.abs(1 - np.sum(mu_orth**2))), (e_w2B - mu_B) / np.sqrt(np.sum(((e_w2 - mu_orth) - mu_B)**2)))

    e1 = e_corrected_w1B + mu_orth
    e2 = e_corrected_w2B + mu_orth

    return e1, e2

if __name__ == "__main__":
    utils = utils()
    utils.read_glove_vecs("glove.6B.50d.txt")
    # _______________Test consine_similarity function________________

    # father = utils.word_to_vec_map["father"]
    # mother = utils.word_to_vec_map["mother"]
    # ball = utils.word_to_vec_map["ball"]
    # crocodile = utils.word_to_vec_map["crocodile"]
    # france = utils.word_to_vec_map["france"]
    # italy = utils.word_to_vec_map["italy"]
    # paris = utils.word_to_vec_map["paris"]
    # rome = utils.word_to_vec_map["rome"]
    #
    # print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
    # print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
    # print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))


    # ________________Test complete_analogy function________________
    # triads_to_try = [('apple', 'company', 'orange'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
    # for triad in triads_to_try:
    #     print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,utils.word_to_vec_map)))

    g = utils.word_to_vec_map["woman"] - utils.word_to_vec_map["man"]

    # girls and boys name
    # name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
    #
    # for w in name_list:
    #     print (w, cosine_similarity(utils.word_to_vec_map[w], g))

    # print('Other words and their similarities:')
    # word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',
    #          'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
    # for w in word_list:
    #     print (w, cosine_similarity(utils.word_to_vec_map[w], g))

    # e = "receptionist"
    # print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(utils.word_to_vec_map["receptionist"], g))
    #
    # e_debiased = neutralize("receptionist", g, utils.word_to_vec_map)
    # print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))

    print("cosine similarities before equalizing:")
    print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(utils.word_to_vec_map["man"], g))
    print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(utils.word_to_vec_map["woman"], g))
    print()
    e1, e2 = equalize(("man", "woman"), g, utils.word_to_vec_map)
    print("cosine similarities after equalizing:")
    print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
    print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
