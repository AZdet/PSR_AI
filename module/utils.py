import random as rd
import numpy as np

lookTab = {"r": 0, "s": 1, "p": 2}
[r, s, p] = ["r", "s", "p"]
actions = [r, s, p]
(win, tie, lose) = (1, 0, -1)


def generateAction(p1, p2, p3):
    n = rd.random()
    if n < p1:
        return actions[0]
    elif n < p1 + p2:
        return actions[1]
    else:
        return actions[2]


def getRes(action1, action2):
    if action1 == action2:
        return (tie, tie)
    elif (action1 == r and action2 == s) or \
            (action1 == s and action2 == p) or \
            (action1 == p and action2 == r):
        return win, lose
    else:
        return lose, win


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def one_hots(numerical_list, vocab_size):
    result = np.zeros((len(numerical_list), vocab_size))
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result


def tokenize(dic, path):
    """Tokenizes a text file."""
    # Add words to the dictionary
    with open(path, 'r') as f:  # in my case, only one word each line
        tokens = 0
        for line in f:
            # words = line.split() + ['<eos>']
            tokens += 1  # len(words)
            # for word in words:
            # self.dictionary.add_word(word)
            dic.add_word(line)

    # Tokenize file content
    with open(path, 'r') as f:
        ids = np.zeros((tokens,), dtype='int32')
        token = 0
        for line in f:
            # words = line.split() + ['<eos>']
            # for word in words:
            # ids[token] = self.dictionary.word2idx[word]
            # token += 1
            ids[token] = dic.word2idx[line]
            token += 1

    return ids


def tokenize_list(lst, data_dict):
    tokens = lst.len
    ids = np.zeros((tokens,), dtype='int32')
    token = 0
    for ele in lst:
        ids[token] = data_dict.word2idx[ele]
        token += 1
    return ids
