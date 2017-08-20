from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import random as rd

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_file = 'test1.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    #content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 500
n_input = 3

# number of units in RNN cell
n_hidden = 512


# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])
# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
lookTab = {"r": 0, "s": 1, "p": 2}
[r, s, p] = ["r", "s", "p"]
actions = [r, s, p]
alpha = 1.1
upbound = 5000
[p11, p12, p13] = [0.001, 0.004, 0.995]  # w-, w+, w0
[p21, p22, p23] = [0.063, 0.791, 0.146]  # t-, t+, t0
[p31, p32, p33] = [0.989, 0.001, 0.01] # l-, l+, l0
(win, tie, lose) = (1, 0, -1)

sess = tf.Session()
saver.restore(sess, "/tmp/model.ckpt")
print('model restored')
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
    elif (action1 == r and action2 == s)or\
         (action1 == s and action2 == p)or\
         (action1 == p and action2 == r):
        return win, lose
    else:
        return lose, win





class Player:
    def __init__(self):
        self.state = tie
        self.action = s

    def getAction(self, opposite_action):
        idx = lookTab[self.action]
        pr = [1, 2, 3]
        if self.state == win:
            pr[idx] = p13
            pr[(idx+1)%3] = p11
            idx -= 1
            if idx < 0:
                idx = 2
            pr[idx] = p12
        elif self.state == tie:
            pr[idx] = p23
            pr[(idx + 1) % 3] = p21
            idx -= 1
            if idx - 1 < 0:
                idx = 2
            pr[idx] = p22
        else: #self.state == lose
            pr[idx] = p33
            pr[(idx + 1) % 3] = p31
            idx -= 1
            if idx - 1 < 0:
                idx = 2
            pr[idx] = p32
        return generateAction(pr[0], pr[1], pr[2])


    def setAction(self, new_action):
        self.action = new_action

class AI(Player):
    def __init__(self):
        Player.__init__(self)
        self.time = 0
        self.lst = []
    
    def getAction(self, opposite_action):
        if self.time < 3:
            self.time += 1
            self.lst.append(opposite_action + str(-self.state))
            return Player.getAction(self, opposite_action)
        else:
            words = self.lst
            #print(words)
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            keys = np.reshape(np.array(symbols_in_keys), [-1, 3, 1])
            onehot_pred = sess.run(pred, feed_dict={x: keys})
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval(session=sess))
            symbols_in_keys = symbols_in_keys[1:]
            symbols_in_keys.append(onehot_pred_index)
            self.lst = self.lst[1:]
            self.lst.append(opposite_action + str(-self.state))
            res =  reverse_dictionary[onehot_pred_index][0]
            assert(res in actions)
            if res == 'r':
                return 'p'
            elif res == 's':
                return 'r'
            else: # res == 'p'
                return 's'
            return res
        
p1 = Player()
p2 = AI()
action1 = generateAction(1/3, 1/3, 1/3)
action2 = generateAction(1/3, 1/3, 1/3)
p1.setAction(action1)
p2.setAction(action2)
(state1, state2) = getRes(action1, action2)
p1.state = state1
p2.state = state2
f = open('./test4.txt', 'w+')#'C:\Users\lenovo\Documents\SRTP\PSR_AI\testcase\test.txt', 'w')
action1s = []
action2s = []
action1 = action2 = None
for i in range(upbound):
    action1 = p1.getAction(action2)
    action2 = p2.getAction(action1)
    p1.setAction(action1)
    p2.setAction(action2)
    #print("action(p1,p2): ",action1," ",action2, "\n")
    action1s.append(action1)
    action2s.append(action2)
    #f.write(action1 + ' ' + action2 + ' ')
    (state1, state2) = getRes(action1, action2)
    p1.state = state1
    p2.state = state2
for ac1, ac2 in zip(action1s, action2s):
    f.write(ac1 + str(getRes(ac1, ac2)[0]) + '\n')#ac2 + '\n')#+ ' ' + 
f.close()
print('ok')