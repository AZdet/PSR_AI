
# coding: utf-8

# In[2]:

import keras
from keras.models import Sequential
from keras.layers import *


# In[3]:

import numpy as np


# In[120]:

time_step = 10
BATCH_SIZE = 32
EPOCH = 10


# In[116]:

model = Sequential()
model.add(Embedding(9 , 9, input_length= 1))
model.add(GRU(5, input_shape = (None, 1)))
model.add(Dense(3))
model.add(Activation("softmax"))


# In[117]:

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[26]:

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


# In[27]:

data_dict = Dictionary()
def tokenize(dic, path):
        """Tokenizes a text file."""
        # Add words to the dictionary
        with open(path, 'r') as f:    # in my case, only one word each line
            tokens = 0
            for line in f:
                #words = line.split() + ['<eos>']
                tokens += 1#len(words)
                #for word in words:
                    #self.dictionary.add_word(word)
                dic.add_word(line)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros((tokens,), dtype='int32')
            token = 0
            for line in f:
                #words = line.split() + ['<eos>']
                #for word in words:
                    #ids[token] = self.dictionary.word2idx[word]
                    #token += 1
                ids[token] = dic.word2idx[line]
                token += 1

        return ids


# In[96]:

"""make sure that data_dict is initialized"""
def tokenize_list(lst):
    tokens = lst.len
    ids = np.zeros((tokens,), dtype='int32')
    token = 0
    for ele in lst:
        ids[token] = data_dict.word2idx[ele]
        token += 1
    return ids


# In[ ]:




# In[28]:

data1 = tokenize(data_dict, 'test6.txt')


# In[154]:

data_dict.word2idx.items()


# In[155]:

data_dict.idx2word


# In[37]:

data = one_hots(data1[0:30] ,9)


# In[81]:

data2 = data1[:5000]


# In[92]:

data1


# In[32]:

label_dict = Dictionary()
label1 = tokenize(label_dict, 'label2.txt')


# In[153]:

label_dict.idx2word


# In[86]:

label = one_hots(label1, 3)


# In[137]:

label


# In[161]:

test_data = tokenize(data_dict, 'test7.txt')
label2 = tokenize(label_dict, 'label3.txt')
test_label = one_hots(label2, 3)


# In[ ]:




# In[158]:

model.fit(data[:-1], label, batch_size = BATCH_SIZE, epochs = EPOCH)


# In[122]:

test_data = data1[5000:-1]
test_label = label[5000:]


# In[162]:

model.evaluate(test_data[:-1], test_label)


# In[159]:

model.predict(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))


# In[127]:

np.argmax(model.predict(np.array([1])))


# In[129]:

import random as rd


# In[ ]:

lookTab = {"r": 0, "s": 1, "p": 2}
[r, s, p] = ["r", "s", "p"]
actions = [r, s, p]
alpha = 1.1
upbound = 5000
[p11, p12, p13] = [0.001, 0.004, 0.995]  # w-, w+, w0
[p21, p22, p23] = [0.063, 0.791, 0.146]  # t-, t+, t0
[p31, p32, p33] = [0.989, 0.001, 0.01] # l-, l+, l0
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
    elif (action1 == r and action2 == s)or         (action1 == s and action2 == p)or         (action1 == p and action2 == r):
        return win, lose
    else:
        return lose, win




class Player:
    def __init__(self):
        self.state = tie
        self.action = s

    def getAction(self):
        idx = lookTab[self.action]
        pr = [1/3, 1/3, 1/3]
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
            if idx < 0:
                idx = 2
            pr[idx] = p22
        else: #self.state == lose
            pr[idx] = p33
            pr[(idx + 1) % 3] = p31
            idx -= 1
            if idx < 0:
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
        
    def receiveHistory(self, history):
        self.lst.append(history)
        
    def getAction(self):
        if self.time < 1:
            print("first time")
            self.time += 1
            #self.lst.append(opposite_action + str(-self.state) + '\n')
            return generateAction(1/3, 1/3, 1/3)
        else:
            #last_action = np.array([data_dict.word2idx[self.lst[-1]]])
            #last_state = opposite_action + str(-self.state) + '\n'
            last_data = np.array([data_dict.word2idx[self.lst[-1]]])
            idx = np.argmax(model.predict(last_data))
            res = label_dict.idx2word[idx]
            #print(words)
            #symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            #keys = np.reshape(np.array(symbols_in_keys), [-1, 3, 1])
            #onehot_pred = sess.run(pred, feed_dict={x: keys})
            #onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval(session=sess))
            #symbols_in_keys = symbols_in_keys[1:]
            #symbols_in_keys.append(onehot_pred_index)
            #self.lst = self.lst[1:]
            #self.lst.append(opposite_action + str(-self.state) + '\n')
            #res =  reverse_dictionary[onehot_pred_index][0]
            
            #assert(res in actions)
            if res == 'r\n':   # shift the predicted action to win the game 
                return 'p'
            elif res == 's\n':
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
p2.receiveHistory(action1 + str(state1) + '\n')
f = open('./res.txt', 'w+')#'C:\Users\lenovo\Documents\SRTP\PSR_AI\testcase\test.txt', 'w')
action1s = []
action2s = []
#action1 = action2 = None
stat = {-1: 0, 0: 0, 1: 0}
for i in range(upbound):
    action1 = p1.getAction()
    action2 = p2.getAction()
    p1.setAction(action1)
    p2.setAction(action2)
    #print("action(p1,p2): ",action1," ",action2, "\n")
    action1s.append(action1)
    action2s.append(action2)
    #f.write(action1 + ' ' + action2 + ' ')
    (state1, state2) = getRes(action1, action2)
    stat[state1] += 1
    p1.state = state1
    p2.state = state2
    p2.receiveHistory(action1 + str(state1) + '\n')
#count = 0
#for ac1, ac2 in zip(action1s, action2s):
#    f.write(ac1 + ' ' + ac2 + ' ' + str(getRes(ac1, ac2)[0]) + '\n')#ac2 + '\n')#+ ' ' + 
#    count += getRes(ac1, ac2)[0]
#f.close()
#print(count)
print(stat.items())
print('ok')


# In[ ]:

lookTab = {"r": 0, "s": 1, "p": 2}
[r, s, p] = ["r", "s", "p"]
actions = [r, s, p]
alpha = 1.1
upbound = 5000
[p11, p12, p13] = [0.001, 0.004, 0.995]  # w-, w+, w0
[p21, p22, p23] = [0.063, 0.791, 0.146]  # t-, t+, t0
[p31, p32, p33] = [0.989, 0.001, 0.01] # l-, l+, l0
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
    elif (action1 == r and action2 == s)or         (action1 == s and action2 == p)or         (action1 == p and action2 == r):
        return win, lose
    else:
        return lose, win




class Player:
    def __init__(self):
        self.state = tie
        self.action = s

    def getAction(self):
        act = input(">input: ")
        assert(act in actions)
        return act


    def setAction(self, new_action):
        self.action = new_action

class AI(Player):
    def __init__(self):
        Player.__init__(self)
        self.time = 0
        self.lst = []
        
    def receiveHistory(self, history):
        self.lst.append(history)
        
    def getAction(self):
        if self.time < 1:
            print("first time")
            self.time += 1
            return generateAction(1/3, 1/3, 1/3)
        else:
            last_data = np.array([data_dict.word2idx[self.lst[-1]]])
            idx = np.argmax(model.predict(last_data))
            res = label_dict.idx2word[idx]
        
            if res == 'r\n':   # shift the predicted action to win the game 
                return 'p'
            elif res == 's\n':
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
p2.receiveHistory(action1 + str(state1) + '\n')
#f = open('./res.txt', 'w+')#'C:\Users\lenovo\Documents\SRTP\PSR_AI\testcase\test.txt', 'w')
action1s = []
action2s = []
#action1 = action2 = None
stat = {-1: 0, 0: 0, 1: 0}
for i in range(upbound):
    action1 = p1.getAction()
    action2 = p2.getAction()
    p1.setAction(action1)
    p2.setAction(action2)
    #print("action(p1,p2): ",action1," ",action2, "\n")
    action1s.append(action1)
    action2s.append(action2)
    #f.write(action1 + ' ' + action2 + ' ')
    (state1, state2) = getRes(action1, action2)
    stat[state1] += 1
    p1.state = state1
    p2.state = state2
    p2.receiveHistory(action1 + str(state1) + '\n')
    print(stat.items())
#count = 0
#for ac1, ac2 in zip(action1s, action2s):
#    f.write(ac1 + ' ' + ac2 + ' ' + str(getRes(ac1, ac2)[0]) + '\n')#ac2 + '\n')#+ ' ' + 
#    count += getRes(ac1, ac2)[0]
#f.close()
print(count)
print(stat.items())
print('ok')

