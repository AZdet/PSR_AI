from .utils import *
import numpy as np


class Player:
    def __init__(self):
        self.state = tie
        self.action = s

    def getAction(self):
        pass

    def setAction(self, new_action):
        self.action = new_action

    def receiveHistory(self, history):
        pass


class RandomPlayer(Player):
    def __init__(self, input_matrix=None):
        Player.__init__(self)
        if input_matrix is None:
            input_matrix = [
                [0.33, 0.33, 0.33],
                [0.33, 0.33, 0.33],
                [0.33, 0.33, 0.33]
            ]

        self.input_matrix = input_matrix
        [self.p11, self.p12, self.p13] = input_matrix[0]  # w-, w+, w0
        [self.p21, self.p22, self.p23] = input_matrix[1]  # t-, t+, t0
        [self.p31, self.p32, self.p33] = input_matrix[2]  # l-, l+, l0

    def receiveHistory(self, history):
        pass

    def getAction(self):
        idx = lookTab[self.action]
        pr = [1, 2, 3]
        if self.state == win:
            pr[idx] = self.p13
            pr[(idx + 1) % 3] = self.p11
            idx -= 1
            if idx < 0:
                idx = 2
            pr[idx] = self.p12
        elif self.state == tie:
            pr[idx] = self.p23
            pr[(idx + 1) % 3] = self.p21
            idx -= 1
            if idx < 0:
                idx = 2
            pr[idx] = self.p22
        else:  # self.state == lose
            pr[idx] = self.p33
            pr[(idx + 1) % 3] = self.p31
            idx -= 1
            if idx < 0:
                idx = 2
            pr[idx] = self.p32
        return generateAction(pr[0], pr[1], pr[2])


class AI(Player):
    def __init__(self, data_dict, label_dict, model):
        Player.__init__(self)
        self.time = 0
        self.lst = []
        self.data_dict = data_dict
        self.label_dict = label_dict
        self.model = model

    def receiveHistory(self, history):
        self.lst.append(history)

    def getAction(self):
        if self.time < 1:
            print("first time")
            self.time += 1
            # self.lst.append(opposite_action + str(-self.state) + '\n')
            return generateAction(1 / 3, 1 / 3, 1 / 3)
        else:
            # last_action = np.array([data_dict.word2idx[self.lst[-1]]])
            # last_state = opposite_action + str(-self.state) + '\n'
            last_data = np.array([self.data_dict.word2idx[self.lst[-1]]])
            idx = np.argmax(self.model.predict(last_data))
            res = self.label_dict.idx2word[idx]
            # print(words)
            # symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            # keys = np.reshape(np.array(symbols_in_keys), [-1, 3, 1])
            # onehot_pred = sess.run(pred, feed_dict={x: keys})
            # onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval(session=sess))
            # symbols_in_keys = symbols_in_keys[1:]
            # symbols_in_keys.append(onehot_pred_index)
            # self.lst = self.lst[1:]
            # self.lst.append(opposite_action + str(-self.state) + '\n')
            # res =  reverse_dictionary[onehot_pred_index][0]

            # assert(res in actions)
            if res == 'r\n':  # shift the predicted action to win the game
                return 'p'
            elif res == 's\n':
                return 'r'
            else:  # res == 'p'
                return 's'


class Agent:
    def __init__(self, player1, player2):
        action1 = generateAction(1 / 3, 1 / 3, 1 / 3)
        action2 = generateAction(1 / 3, 1 / 3, 1 / 3)
        player1.setAction(action1)
        player2.setAction(action2)
        (state1, state2) = getRes(action1, action2)
        player1.state = state1
        player2.state = state2
        player2.receiveHistory(action1 + str(state1) + '\n')
        player1.receiveHistory(action2 + str(state2) + '\n')

        self.player1 = player1
        self.player2 = player2

    def run(self, output_data_path=None, output_label_path=None, upbound=10000):
        action1s = []
        action2s = []
        stat = {-1: 0, 0: 0, 1: 0}

        if output_data_path is not None:
            data_file = open(output_data_path, "w+")
        else:
            data_file = None

        if output_label_path is not None:
            label_file = open(output_label_path, "w+")
        else:
            label_file = None

        for i in range(0, upbound):
            action1 = self.player1.getAction()
            if i > 0 and label_file is not None:
                label_file.write(action1 + '\n')  # write the next action as the label
            action2 = self.player2.getAction()
            self.player1.setAction(action1)
            self.player2.setAction(action2)
            # print("action(p1,p2): ",action1," ",action2, "\n")
            action1s.append(action1)
            action2s.append(action2)
            # f.write(action1 + ' ' + action2 + ' ')
            (state1, state2) = getRes(action1, action2)
            stat[state1] += 1
            self.player1.state = state1
            self.player2.state = state2
            self.player2.receiveHistory(action1 + str(state1) + '\n')
            self.player1.receiveHistory(action2 + str(state2) + '\n')

        if data_file is not None:
            for ac1, ac2 in zip(action1s, action2s):
                data_file.write(ac1 + str(getRes(ac1, ac2)[0]) + '\n')  # ac2 + '\n')#+ ' ' +
            data_file.close()

        if label_file is not None:
            label_file.close()

        return stat
