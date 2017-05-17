import random as rd
lookTab = {"r": 0, "s": 1, "p": 2}
[r, s, p] = ["r", "s", "p"]
actions = [r, s, p]
alpha = 1.1
upbound = 20
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
    elif (action1 == r and action2 == s)or\
         (action1 == s and action2 == p)or\
         (action1 == p and action2 == r):
        return win, lose
    else:
        return lose, win





class Player():
    def __init__(self):
        self.state = tie
        self.action = s

    def getAction(self):
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


p1 = Player()
p2 = Player()
action1 = generateAction(1/3, 1/3, 1/3)
action2 = generateAction(1/3, 1/3, 1/3)
p1.setAction(action1)
p2.setAction(action2)
(state1, state2) = getRes(action1, action2)
p1.state = state1
p2.state = state2
for i in range(0, upbound):
    action1 = p1.getAction()
    action2 = p2.getAction()
    p1.setAction(action1)
    p2.setAction(action2)
    print("action(p1,p2): ",action1," ",action2, "\n")
    (state1, state2) = getRes(action1, action2)
    p1.state = state1
    p2.state = state2



