(win, tie, lose) = (1, 0, -1)
class Player():
    def __init__(self):
        self.state = tie

    def getAction(self, alpha):
        if self.state == win:
            action = generateAction(p11, p12, p13)
        elif self.state == tie:
            action = generateAction(p21, p22, p23)
        else:
            action = generateAction(p31, p32, p33)
        return action


p1 = Player()
p2 = Player()
action1 = generateAction(1/3, 1/3, 1/3)
action2 = generateAction(1/3, 1/3, 1/3)
(state1, state2) = getRes(action1, action2)
p1.state = state1
p2.state = state2
for i in range(0, upbound):
    action1 = p1.getAction()
    action2 = p2.getAction()
    print("action(p1,p2): ",action1," ",action2, "\n")
    (state1, state2) = getRes(action1, action2)
    p1.state = state1
    p2.state = state2



