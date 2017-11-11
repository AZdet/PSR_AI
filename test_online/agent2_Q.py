# first
import random as rd
import util

f = open('res2.txt', 'a')
def getStatus(input, output):
	if input == output:
		return 'tie'
	elif (input == 'R' and output == 'S') or \
		(input == 'P' and output == 'R') or \
		(input == 'S' and output == 'P'):
		return 'lost'
	else:
		return 'win'

def getChange(input, last_input):
	if input == last_input:
		return 'o'
	elif (last_input == 'R' and input == 'P') or \
	(last_input == 'P' and input == 'S') or \
	(last_input == 'S' and input == 'R'):
		return '+'
	else:
		return '-'

if input == '':
	actions = ['R', 'P', 'S']
	output = rd.choice(actions)
	first = True
	Q_values = util.Counter()
	'''hist = {(('tie', '-'), 'P'): 1.9992379122545871, (('lost', '-'), 'R'): 1.0831656209123348, (('win', 'o'), 'R'): 2.1057877678126733, (('lost', 'o'), 'R'): 2.12119587862252, (('tie', '+'), 'P'): 1.4305605736702227, (('lost', '+'), 'S'): 2.4613080508747207, (('tie', 'o'), 'P'): 0.37832855980881985, (('start', 'start'), 'P'): -0.4, (('lost', 'o'), 'S'): 1.045106455750678, (('tie', '-'), 'S'): 0.9058400871092706, (('tie', '+'), 'S'): 0.6334305236760692, (('tie', '-'), 'R'): 1.0395658644023686, (('win', '-'), 'P'): 1.4156733555353078, (('tie', 'o'), 'S'): 2.2352190051465697, (('win', '+'), 'P'): 1.5348469631244437, (('lost', '+'), 'P'): 1.1330197536381663, (('lost', '-'), 'S'): 0.9701302053926766, (('tie', '+'), 'R'): 1.5835010565595593, (('win', 'o'), 'P'): 0.8413874021200219, (('lost', 'o'), 'P'): 1.5111450198810914, (('tie', 'o'), 'R'): 0.971434885386365, (('win', '-'), 'S'): 1.5825912693857527, (('win', '+'), 'S'): 1.9159539585003382, (('lost', '-'), 'P'): 0.8731603360904714, (('win', 'o'), 'S'): 0.49913350362229203, (('win', '-'), 'R'): 1.5675332858825946, (('win', '+'), 'R'): 1.0828297904881157, (('lost', '+'), 'R'): 1.1491198263709426}

	for key, val in hist.items():
		Q_values[key] = val'''

	'''states = [('start', 'start'),
		 ('win', '+'), ('win', '-'), ('win', 'o'),
		 ('tie', '+'), ('tie', '-'), ('tie', 'o'),
		 ('lost', '+'), ('lost', '-'), ('lost', 'o')]
	episode_num = 10
	step_num = 1000'''
	alpha = 1.0
	gamma = 0.9
	epsilon = 0.2
	decay = 0.99954
	last_input = ''
	count = 0
	
	
elif first:
	last_input = input
	output = rd.choice(actions)
	state = ('start', 'start')
	first = False
else:
	# alpha = (alpha) / (alpha + count / 100)   # decay learning rate
	count += 1
	#if count % 300 == 0:
	#	Q_values = util.Counter()
	status = getStatus(input, output) # win or tie or lost
	if status == 'win':
		reward = 1
	elif status == 'tie':
		reward = 0
	else:
		reward = -1
	change = getChange(input, last_input) # + or - or o
	new_state = (status, change)  # minimax Q-learning, add opposite's action as part of states
	poss = []
	for action in actions:
		tmp = []
		for oppo_action in actions:
			tmp.append(Q_values[(new_state, action, oppo_action)])
		poss.append(min(tmp)) 
	new_q = max(poss)
	idx = rd.choice([idx for idx in [0,1,2] if poss[idx] == new_q]) 
	Q_values[(state, output, input)] = (1.0 - alpha) * Q_values[(state, output, input)] + alpha * (reward + gamma * new_q)
	alpha *= decay
	state = new_state
	if count % 100 == 1:
		f.write('{0} {1} {2} {3}\n'.format(input, output, reward, Q_values))
	if util.flipCoin(epsilon):   # epsilon greedy
		output = rd.choice(actions)
	else:
		output = actions[idx]
		 # TODO, exploration 
f.close()
