# See http://overview.cc/RockPaperScissors for more information about rock, paper, scissors
# Although bayes7 performed very well I think there is a bug in the counter_prob function.
# This version should fix that.
from collections import defaultdict
import operator
import random

if input == "":
  score = {'RR': 0, 'PP': 0, 'SS': 0, 'PR': 1, 'RS': 1, 'SP': 1,'RP': -1, 'SR': -1, 'PS': -1,}
  cscore = {'RR': 'r', 'PP': 'r', 'SS': 'r', 'PR': 'b', 'RS': 'b', 'SP': 'b','RP': 'c', 'SR': 'c', 'PS': 'c',}
  
  rps = ['R', 'P', 'S']
  
  def counter_prob(probs):
    weighted_list = []
    for h in ['R', 'P', 'S']:
      weighted = 0
      for p in probs.keys():
        points = score[h+p]
        prob = probs[p]
        weighted += points * prob
      weighted_list.append((h, weighted))

    return max(weighted_list, key=operator.itemgetter(1))[0]

  played_probs = defaultdict(lambda: 1)
  opp_probs = defaultdict(lambda: defaultdict(lambda: 1))
  my_probs = defaultdict(lambda: defaultdict(lambda: 1))
  both_probs = defaultdict(lambda: defaultdict(lambda: 1))

  opp2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  my2_probs = defaultdict(lambda: defaultdict(lambda: 1))
  both2_probs = defaultdict(lambda: defaultdict(lambda: 1))

  opp_answers = {'c': 1, 'b': 1, 'r': 1}
  my_answers = {'c': 1, 'b': 1, 'r': 1}

  opp2_answers = {'c': 1, 'b': 1, 'r': 1}
  my2_answers = {'c': 1, 'b': 1, 'r': 1}  

  patterndict = defaultdict(str)
  patterndict2 = defaultdict(str)

  output = random.choice(["R", "P", "S"])
  hist = "" 
  my = opp = my2 = opp2 = ""
else:
  if my and opp:
    opp_answers[cscore[input+opp]] += 1
    my_answers[cscore[input+my]] += 1
  if my2 and opp2:
    opp2_answers[cscore[input+opp2]] += 1
    my2_answers[cscore[input+my2]] += 1

  for length in range(min(10, len(hist)), 0, -2):
    pattern = patterndict[hist[-length:]]
    if pattern:
      #print "Pattern:", pattern, hist[-length:]
      for length2 in range(min(10, len(pattern)), 0, -2):
        #print "pattern2:", pattern[-length2:], repr(patterndict2[pattern[-length2:]]), my, opp
        patterndict2[pattern[-length2:]] += output + input
    patterndict[hist[-length:]] += output + input

  played_probs[input] += 1
  opp_probs[opp][input] += 1
  my_probs[my][input] += 1
  both_probs[my+opp][input] += 1

  opp2_probs[opp2][input] += 1
  my2_probs[my2][input] += 1
  both2_probs[my2+opp2][input] += 1

  hist += output + input

  my = opp = my2 = opp2 = ""

  for length in range(min(10, len(hist)), 0, -2):
    pattern = patterndict[hist[-length:]]
    if pattern != "":
      my = pattern[-2]
      opp = pattern[-1]
      for length2 in range(min(10, len(pattern)), 0, -2):
        pattern2 = patterndict2[pattern[-length2:]]
        if pattern2 != "":
          my2 = pattern2[-2]
          opp2 = pattern2[-1]
          #print "my2:", my2, "opp2:", opp2
          break
      break

  probs = {}
  for hand in rps:
    probs[hand] = played_probs[hand]
        
  if my and opp:
    for hand in rps:
      probs[hand] *= opp_probs[opp][hand] * my_probs[my][hand] * both_probs[my+opp][hand]
      probs[hand] *= opp_answers[cscore[hand+opp]] * my_answers[cscore[hand+my]]
                
  if my2 and opp2:
    for hand in rps:
      probs[hand] *= opp2_probs[opp2][hand] * my2_probs[my2][hand] * both2_probs[my2+opp2][hand]
      probs[hand] *= opp2_answers[cscore[hand+opp2]] * my2_answers[cscore[hand+my2]]
      
  output = counter_prob(probs)