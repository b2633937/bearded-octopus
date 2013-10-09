from game import Game
from agents import *
from locals import *
import numpy as np
import matplotlib.pyplot as plt

""" Class seperation made in order to enable easy testing by setting deiffferent variables through argument passing
 though this remains work in progress for the moment"""

def main():

	runs = 10
	episodes = 100
	maxTurn = 0 #0 amounts to no maximum number of turns per round
	stats = np.zeros((episodes,4))
	algorithm = 'Sarsa' #algorithm to be used by TD agents (either Sarsa or Q-learning)
	QtableFN = None  
	Qinitval = 0
	alpha = 0.5
	gamma = 0.7
	selParam = 0.2 #selection parameter Tau or Epsilon
	selAlgh = 'eGreedy' # eGreedy or softMax 
	simultaniousActions = True

	###################################################################################

	for rnd in xrange(runs):
		print 'running round ', rnd + 1 
		game = Game(boardSize=(11,11), verbose=False, draw=False, episodes=episodes, maxTurn=maxTurn, simultaniousActions=simultaniousActions)  


		# game.addPlayer(Player(agent=TemporalDifference(
		# 	algorithm = algorithm,
		# 	QtableFN = QtableFN,  
		# 	Qinitval = Qinitval,
		# 	alpha = alpha,
		# 	gamma = gamma,
		# 	selParam = selParam,
		# 	selAlgh = selAlgh)),
		# 	role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 

		game.addPlayer(Player(agent=OnPolicyMonteCarlo(gamma, selParam, (11,11))), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])

	 	# game.addPlayer(Player(agent=RandomComputer()), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
	 	game.addPlayer(Player(agent=RandomTripper()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	stats[:,0] += game.play()
	stats[:,0] = stats[:,0] / float(runs)
	try:
	    pickle.dump(stats, open('stats'+algorithm+'a'+str(alpha[i])+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
	except:
	    print "can't write stats file"

	fig, ax = plt.subplots(figsize=(12,5))
	ax.plot(stats[:,0], label="Random Policy")
	ax.legend(loc=1); 
	ax.set_xlabel('episode')
	ax.set_ylabel('turns')
	ax.set_title('average episode length over ' + str(runs) + ' runs')
	plt.show()

#----- function used for plotting error margins -------
def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

if __name__ == '__main__':
    main()
