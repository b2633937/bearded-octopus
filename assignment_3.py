from game import Game
from agents import *
from locals import *
import numpy as np
import matplotlib.pyplot as plt

""" Class seperation made in order to enable easy testing by setting deiffferent variables through argument passing
 though this remains work in progress for the moment"""

def main():

	rounds = 10
	episodes = 100
	maxEpLen = 0
	stats = np.zeros((episodes,4))
	algorithm = 'Sarsa'
	QtableFN = None  
	Qinitval = 0
	alpha = 0.5
	gamma = 0.7#[0.1,0.5,0.7,0.9]
	selParam = 0.2 #selection parameter Tau or Epsilon
	selAlgh = 'eGreedy' # eGreedy or softMax 

	###################################################################################

	for rnd in xrange(rounds):
		print 'running round ', rnd + 1 
		game = Game(boardSize=(11,11), verbose=False, draw=False, episodes=episodes, maxEpLen=maxEpLen)  
	 	# game.addPlayer(Player(agent=TemporalDifference(
	 	#     algorithm = algorithm,
	  #       QtableFN = QtableFN,  
	  #       Qinitval = Qinitval,
	  #       alpha = alpha,
	  #       gamma = gamma,
	  #       selParam = selParam,
	  #       selAlgh = selAlgh
	 	# 	)),role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 
	 	game.addPlayer(Player(agent=RandomComputer()), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
	 	game.addPlayer(Player(agent=RandomTripper()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	stats[:,0] += game.play()
	stats[:,0] = stats[:,0] / float(rounds)
	try:
	    pickle.dump(stats, open('stats'+algorithm+'a'+str(alpha[i])+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
	except:
	    print "can't write stats file"

	fig, ax = plt.subplots(figsize=(12,5))
	ax.plot(stats[:,0], label="eGreedy Sarsa")
	ax.plot(stats[:,1], label="softMax Sarsa")
	ax.plot(stats[:,2], label="eGreedy Q-learning")
	ax.plot(stats[:,3], label="softmax Q-learning")
	ax.legend(loc=1); 
	ax.set_xlabel('episode')
	ax.set_ylabel('average number of turns')
	ax.set_title('alpha 0.5, Qinit 0, gamma 0.7 ,epsilon and Tau 0.2')
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
