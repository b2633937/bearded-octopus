from game import Game
from agents import *
from locals import *
import numpy as np
import matplotlib.pyplot as plt

""" Class seperation made in order to enable easy testing by setting deiffferent variables through argument passing
 though this remains work in progress for the moment"""

def main():

	rounds = 100
	episodes = 1000
	maxEpisodeLength = 1000
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
		game = Game(boardSize=(11,11), verbose=False, draw=False, episodes=episodes, maxEpisodeLength = maxEpisodeLength)  
	 	game.addPlayer(Player(agent=TemporalDifference(
	 	    algorithm = algorithm,
	        QtableFN = QtableFN,  
	        Qinitval = Qinitval,
	        alpha = alpha,
	        gamma = gamma,
	        selParam = selParam,
	        selAlgh = selAlgh
	 		)),role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 
	 	# game.addPlayer(Player(agent=MonteCarloOnP(selAlgh, selParam)), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
	 	game.addPlayer(Player(agent=RandomComputer()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	stats[:,0] += game.play()
	stats[:,0] = stats[:,0] / float(rounds)
	try:
	    pickle.dump(stats, open('stats'+algorithm+'a'+str(alpha[i])+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
	except:
	    print "can't write stats file"

	###################################################################################

	selAlgh = 'softMax'

	for rnd in xrange(rounds):
		print 'running round ', rnd + 1 
		game = Game(boardSize=(11,11), verbose=False, draw=False, episodes=episodes, maxEpisodeLength = maxEpisodeLength)  
	 	game.addPlayer(Player(agent=TemporalDifference(
	 	    algorithm = algorithm,
	        QtableFN = QtableFN,  
	        Qinitval = Qinitval,
	        alpha = alpha,
	        gamma = gamma,
	        selParam = selParam,
	        selAlgh = selAlgh
	 		)),role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 
	 	# game.addPlayer(Player(agent=MonteCarloOnP(selAlgh, selParam)), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
	 	game.addPlayer(Player(agent=RandomComputer()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	stats[:,1] += game.play()
	stats[:,1] = stats[:,1] / float(rounds)
	try:
	    pickle.dump(stats, open('stats'+algorithm+'a'+str(alpha[i])+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
	except:
	    print "can't write stats file"

   ###################################################################################

	selAlgh = 'eGreedy'
	algorithm = 'Q-learning'

	for rnd in xrange(rounds):
		print 'running round ', rnd + 1 
		game = Game(boardSize=(11,11), verbose=False, draw=False, episodes=episodes, maxEpisodeLength = maxEpisodeLength)  
	 	game.addPlayer(Player(agent=TemporalDifference(
	 	    algorithm = algorithm,
	        QtableFN = QtableFN,  
	        Qinitval = Qinitval,
	        alpha = alpha,
	        gamma = gamma,
	        selParam = selParam,
	        selAlgh = selAlgh
	 		)),role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 
	 	# game.addPlayer(Player(agent=MonteCarloOnP(selAlgh, selParam)), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
	 	game.addPlayer(Player(agent=RandomComputer()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	stats[:,2] += game.play()

	stats[:,2] = stats[:,2] / float(rounds)
	try:
	    pickle.dump(stats, open('stats'+algorithm+'a'+str(alpha[i])+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
	except:
	    print "can't write stats file"

	###################################################################################

	selAlgh = 'softMax'

	for rnd in xrange(rounds):
		print 'running round ', rnd + 1 
		game = Game(boardSize=(11,11), verbose=False, draw=False, episodes=episodes, maxEpisodeLength = maxEpisodeLength)  
	 	game.addPlayer(Player(agent=TemporalDifference(
	 	    algorithm = algorithm,
	        QtableFN = QtableFN,  
	        Qinitval = Qinitval,
	        alpha = alpha,
	        gamma = gamma,
	        selParam = selParam,
	        selAlgh = selAlgh
	 		)),role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 
	 	# game.addPlayer(Player(agent=MonteCarloOnP(selAlgh, selParam)), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
	 	game.addPlayer(Player(agent=RandomComputer()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	stats[:,3] += game.play()

	stats[:,3] = stats[:,3] / float(rounds)
	# std = stats.std(0)
	# print 'average of: ', avg
	# print 'standard deviation of: ', std
	try:
	    pickle.dump(stats, open('stats'+algorithm+'a'+str(alpha[i])+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
	except:
	    print "can't write stats file"



	# fig, ax = plt.subplots()

	# ax.plot(x, x**2, label=r"$y = \alpha^2$")
	# ax.plot(x, x**3, label=r"$y = \alpha^3$")
	# ax.set_xlabel(r'$\alpha$', fontsize=18)
	# ax.set_ylabel(r'$y$', fontsize=18)
	# ax.set_title('title')
	# ax.legend(loc=2); # upper left corner

	fig, ax = plt.subplots(figsize=(12,5))
	ax.plot(stats[:,0], label="eGreedy Sarsa")
	ax.plot(stats[:,1], label="softMax Sarsa")
	ax.plot(stats[:,2], label="eGreedy Q-learning")
	ax.plot(stats[:,3], label="softmax Q-learning")
	ax.legend(loc=1); # upper left corner
	ax.set_xlabel('episode')
	ax.set_ylabel('average number of turns')
	ax.set_title('alpha 0.5, Qinit 0, gamma 0.7 ,epsilon and Tau 0.2')
	plt.show()

if __name__ == '__main__':
    main()
