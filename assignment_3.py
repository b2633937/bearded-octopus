from game import Game
from agents import *
from locals import *
import numpy as np
import matplotlib.pyplot as plt

""" Class seperation made in order to enable easy testing by setting deiffferent variables through argument passing
 though this remains work in progress for the moment"""

def main():
	test()


def test():

	runs = 25
	episodes = 100
	maxTurn = 0 #0 amounts to no maximum number of turns per round
	turns = np.zeros(episodes)
	simultaniousActions = True
	preyTrip = True
	verbose = False
	draw = False
	boardSize = (11,11)

	algorithm = 'Q-learning' #algorithm to be used by TD agents (either Sarsa or Q-learning)
	QtableFN = None  
	Qinitval = 0
	alpha = 0.5
	gamma = 0.7
	selParam = 0.1 #selection parameter Tau or Epsilon
	selAlgh = 'eGreedy' # eGreedy or softMax 

	for rnd in xrange(runs):
		print 'running round ', rnd + 1 
		game = Game(boardSize=boardSize, verbose=verbose, draw=draw, episodes=episodes, maxTurn=maxTurn, simultaniousActions=simultaniousActions, preyTrip=preyTrip)
	 	game.addPlayer(Player(agent=Random()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])

		game.addPlayer(Player(agent=TemporalDifference(
		algorithm = algorithm,
		QtableFN = QtableFN,  
		Qinitval = Qinitval,
		alpha = alpha,
		gamma = gamma,
		selParam = selParam,
		selAlgh = selAlgh)),
		role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 

	 	results = game.play()
	 	turns += results['turns']
	turns = turns / float(runs)

	print 'mean episode length: ', turns.mean(), ' with a std of: ', turns.std()
	print 'predators winning chance: ', sum([outcome == 1 for outcome in results['outcomes']]) / float(episodes)

	index = [outcome == 1 for outcome in results['outcomes']]
	winEpLen = np.array([v for i, v in enumerate(turns) if index[i]])
	print 'mean episode length when predators are winning: ', winEpLen.mean(), ' with a std of: ', winEpLen.std()

	# index = [outcome == -1 for outcome in results['outcomes']]
	# loseEpLen = np.array([v for i, v in enumerate(turns) if index[i]])
	# print 'mean episode length when predators are losing: ', loseEpLen.mean(), ' with a std of: ', loseEpLen.std()

	fig, ax = plt.subplots(figsize=(12,5))
	ax.plot(turns, label="Q-learning")
	ax.legend(loc=1); 
	ax.set_xlabel('episode')
	ax.set_ylabel('turns')
	ax.set_title('mean episode length over ' + str(runs) + ' runs')
	plt.show()


"""###################################################################################"""
def assignment4_1():

	runs = 1
	episodes = 1000
	maxTurn = 0 #0 amounts to no maximum number of turns per round
	turns = np.zeros(episodes)
	simultaniousActions = True
	preyTrip = True
	verbose = False
	draw = False
	boardSize = (11,11)

	for rnd in xrange(runs):
		print 'running round ', rnd + 1 
		game = Game(boardSize=boardSize, verbose=verbose, draw=draw, episodes=episodes, maxTurn=maxTurn, simultaniousActions=simultaniousActions, preyTrip=preyTrip)
	 	game.addPlayer(Player(agent=Random()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	game.addPlayer(Player(agent=PseudoRandom()), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
	 	game.addPlayer(Player(agent=PseudoRandom()), role=PREDATOR, fixedInitPos=(0,10), img=IMAGES['boy'])
	 	# game.addPlayer(Player(agent=PseudoRandom()), role=PREDATOR, fixedInitPos=(10,0), img=IMAGES['boy'])
	 	# game.addPlayer(Player(agent=PseudoRandom()), role=PREDATOR, fixedInitPos=(10,10), img=IMAGES['boy'])
	 	results = game.play()
	 	turns += results['turns']
	turns = turns / float(runs)

	print 'mean episode length: ', turns.mean(), ' with a std of: ', turns.std()
	print 'predators winning chance: ', sum([outcome == 1 for outcome in results['outcomes']]) / float(episodes)

	index = [outcome == 1 for outcome in results['outcomes']]
	winEpLen = np.array([v for i, v in enumerate(turns) if index[i]])
	print 'mean episode length when predators are winning: ', winEpLen.mean(), ' with a std of: ', winEpLen.std()

	index = [outcome == -1 for outcome in results['outcomes']]
	loseEpLen = np.array([v for i, v in enumerate(turns) if index[i]])
	print 'mean episode length when predators are losing: ', loseEpLen.mean(), ' with a std of: ', loseEpLen.std()




"""###################################################################################"""
def assignment4_2():

	runs = 1
	episodes = 20
	maxTurn = 0 #0 amounts to no maximum number of turns per round
	turns = np.zeros((episodes,1))
	algorithm = 'Q-learning' #algorithm to be used by TD agents (either Sarsa or Q-learning)
	QtableFN = None  
	Qinitval = 0
	alpha = 0.5
	gamma = 0.7
	selParam = 0.1 #selection parameter Tau or Epsilon
	selAlgh = 'eGreedy' # eGreedy or softMax 
	simultaniousActions = True
	preyTrip = True
	verbose = False
	draw = False
	boardSize = (11,11)
	wins = np.zeros(episodes)

	for rnd in xrange(runs):
		print 'running round ', rnd + 1 
		game = Game(boardSize=boardSize, verbose=verbose, draw=draw, episodes=episodes, maxTurn=maxTurn, simultaniousActions=simultaniousActions, preyTrip=preyTrip)  

		game.addPlayer(Player(agent=TemporalDifference(
			algorithm = algorithm,
			QtableFN = QtableFN,  
			Qinitval = Qinitval,
			alpha = alpha,
			gamma = gamma,
			selParam = selParam,
			selAlgh = selAlgh)),
			role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 

		game.addPlayer(Player(agent=TemporalDifference(
			algorithm = algorithm,
			QtableFN = QtableFN,  
			Qinitval = Qinitval,
			alpha = alpha,
			gamma = gamma,
			selParam = selParam,
			selAlgh = selAlgh)),
			role=PREDATOR, fixedInitPos=(0,10), img=IMAGES['boy']) 

		#-----------------Add the Prey-----------------------
		game.addPlayer(Player(agent=TemporalDifference(
			algorithm = algorithm,
			QtableFN = QtableFN,  
			Qinitval = Qinitval,
			alpha = alpha,
			gamma = gamma,
			selParam = selParam,
			selAlgh = selAlgh)),
			role=PREY, fixedInitPos=(5,5), img=IMAGES['princess']) 

	 	# game.addPlayer(Player(agent=Random()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
	 	results = game.play()
	 	turns[:,0] += results['turns']
 		wins += np.array([1 if outcome == 1 else 0 for outcome in results['outcomes']])

	turns[:,0] = turns[:,0] / float(runs)

	# print 'mean episode length: ', turns[:,0].mean(), ' with a std of: ', turns[:,0].std()
	# print 'predators winning chance: ', sum([outcome == 1 for outcome in results['outcomes']]) / float(episodes)

	wins = wins/float(runs)  
	winEpLen = np.array([v for i, v in enumerate(turns[:,0]) if wins[i]])
	print 'mean episode length when predators are winning: ', winEpLen.mean(), ' with a std of: ', winEpLen.std()

	# index = [outcome == -1 for outcome in results['outcomes']]
	# loseEpLen = np.array([v for i, v in enumerate(turns[:,0]) if index[i]])
	# print 'mean episode length when predators are losing: ', loseEpLen.mean(), ' with a std of: ', loseEpLen.std()

	# try:
	#     pickle.dump(turns, open('turns'+algorithm+'a'+str(alpha)+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
	# except:
	#     print "can't write turns file"

	fig, ax = plt.subplots(figsize=(12,5))
	ax.plot([0]+list(wins), label="Q-learning")
	ax.axis([1, episodes, 0, 1])
	ax.legend(loc=4); 
	ax.set_xlabel('episode')
	ax.set_ylabel('probability')
	ax.set_title('Chance of predators winning.')
	plt.show()


#######################################################
# def assignment4_2():

# 	runs = 10
# 	episodes = 500
# 	maxTurn = 0 #0 amounts to no maximum number of turns per round
# 	turns = np.zeros((episodes,1))
# 	algorithm = 'Q-learning' #algorithm to be used by TD agents (either Sarsa or Q-learning)
# 	QtableFN = None  
# 	Qinitval = 0
# 	alpha = 0.5
# 	gamma = 0.7
# 	selParam = 0.2 #selection parameter Tau or Epsilon
# 	selAlgh = 'eGreedy' # eGreedy or softMax 
# 	simultaniousActions = True
# 	preyTrip = True
# 	verbose = False
# 	draw = False
# 	boardSize = (11,11)

# 	for rnd in xrange(runs):
# 		print 'running round ', rnd + 1 
# 		game = Game(boardSize=boardSize, verbose=verbose, draw=draw, episodes=episodes, maxTurn=maxTurn, simultaniousActions=simultaniousActions, preyTrip=preyTrip)  

# 		game.addPlayer(Player(agent=TemporalDifference(
# 			algorithm = algorithm,
# 			QtableFN = QtableFN,  
# 			Qinitval = Qinitval,
# 			alpha = alpha,
# 			gamma = gamma,
# 			selParam = selParam,
# 			selAlgh = selAlgh)),
# 			role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 

# 		# game.addPlayer(Player(agent=TemporalDifference(
# 		# 	algorithm = algorithm,
# 		# 	QtableFN = QtableFN,  
# 		# 	Qinitval = Qinitval,
# 		# 	alpha = alpha,
# 		# 	gamma = gamma,
# 		# 	selParam = selParam,
# 		# 	selAlgh = selAlgh)),
# 		# 	role=PREDATOR, fixedInitPos=(0,10), img=IMAGES['boy']) 

# 		#-----------------Add the Prey-----------------------
# 		game.addPlayer(Player(agent=TemporalDifference(
# 			algorithm = algorithm,
# 			QtableFN = QtableFN,  
# 			Qinitval = Qinitval,
# 			alpha = alpha,
# 			gamma = gamma,
# 			selParam = selParam,
# 			selAlgh = selAlgh)),
# 			role=PREY, fixedInitPos=(5,5), img=IMAGES['princess']) 

# 	 	# game.addPlayer(Player(agent=Random()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
# 	 	results = game.play()
# 	 	turns[:,0] += results['turns']
# 	turns[:,0] = turns[:,0] / float(runs)

# 	# print 'mean episode length: ', turns[:,0].mean(), ' with a std of: ', turns[:,0].std()
# 	# print 'predators winning chance: ', sum([outcome == 1 for outcome in results['outcomes']]) / float(episodes)

# 	index = [outcome == 1 for outcome in results['outcomes']]
# 	winEpLen = np.array([v for i, v in enumerate(turns[:,0]) if index[i]])
# 	print 'mean episode length when predators are winning: ', winEpLen.mean(), ' with a std of: ', winEpLen.std()

# 	# index = [outcome == -1 for outcome in results['outcomes']]
# 	# loseEpLen = np.array([v for i, v in enumerate(turns[:,0]) if index[i]])
# 	# print 'mean episode length when predators are losing: ', loseEpLen.mean(), ' with a std of: ', loseEpLen.std()

# 	try:
# 	    pickle.dump(turns, open('turns'+algorithm+'a'+str(alpha)+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
# 	except:
# 	    print "can't write turns file"

# 	fig, ax = plt.subplots(figsize=(12,5))
# 	ax.plot(turns[:,0], label="Q-learning")
# 	ax.legend(loc=1); 
# 	ax.set_xlabel('episode')
# 	ax.set_ylabel('turns')
# 	ax.set_title('mean episode length over ' + str(runs) + ' runs')
# 	plt.show()


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
