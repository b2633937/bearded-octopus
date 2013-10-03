from game import Game
from agents import *
from locals import *
import numpy as np
import matplotlib.pyplot as plt

""" Class seperation made in order to enable easy testing by setting deiffferent variables through argument passing
 though this remains work in progress for the moment"""

def main():

	for i in xrange(1): #(5):
		algorithm = 'Sarsa'
		QtableFN = None  
		Qinitval = 15
		alpha = [0.1,0.2,0.3,0.4,0.5]
		gamma = 0.7#[0.1,0.5,0.7,0.9]
		selParam = 0.1 # selection parameter Tau or Epsilon
		selAlgh = 'softMax'

		rounds = 1
		episodes = 1000
		stats = np.zeros((rounds, episodes))

		for rnd in xrange(rounds):
			game = Game(boardSize=(11,11), verbose=True, draw=False, episodes=episodes)  
		 	game.addPlayer(Player(agent=TemporalDifference(
		 	    algorithm = algorithm,
		        QtableFN = QtableFN,  
		        Qinitval = Qinitval,
		        alpha = alpha[i],
		        gamma = gamma,
		        selParam = selParam,
		        selAlgh = selAlgh
		 		)),role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) 
		 	# game.addPlayer(Player(agent=DynamicProgramming()), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy'])
		 	game.addPlayer(Player(agent=RandomComputer()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
		 	stats[rnd, :] = game.play()

		avg = stats.sum(0) / float(rounds)
		std = stats.std(0)
		# print 'average of: ', avg
		# print 'standard deviation of: ', std
		try:
		    pickle.dump(stats, open('stats'+algorithm+'a'+str(alpha[i])+'e'+str(selParam)+'g'+str(gamma)+'i'+str(Qinitval)+'.p', "wb"), pickle.HIGHEST_PROTOCOL)    
		except:
		    print "can't write stats file"

		fig, sp1 = plt.subplots(figsize=(12,5))
        sp1.plot(avg)
        sp1.set_xlabel('episode')
        sp1.set_ylabel('average number of turns')
        sp1.set_title('');
        plt.show()

if __name__ == '__main__':
    main()
