from game import Game
from agents import *
from locals import *
""" Class seperation made in order to enable easy testing by setting deiffferent variables through argument passing
 though this remains work in progress for the moment"""

def main():

	game = Game(boardSize=(11,11), verbose=True, draw=True, rounds=1, episodes=100)  
 	game.addPlayer(Player(agent=GeneralizedPolicyIteration()), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) # fix vor valueteration
 	# game.addPlayer(Player(agent=RandomComputer()), role=PREDATOR, fixedInitPos=(0,0), img=IMAGES['boy']) # fix vor valueteration
 	game.addPlayer(Player(agent=Human()), role=PREY, fixedInitPos=(5,5), img=IMAGES['princess'])
 	game.play()

if __name__ == '__main__':
    main()


