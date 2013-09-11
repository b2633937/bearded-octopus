import random, sys, pygame, pickle, numpy as np
from pygame.locals import *
from locals import *


class Agent(object):
    def __init__(self, player, role, nr, img):
        super(Agent, self).__init__()
        self.player = player
        self.role = role
        self.nr = nr
        self.img = img
        self.id = player.addAgent(self) 
        self.POS = None

    def getAction(self, observation):
        return self.player.getAction(observation, self.id)

    def finalize(self, reward, observation):
        self.player.finalize(reward, observation, self.id) 

    def quit(self):
        self.player.quit(self.id)


class Player(object):

    def __init__(self, gameInstance):
        super(Player, self).__init__()
        self.agents = []
        self.gameInstance = gameInstance

    def addAgent(self, agent):
        self.agents.append(agent)
        return len(self.agents)-1

    def getStateRep(self, observation, id):
        state = []
        for i in xrange(len(observation.positions)):
            if i != self.agents[id].nr:
                state.append(self.translatePos(self.agents[id].POS, observation.positions[i]), self.BOARDSIZE) 
        return tuple(state) 

    def action2Tile(self, action, position):
        boardSize = self.gameInstance.boardSize
        return ((position[0]+EFFECTS[action][0])%boardSize[0], 
            (position[1]+EFFECTS[action][1])%boardSize[1])

    def tile2Action(self, basePosition, adjacentTile):
        if adjacentTile == self.action2Tile(UP, basePosition): return UP
        if adjacentTile == self.action2Tile(DOWN, basePosition): return DOWN
        if adjacentTile == self.action2Tile(LEFT, basePosition): return LEFT
        if adjacentTile == self.action2Tile(RIGHT, basePosition): return RIGHT
        if adjacentTile == basePosition: return STAY

    def translatePos(self, ownPos, otherPos, boardSize): 
        #translates other position after to centering ownPos 
        center = (boardSize[0]/2, boardSize[1]/2)
        return ((otherPos[0] - ownPos[0] + center[0])%boardSize[0], (otherPos[1] - ownPos[1] + center[1])%boardSize[1])

    def getAction(self, observation): 
        sys.exit("function getAction not implemented for agent " + str(agent.nr))

    def finalize(self, reward, observation, id):
        pass

    def quit(self, id):
        pass

class Human(Player):
    """docstring for Human"""
    def __init__(self, gameInstance):
        super(Human, self).__init__(gameInstance)
    
    def getAction(self, observation, id):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    return UP
                elif event.key == K_DOWN:
                    return DOWN
                elif event.key == K_LEFT:
                    return LEFT
                elif event.key == K_RIGHT:
                    return RIGHT
                elif event.key == K_RETURN:
                    return STAY
        return None

class RandomComputer(Player):
    """docstring for RandomComputer"""
    def __init__(self, gameInstance):
        super(RandomComputer,self).__init__(gameInstance)

    def getAction(self, observation, id):
        if self.agents[id].role == 'prey':
            #STAY with a given probability
            rand = random.random()
            if rand <= 0.8: 
                return STAY
            else: #find actions that don't cause shared position
                adjacent = set([self.action2Tile(UP, self.agents[id].POS),
                                self.action2Tile(DOWN, self.agents[id].POS),
                                self.action2Tile(LEFT, self.agents[id].POS),
                                self.action2Tile(RIGHT, self.agents[id].POS)])
                freeAdjacentTiles = adjacent - set(observation.positions)
                possibleActions = []
                for freeAdjacentTile in freeAdjacentTiles:
                    possibleActions.append(self.tile2Action(self.agents[id].POS, freeAdjacentTile))
                #remaining actions have equal probability
                rand = random.random()
                chance = float(1) / len(possibleActions)
                return possibleActions[int(rand / chance)] 

        if self.agents[id].role == 'predator':
            return random.randint(0,4) 

class PolicyIteration(Player):
    """docstring for PolicyIteration"""
    def __init__(self, gameInstance):
        super(PolicyIteration,self).__init__(gameInstance)
    
    def getAction(self, observation, id):
        return None

class ValueIteration(Player):
    """docstring for ValueIteration"""
    def __init__(self, gameInstance):
        super(ValueIteration,self).__init__(gameInstance)
    
    def getAction(self, observation, id):
        return None

class Qlearning(Player):
    """docstring for Qlearning"""
    def __init__(self, gameInstance):
        super(Qlearning,self).__init__(gameInstance)
        self.QtableFN = 'singlePlayer.p'
        self.training = 0
        self.Qinitval = 2
        self.alpha = 0.5
        self.gamma = 0.7
        self.epsilon = 0.05 
        if not self.training:  # don't select suboptimal actions if not training!
            self.epsilon = 0 # amounts to greedy action selection
        try:
            self.Qtable = pickle.load(open( self.QtableFN, "rb"))
            print 'Qtable found and loaded'
        except:
            print 'Qtable not found, creating new'
            self.Qtable = {}
    
    def getAction(self, observation, id): 
        if id == 0:
            self.shape = tuple([5]*len(self.agents)) # TODO: preferably init only once!
            self.state = self.getStateRep(observation, id)
            print 'state: ', self.state
            possActions = self.Qtable.get(self.state, [self.Qinitval]*5**len(self.agents))
            maxval = max(possActions)

            ind = [i for i, v in enumerate(possActions) if v != maxval]
            if random.random() < self.epsilon and len(ind) != 0:
                #select sub optimal action with eps prob if present
                rand = random.random()
                chance = float(1) / len(ind)
                self.action = ind[int(rand / chance)]
                #print 'returning from suboptimal possActions', ind, 'action', self.action
            else:
                #select from max possActions with 1-eps prob
                ind = [i for i, v in enumerate(possActions) if v == maxval]
                rand = random.random()
                chance = float(1) / len(ind)
                self.action = ind[int(rand / chance)]
                #print 'returning from max possActions', ind, 'action', self.action

            # print 'action: ', self.action
            self.actions = np.unravel_index(self.action, self.shape) #tuple with an action for each agent
            print self.actions[id]
        return self.actions[id]

    def finalize(self, reward, observation, id):
        if id == 0 and self.training == 1:
            #Qlearn
            self.Qtable[self.state] = self.Qtable.get(self.state, [self.Qinitval]*5**len(self.agents))
            newState = self.getStateRep(observation, id)
            self.Qtable[self.state][self.action] = self.Qtable[self.state][self.action] + self.alpha * (reward + self.gamma * max(self.Qtable.get(newState, [self.Qinitval]*5**len(self.agents))) - self.Qtable[self.state][self.action])
            # print 'state: ', self.state, 'action: ', self.action, 'reward: ', reward 

    def quit(self, id):
        if id == 0 and self.training == 1:
            try:
                pickle.dump(self.Qtable, open('agent' + str(self.agents[id].nr) + self.QtableFN, "wb" ), pickle.HIGHEST_PROTOCOL)
            except:
                print self.QtableFN + str(self.agents[id].nr)
                print "agent %s can't write Qtable" % self.agents[id].nr

