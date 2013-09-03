import random, sys, pygame, pickle, numpy as np
from pygame.locals import *
from locals import *

BOARDSIZE = 11 # always uneven in order for state to work!

def action2Tile(action, position):
    X,Y = position
    if action == UP:
        Y = Y-1 if Y > 0 else BOARDSIZE-1
    elif action == DOWN:
        Y = Y+1 if Y < BOARDSIZE-1 else 0
    elif action == LEFT:
        X = X-1 if X > 0 else BOARDSIZE-1
    elif action == RIGHT:
        X = X+1 if X < BOARDSIZE-1 else 0
    elif action == STAY:
        pass
    else: 
        sys.exit('ERROR: unknown action: ' +  str(action))
    return (X,Y)

def tile2Action(basePosition, adjacentTile):
    if adjacentTile == action2Tile(UP, basePosition): return UP
    if adjacentTile == action2Tile(DOWN, basePosition): return DOWN
    if adjacentTile == action2Tile(LEFT, basePosition): return LEFT
    if adjacentTile == action2Tile(RIGHT, basePosition): return RIGHT
    if adjacentTile == basePosition: return STAY

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
    @staticmethod
    def new(playerType):
        if playerType == 'Human':
            return Human()
        elif playerType == 'RandomComputer':
            return RandomComputer()
        elif playerType == 'Optimizer':
            return Optimizer()
        elif playerType == 'Qlearning':
            return Qlearning()
        else:
            sys.exit('ERROR: unknown player type: ' + playerType)

    def __init__(self):
        super(Player, self).__init__()
        self.agents = []

    def addAgent(self, agent):
        self.agents.append(agent)
        return len(self.agents)-1

    def getState(self, observation, id):
        state = []
        for i in xrange(len(observation.positions)):
            if i != self.agents[id].nr:
                state.append(self.translatePos(self.agents[id].POS, observation.positions[i])) 
        return tuple(state) 

    def translatePos(self, ownPos, otherPos):
        #translates other position relative to centered ownPos 
        center = (BOARDSIZE/2, BOARDSIZE/2)
        translation = (center[0] - ownPos[0], center[1] - ownPos[1])
        translOtherPos = [translation[0] + otherPos[0], translation[1] + otherPos[1]]
        if translOtherPos[0] < 0:
            translOtherPos[0] = translOtherPos[0] + BOARDSIZE
        elif translOtherPos[0] >= BOARDSIZE:
            translOtherPos[0] = translOtherPos[0] - BOARDSIZE
        if translOtherPos[1] < 0:
            translOtherPos[1] = translOtherPos[1] + BOARDSIZE
        elif translOtherPos[1] >= BOARDSIZE:
            translOtherPos[1] = translOtherPos[1] - BOARDSIZE
        return (translOtherPos[0] - center[0], translOtherPos[1] - center[1])

    def translatePosOld(self, ownPos, otherPos): #TODO: delete
        #translates other position relative to centered ownPos 
        center = (BOARDSIZE/2, BOARDSIZE/2)
        translation = (center[0] - ownPos[0], center[1] - ownPos[1])
        translOtherPos = [translation[0] + otherPos[0], translation[1] + otherPos[1]]
        if translOtherPos[0] < 0:
            translOtherPos[0] = translOtherPos[0] + BOARDSIZE
        elif translOtherPos[0] >= BOARDSIZE:
            translOtherPos[0] = translOtherPos[0] - BOARDSIZE
        if translOtherPos[1] < 0:
            translOtherPos[1] = translOtherPos[1] + BOARDSIZE
        elif translOtherPos[1] >= BOARDSIZE:
            translOtherPos[1] = translOtherPos[1] - BOARDSIZE
        return tuple(translOtherPos)

    def getAction(self, observation): 
        sys.exit("function getAction not implemented for agent " + str(agent.nr))

    def finalize(self, reward, observation, id):
        pass

    def quit(self, id):
        pass


class Optimizer(Player):
    """ does not optimize tactically
        TODO: minimize or maximize to mulitple targets
    """
    def __init__(self):
        super(Optimizer,self).__init__() 

    def reverseAction(self, action):
        if action == UP:
            return DOWN
        elif action == DOWN:
            return UP        
        elif action == LEFT:
            return RIGHT
        elif action == RIGHT:
            return LEFT

    def getAction(self, observation, id):
        #TODO: FIX should minimize distance alog all other agents.
        #at the moment minimizes over the min of all X coords and same for y
        state = self.getState(observation, id)
        print state
        X = [abs(x[0]) for x in state]
        x = state[X.index(min(X))][0]
        Y = [abs(y[1]) for y in state]
        y = state[Y.index(min(Y))][1]
        print x,y

        # minimize distance 
        if x == y:
            if random.randint(0,1):
                if x > 0:
                    action = RIGHT
                else:
                    action = LEFT
            else:
                if y > 0:
                    action = DOWN
                else:
                    action = UP
        elif x > y:
            if x > 0:
                action = RIGHT
            else:
                action = LEFT
        elif x < y:
            if y > 0:
                action = DOWN
            else:
                action = UP

        # maximize distance        
        if self.agents[id].role == 'prey':
            action = self.reverseAction(action)
        return action


class Human(Player):
    def __init__(self):
        super(Human,self).__init__()
    
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
    def __init__(self):
        super(RandomComputer,self).__init__()

    def getAction(self, observation, id):

        if self.agents[id].role == 'prey':
            #STAY with a given probability
            rand = random.random()
            if rand <= 0.8: #TODO: change to 0.8!
                return STAY
            else: #find actions that don't cause shared position
                adjacent = set([action2Tile(UP, self.agents[id].POS),
                                action2Tile(DOWN, self.agents[id].POS),
                                action2Tile(LEFT, self.agents[id].POS),
                                action2Tile(RIGHT, self.agents[id].POS)])
                freeAdjacentTiles = adjacent - set(observation.positions)
                possibleActions = []
                for freeAdjacentTile in freeAdjacentTiles:
                    possibleActions.append(tile2Action(self.agents[id].POS, freeAdjacentTile))
                #remaining actions have equal probability
                rand = random.random()
                chance = float(1) / len(possibleActions)
                return possibleActions[int(rand / chance)] 

        if self.agents[id].role == 'predator':
            adjacent = set([action2Tile(UP, self.agents[id].POS),
                            action2Tile(DOWN, self.agents[id].POS),
                            action2Tile(LEFT, self.agents[id].POS),
                            action2Tile(RIGHT, self.agents[id].POS)])
            freeAdjacentTiles = adjacent - set(observation.positions)
            possibleActions = []
            for freeAdjacentTile in freeAdjacentTiles:
                possibleActions.append(tile2Action(self.agents[id].POS, freeAdjacentTile))
            possibleActions.append(STAY)
            rand = random.random()
            chance = float(1) / len(possibleActions)
            return possibleActions[int(rand / chance)] 


class Qlearning(Player):
    """docstring for Qlearning"""
    def __init__(self):
        super(Qlearning,self).__init__()
        self.QtableFN = 'singlePlayer.p'
        self.Qinitval = 2
        self.alpha = 0.5
        self.gamma = 0.7
        self.training = 0
        if self.training:
            self.epsilon = 0 
        else:
            self.epsilon = 0.05 # TODO: change into 0.05'ish
        try:
            self.Qtable = pickle.load(open( self.QtableFN, "rb"))
            print 'Qtable found and loaded'
        except:
            print 'Qtable not found, creating new'
            self.Qtable = {}
    
    def getAction(self, observation, id): 
        if id == 0:
            self.shape = tuple([5]*len(self.agents)) # TODO: preferably init only once!
            # print 'shape: ', self.shape
            # print 'nro agents controlled by player: ', len(self.agents)
            # print [self.Qinitval]*5**len(self.agents)
            self.state = self.getState(observation, id)
            print 'state: ', self.state
            possActions = self.Qtable.get(self.state, [self.Qinitval]*5**len(self.agents))
            maxval = max(possActions)

            # TODO: do not select suboptimal actions if not training!
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
            newState = self.getState(observation, id)
            self.Qtable[self.state][self.action] = self.Qtable[self.state][self.action] + self.alpha * (reward + self.gamma * max(self.Qtable.get(newState, [self.Qinitval]*5**len(self.agents))) - self.Qtable[self.state][self.action])
            # print 'state: ', self.state, 'action: ', self.action, 'reward: ', reward 

    def quit(self, id):
        if id == 0 and self.training == 1:
            print 'qquitting'
            try:
                pickle.dump(self.Qtable, open('agent' + str(self.agents[id].nr) + self.QtableFN, "wb" ), pickle.HIGHEST_PROTOCOL)
            except:
                print self.QtableFN + str(self.agents[id].nr)
                print "agent %s can't write Qtable" % self.agents[id].nr

