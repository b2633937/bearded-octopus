import random, sys, pygame
from pygame.locals import *
from locals import *

BOARDSIZE = 11 # always uneven in order for state to work!

def action2Tile(action, position):
    #returns same tile if action is not in {0,1,2,3}
    #STAY is thus returned by this principle rather than explicitly checked!
    X,Y = position
    if action == UP:
        Y = Y-1 if Y > 0 else BOARDSIZE-1
    elif action == DOWN:
        Y = Y+1 if Y < BOARDSIZE-1 else 0
    elif action == LEFT:
        X = X-1 if X > 0 else BOARDSIZE-1
    elif action == RIGHT:
        X = X+1 if X < BOARDSIZE-1 else 0
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
        self.agent.quit(self.id)


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

    agents = []

    def __init__(self):
        super(Player, self).__init__()

    def addAgent(self, agent):
        self.agents.append(agent)
        return len(self.agents)-1

    def getState(self, observation, id):
        state = []
        for i in xrange(len(observation.positions)):
            if i != id:
                state.append(self.translatePos(self.agents[id].POS, observation.positions[i])) 
        return state 


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
        # return STAY
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

class Qlearning(Player):
    """docstring for Qlearning"""
    def __init__(self):
        super(Qlearning,self).__init__()
        self.QtableFN = 'Qtable.p'
        self.Qinitval = [0,0,0,0,0]#[10, 10, 10, 10, 10] 
        self.epsilon = 0.05 # TODO: change into 0.05'ish
        self.alpha = 0.3
        self.gamma = 0.9
        try:
            self.Qtable = pickle.load(open( self.QtableFN, "rb"))
            print 'Qtable found and loaded'
        except:
            print 'Qtable not found, creating new'
            self.Qtable = {}
    
    def getAction(self, observation, id): #Qinitval name in AA?
        self.state = self.getState(observation)
        actions = self.Qtable.get(self.state, list(self.Qinitval))
        maxval = max(actions)

        ind = [i for i, v in enumerate(actions) if v != maxval]
        if random.random() < self.epsilon and len(ind) != 0:
            #select sub optimal action with eps prob if present
            rand = random.random()
            chance = float(1) / len(ind)
            self.action = ind[int(rand / chance)]
            #print 'returning from suboptimal actions', ind, 'action', self.action
        else:
            #select from max actions with 1-eps prob
            ind = [i for i, v in enumerate(actions) if v == maxval]
            rand = random.random()
            chance = float(1) / len(ind)
            self.action = ind[int(rand / chance)]
            #print 'returning from max actions', ind, 'action', self.action
        return self.action

    def finalize(self, reward, observation, id):
        #Qlearn
        self.Qtable[self.state] = self.Qtable.get(self.state, list(self.Qinitval))
        newState = self.getState(observation)
        self.Qtable[self.state][self.action] = self.Qtable[self.state][self.action] + self.alpha * (reward + self.gamma * max(self.Qtable.get(newState, list(self.Qinitval))) - self.Qtable[self.state][self.action])
        # print 'state: ', self.state, 'action: ', self.action, 'reward: ', reward 

    def quit(self, id):
        try:
            pickle.dump(self.Qtable, open( self.QtableFN, "wb" ), pickle.HIGHEST_PROTOCOL)
        except:
            print "agent %s can't write Qtable" % self.pID