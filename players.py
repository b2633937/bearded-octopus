import random, sys, pygame, pickle, numpy as np
from pygame.locals import *
from locals import *
import matplotlib.pyplot as plt
#from pylab import *


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
        self.boardSize = gameInstance.boardSize
        self.gameInstance = gameInstance

    def addAgent(self, agent):
        self.agents.append(agent)
        return len(self.agents)-1

    def getStateRep(self, observation, id):
        state = []
        for i in xrange(len(observation.positions)):
            if i != self.agents[id].nr:
                state.append(self.translatePos(self.agents[id].POS, observation.positions[i], self.boardSize))
        return tuple(state) 

    def action2Tile(self, action, position):
        return ((position[0]+EFFECTS[action][0])%self.boardSize[0], 
            (position[1]+EFFECTS[action][1])%self.boardSize[1])

    def tile2Action(self, basePosition, adjacentTile):
        if adjacentTile == self.action2Tile(UP, basePosition): return UP
        if adjacentTile == self.action2Tile(DOWN, basePosition): return DOWN
        if adjacentTile == self.action2Tile(LEFT, basePosition): return LEFT
        if adjacentTile == self.action2Tile(RIGHT, basePosition): return RIGHT
        if adjacentTile == basePosition: return STAY

    def translatePos(self, ownPos, otherPos, boardSize): 
        #translates other position after centering ownPos 
        center = (boardSize[0]/2, boardSize[1]/2)
        return ((otherPos[0] - ownPos[0] + center[0])%boardSize[0], (otherPos[1] - ownPos[1] + center[1])%boardSize[1])

    def getAction(self, observation): 
        sys.exit("function getAction not implemented for agent " + str(agent.nr))

    def finalize(self, R, observation, id):
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
            rand =random.random()
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
                return possibleActions[int(random.random()*len(possibleActions))] 

        if self.agents[id].role == 'predator':
            return random.randint(0,4) 

class GeneralizedPolicyIteration(Player):
    """Implementation of Policy Iteration and Value Iteration, works for 2 player case when fullfilling the role of predator"""

    def __init__(self, gameInstance):
        super(GeneralizedPolicyIteration,self).__init__(gameInstance)
        Vinitval = 0
        V = np.zeros((self.boardSize[0], self.boardSize[1])) + Vinitval #TODO: only works for 2 player scenario
        threshold = 0.00000001 #TODO: change to sensible value
        gamma = 0.8
        terminalState = (self.boardSize[0]/2, self.boardSize[1]/2)
        policy = {}
        deltas = []
        valueIteration = True #stops after one sweep of evaluation and updates V differently (argmax)
        k = None #number of evaluation sweeps per iteration

        while True:

            #Policy Evaluation
            while k > 0 or k == None:
                delta = 0
                vCopy = V.copy()
                for i in xrange(V.shape[0]): #for each state
                    for j in xrange(V.shape[1]):
                        s = (i, j)
                        v = 0
                        if s != terminalState: #terminal state not part of S
                            policy[s] = policy.get(s, {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2})
                            actionValues = {}
                            for a, p_a in policy[s].items(): #for each action predator considers
                                sPrimes = self.P(s, a)
                                for sPrime, p_sPrime in sPrimes.items():
                                    if valueIteration:
                                        actionValues[a] = actionValues.get(a, 0) + p_a * p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                                    else: 
                                        v += p_a * p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                            if valueIteration:
                                v = max(actionValues.values()) #key diff between Policy Iteration and Value Iteration!
                            delta = max(delta, abs(v - V[s])) 
                            vCopy[s] = v #V[s] = v
                V = vCopy
                if delta < threshold:
                    break

            #Policy Improvement
            policyChanged = False
            for s in policy:
                v = 0
                if s != terminalState: #terminal state not part of S
                    actionValues = {}
                    for a in xrange(5): #for each possible action 
                        sPrimes = self.P(s,a)
                        for sPrime, p_sPrime in sPrimes.items():
                            actionValues[a] = actionValues.get(a, 0) + p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                    greedyActions = [i for i, v in actionValues.items() if 
                    v < max(actionValues.values()) + 0.00000001 and v > max(actionValues.values()) - 0.00000001] # small intervall to cope with numerical errors
                    p = 1/float(len(greedyActions))
                    oldPolicy = set(policy[s].keys()) 
                    policy[s] = {}
                    for a in greedyActions:
                        policy[s][a] = p

                    if oldPolicy != set(policy[s].keys()):
                        policyChanged = True

            if not policyChanged:
                break

            if valueIteration: # Value Iteration just updates policy once
                break

        self.policy = policy


        # fig, sp1 = plt.subplots(figsize=(12,5))
        # sp1.plot(deltas)
        # sp1.set_xlabel('iterations')
        # sp1.set_ylabel('delta')
        # sp1.set_title('Policy iteration convergence using gamma of 0.9');
        # plt.show()

        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        print V
        self.printPolicy(policy)
        sys.exit()


    # Reward function
    def R(self, s): 
        terminalState = (self.boardSize[0]/2, self.boardSize[1]/2)
        if s == terminalState:
            return 10
        else:
            return 0

    # Transition Probabillity function returns a dictionary {s': prob, ...}
    def P(self, s,a): 
        terminalState = (self.boardSize[0]/2, self.boardSize[1]/2)
        if s == terminalState: 
            return {s: 1}
        else:
            halfwayState = ((s[0]+EFFECTS[a][0])%self.boardSize[0], (s[1]+EFFECTS[a][1])%self.boardSize[1])
            if halfwayState == terminalState: #no escape possible
                return {halfwayState: 1}
            else: 
                sPrimes = {}
                for prey_a in xrange(1,5):
                    sPrime =   ((halfwayState[0]+REVEFFECTS[prey_a][0])%self.boardSize[0], 
                                (halfwayState[1]+REVEFFECTS[prey_a][1])%self.boardSize[1])
                    if sPrime != terminalState:
                        sPrimes[sPrime] = None
                prob = 0.2/len(sPrimes) #now we know how many states follow, we know their probs
                for sPrime in sPrimes:
                    sPrimes[sPrime] = prob 
                sPrimes[halfwayState] = 0.8 #prey stays at same location 
                return sPrimes

    def getStateRep(self, observation, id): #@overwrite since we're now interested at state rel to prey
        state = []
        for i in xrange(len(observation.positions)):
            if i == self.agents[id].nr:
                state.append(self.translatePos(self.agents[id].POS, observation.positions[i], self.boardSize))
        return tuple(state) 

    def getAction(self, observation, id):
        state = self.getStateRep(observation, id)[0] # we know state will be ((y,x),)
        actions = self.policy[state] 
        rand = random.random()
        accum = 0
        for a in actions:
            accum += actions[a] 
            if rand <= accum:
                return a

    def printPolicy(self, p):
        for i in xrange(self.boardSize[0]):
            print 
            for j in xrange(self.boardSize[1]):
                if not (i,j) == (5,5):
                    print p[i,j].keys(),
                else:
                    print 0, 
        print 

    def printV(self, V):
        phi_m = linspace(0, 11, 12)
        phi_p = linspace(0, 11, 12)
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0,12,1))
        ax.set_yticks(np.arange(0,12,1))
        plt.grid()

        for i in xrange(11):
            for j in xrange(11):
                v = V[i][j]
                if v == 10: 
                    v = round(v,1) # in order too pretty print
                ax.annotate(str(v),xy=(i+0.25,j+0.35))

        show()

