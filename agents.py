import random, sys, pygame, pickle, numpy as np
from pygame.locals import *
from locals import *
import matplotlib.pyplot as plt
# import pylab
# from pylab import *

""" Contains all the different Brain types"""

class Player(object):
    def __init__(self, agent):
        super(Player, self).__init__()
        self.agent = agent 
        self.role = None 
        # self.img = img 
        # self.pos = None
        self.nr = None # player number in game 
        self.observation = None
        self.id = agent.addPlayer(self) # player number in agent 

    def getAction(self):
        return self.agent.getAction(self.id)

    def observe(self, observation):
            self.observation = observation
            self.nr = observation.playerNr
            self.role = observation.roles[self.nr] 
            self.pos = observation.positions[self.nr]
            self.agent.boardSize = observation.boardSize


    def finalize(self, reward):
        self.agent.finalize(reward, self.id) 

    def quit(self):
        self.agent.quit(self.id)

class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()
        self.players = []
        self.boardSize = None

    def addPlayer(self, player):
        self.players.append(player)
        return len(self.players)-1

    def getStateRep(self, observation, id):
        # self.players[id].pos = observation.positions[self.players[id].nr]
        state = []
        for i in xrange(len(observation.positions)):
            if i != self.players[id].nr:
                state.append(self.translatePos(self.players[id].pos, observation.positions[i], self.boardSize))
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

    def getAction(self): 
        sys.exit("function getAction not implemented for player " + str(player.nr))

    def eGreedy(self, possActions):
        maxval = max(possActions)
        ind = [i for i, v in enumerate(possActions) if v == maxval]
        if random.random() > self.epsilon:  #select optimal action with 1 - eps prob
            action = ind[int(random.random() * len(ind))] 
        else:   #select random action (includes optimal action)
            action = int(random.random() * len(possActions))
        return action

    def finalize(self, R, id):
        pass

    def quit(self, id):
        pass


class MonteCarlo(Agent):
    """Implementation of both On- and Off-Policy Monte-Carlo Control Algorithms"""
    def __init__(self):
        super(MonteCarlo, self).__init__()
        
class Qlearning(Agent):
    """docstring for Qlearning"""
    def __init__(self):
        super(Qlearning,self).__init__()
        self.QtableFN = 'singleAgent.p'
        self.training = 1
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
    
    def getAction(self, id): 
        if id == 0: #player 0 plans for all of the agent's players
            self.shape = tuple([5]*len(self.players)) #TODO: preferably init only once!
            self.state = self.getStateRep(id)
            print 'state: ', self.state
            possActions = self.Qtable.get(self.state, [self.Qinitval]*5**len(self.players))
            self.action = self.eGreedy(possActions)
            self.actions = np.unravel_index(self.action, self.shape) #tuple with an action for each player
        return self.actions[id]

    def finalize(self, R, id):
        if id == 0 and self.training == 1:
            #Qlearn
            self.Qtable[self.state] = self.Qtable.get(self.state, [self.Qinitval]*5**len(self.players))
            newState = self.getStateRep(id)
            self.Qtable[self.state][self.action] = self.Qtable[self.state][self.action] + self.alpha * (R + self.gamma * max(self.Qtable.get(newState, [self.Qinitval]*5**len(self.players))) - self.Qtable[self.state][self.action])
            # print 'state: ', self.state, 'action: ', self.action, 'R: ', R 

    def quit(self, id):
        if id == 0 and self.training == 1:
            try:
                pickle.dump(self.Qtable, open('player' + str(self.players[id].nr) + self.QtableFN, "wb" ), pickle.HIGHEST_PROTOCOL)
            except:
                print self.QtableFN + str(self.players[id].nr)
                print "player %s can't write Qtable" % self.players[id].nr

class Human(Agent):
    """docstring for Human"""
    def __init__(self):
        super(Human, self).__init__()
    
    def getAction(self, id):
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

class RandomComputer(Agent):
    """docstring for RandomComputer"""
    def __init__(self):
        super(RandomComputer,self).__init__()

    def getAction(self, id):
        if self.players[id].role == PREY:
            #STAY with a given probability
            rand =random.random()
            if rand <= 0.8: 
                return STAY
            else: #find actions that don't cause shared position
                adjacent = set([self.action2Tile(UP, self.players[id].pos),
                                self.action2Tile(DOWN, self.players[id].pos),
                                self.action2Tile(LEFT, self.players[id].pos),
                                self.action2Tile(RIGHT, self.players[id].pos)])
                freeAdjacentTiles = adjacent - set(self.players[id].observation.positions)
                possibleActions = []
                for freeAdjacentTile in freeAdjacentTiles:
                    possibleActions.append(self.tile2Action(self.players[id].pos, freeAdjacentTile))
                #remaining actions have equal probability
                return possibleActions[int(random.random()*len(possibleActions))] 

        if self.players[id].role == PREDATOR:
            return random.randint(0,4) 

class GeneralizedPolicyIteration(Agent):
    """Implementation of Policy Iteration and Value Iteration, works for 2 agent case when fullfilling the role of predator"""

    def __init__(self):
        super(GeneralizedPolicyIteration,self).__init__()
        Vinitval = 0
        V = np.zeros((self.boardSize[0], self.boardSize[1])) + Vinitval #TODO: only works for 2 agent scenario
        threshold = 0.001 #0.00000001 #TODO: change to sensible value
        gamma = 0.8
        terminalState = (self.boardSize[0]/2, self.boardSize[1]/2)
        policy = {}
        deltas = []
        valueIteration = False #stops after one sweep of evaluation and updates V differently (argmax)
        k = None #number of evaluation sweeps per iteration

        
        sweeps = 0 #testing Policy Iteration
        deltas = [] #testing Policy Iteration

        while True:

            #Policy Evaluation
            while k > 0 or k == None:
                sweeps +=1
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
                                        actionValues[a] = actionValues.get(a, 0) + p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                                    else: 
                                        v += p_a * p_sPrime * (self.R(sPrime) + gamma * V[sPrime])
                            if valueIteration:
                                v = max(actionValues.values()) #key diff between Policy Iteration and Value Iteration!
                            delta = max(delta, abs(v - V[s])) 
                            vCopy[s] = v #V[s] = v
                deltas.append(delta) #testing Policy Iteration
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

            if not policyChanged or valueIteration:
                break

        self.policy = policy

        #Some printing for the assignments

        # print deltas #testing Policy Iteration
        # print sweeps #testing Policy Iteration

        # fig, sp1 = plt.subplots(figsize=(12,5))
        # sp1.plot(deltas)
        # sp1.set_xlabel('iterations')
        # sp1.set_ylabel('delta')
        # # sp1.set_title('Policy Iteration convergence using gamma of 0.9')
        # pylab.xlim([0,60])
        # pylab.ylim([0,10])
        # plt.show()

        # np.set_printoptions(precision=2)
        # np.set_printoptions(suppress=True)
        # print V
        
        # self.printPolicy(policy)
        # self.latexPrintV(V)
        # sys.exit()


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

    def getStateRep(self, id): #@overwrite since we're now interested at state rel to prey
        self.players[id].pos = observation.positions[self.players[id].nr]
        state = []
        for i in xrange(len(self.players[id].observation.positions)):
            if i != self.players[id].nr:
                state.append(self.translatePos(self.players[id].observation.positions[i], self.players[id].pos, self.boardSize))
        return tuple(state) 

    def getAction(self, id):
        state = self.getStateRep(id)[0] # we know state will be ((y,x),)
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

    def latexPrintV(self, V):
        print '\\begin{center}'
        print '\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}' 
        print '\\hline'
        for i in xrange(11):
            for j in xrange(11):
                if j != 0: 
                    print '&',
                v = V[i][j]
                temp = round(v, 2)
                print  temp, 
            print '\\\ \\hline' 
        print '\\end{tabular}'
        print '\\end{center}'

